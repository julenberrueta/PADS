"""High-level orchestration: 4 public steps composing the data/model layers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pads import tracking
from pads.config import PADSConfig, TestType
from pads.data import loader, normalizer
from pads.data.preprocess import preprocess
from pads.data.schema import (
    IMPUTE_FIRST_ROW,
    N_TIME_OFFSETS,
    validate_dataset,
)
from pads.data.windowing import (
    build_discharge_windows,
    build_rolling_windows,
    disch_label_series,
    disch_outcomes_by_stay,
    stay_to_mortality_outcome,
)
from pads.eval.errors import compute_errors
from pads.eval.metrics import evaluate
from pads.eval.thresholds import ThresholdMethod
from pads.models.registry import load_model, load_or_create_model, save_model
from pads.training import trainer
from pads.training.splits import (
    build_xy_discharge,
    build_xy_rolling,
    prepare_train_val_test,
    split_stays,
)
from pads.viz.plots import plot_error, plot_roc_combined


class PADSPipeline:
    """Composes the package modules into the 5-step PADS workflow."""

    def __init__(self, config: PADSConfig):
        self.config = config
        self.config.ensure_dirs()
        trainer.set_global_seed(config.seed)
        trainer.configure_gpu()

    def _common_tags(self, data_filename: str | None = None) -> dict[str, str]:
        """Run tags shared across steps, including dataset provenance.

        When a `data_filename` is given (inference), the dataset is hashed
        directly. When it is omitted (retrain/metrics, which only see the
        processed .pkl files), the source dataset name + hash are recovered
        from the provenance file written by `prepare_data`, so every run still
        records which raw dataset its inputs came from.
        """
        tags = {
            "retrain_type": self.config.retrain_type,
            "test_type": self.config.test_type,
            "seed": self.config.seed,
        }
        if data_filename is not None:
            data_path = self._data(data_filename)
            if data_path.is_file():
                tags["dataset"] = data_filename
                tags["dataset_sha256"] = tracking.file_sha256(data_path)[:16]
        elif self._provenance_path().is_file():
            info = loader.load_json(self._provenance_path())
            tags["dataset"] = info["dataset"]
            tags["dataset_sha256"] = info["dataset_sha256"][:16]
            if "n_train_stays" in info:
                tags["n_train_stays"] = info["n_train_stays"]
                tags["n_test_stays"] = info["n_test_stays"]
        return tags

    # --- path helpers -------------------------------------------------------
    def _data(self, name: str) -> Path:
        """Raw input dataset lives directly in data/."""
        return self.config.data_dir / name

    def _processed(self, name: str) -> Path:
        """Generated intermediate artifacts live in data/processed/."""
        return self.config.processed_dir / name

    def _provenance_path(self) -> Path:
        """JSON recording which raw dataset produced the processed files."""
        return self._processed("source_dataset.json")

    def _model(self, name: str) -> Path:
        return self.config.model_dir / name

    def _norm(self, name: str) -> Path:
        return self.config.norm_dir / name

    def _results(self, *parts: str) -> Path:
        # Per-model subfolder (results/<retrain_type>/...) so sweeps don't collide.
        return self.config.run_results_dir.joinpath(*parts)

    def _load_test_thresholds(self) -> tuple[float, float]:
        """Decision thresholds from the test split (calculate_metrics step).

        These are the leakage-free thresholds inference must apply. The job runs
        calculate_metrics before inference, so the file normally exists; if it is
        missing (e.g. an inference-only call) we fail loudly rather than silently
        recomputing on the inference data.
        """
        path = self._processed("model_parameters_test.json")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}: run the 'calculate_metrics' step before "
                f"inference so the test-split thresholds are available."
            )
        params = loader.load_json(path)
        return float(params["th_mort"]), float(params["th_disch"])

    # --- step 1: prepare all data artifacts ---------------------------------
    def prepare_data(self, data_filename: str) -> None:
        """Build every artifact the retraining needs from the raw dataset.

        Merges the former ``generate_files`` + ``generate_retrain_data`` steps:
        median imputation values, the train/test stay split, the fitted
        normalizers, and the windowed mortality/discharge ``.pkl`` datasets.
        Finally records dataset provenance so later retrain/metrics runs (which
        only load the .pkl files) can report which raw dataset produced them.
        """
        data_path = self._data(data_filename)
        df = loader.load_raw_dataset(data_path)

        # medians
        medians = df.loc[df["hr"] <= N_TIME_OFFSETS, IMPUTE_FIRST_ROW].median().to_dict()
        loader.save_json(medians, self._processed("medians_48h.json"))

        # train/test stay split
        eligible = df.loc[df["los"] >= N_TIME_OFFSETS, "stay_id"].unique().tolist()
        train_ids, test_ids = split_stays(eligible, seed=self.config.seed)
        loader.save_stays(train_ids, self._processed("train_stays.txt"))
        loader.save_stays(test_ids, self._processed("test_stays.txt"))

        # fit normalizers on the training stays
        train_data = self._build_window_dataset(data_filename, split="train", medians=medians)
        self._fit_mortality_normalizer(train_data["data"])
        self._fit_discharge_normalizer(train_data["data"])

        # windowed retrain/test datasets for both models
        for split in ("train", "test"):
            self._build_mortality_dataset(data_filename, split, medians)
            self._build_discharge_dataset(data_filename, split, medians)

        # provenance: which raw dataset produced the processed files above
        self._write_provenance(data_path)

    def _write_provenance(self, data_path: Path) -> None:
        """Record the source dataset (name + full hash + timestamp + split sizes)."""
        n_train = len(loader.load_stays(self._processed("train_stays.txt")))
        n_test = len(loader.load_stays(self._processed("test_stays.txt")))
        loader.save_json(
            {
                "dataset": data_path.name,
                "dataset_sha256": tracking.file_sha256(data_path),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_train_stays": n_train,
                "n_test_stays": n_test,
            },
            self._provenance_path(),
        )

    # --- step 2: retrain models --------------------------------------------
    def retrain_models(self) -> None:
        if self.config.retrain_type == "original":
            raise ValueError(
                "retrain_type='original' is evaluation-only (the shipped base "
                "model); it cannot be retrained. Use it with calculate_metrics "
                "and inference to baseline the off-the-shelf model instead."
            )
        # New pipeline iteration -> start with a fresh parent linkage.
        tracking.clear_parent_run_id(self.config.base_path)
        rt = self.config.retrain_type
        with tracking.run(f"retrain_models_{rt}", step="retrain_models", **self._common_tags()) as r:
            if r is not None:
                tracking.save_parent_run_id(r.info.run_id, self.config.base_path)
            tracking.log_params(self.config.model_dump(exclude={"base_path"}))
            tracking.log_artifact(self._provenance_path(), artifact_path="dataset")
            self.retrain_mortality()
            self.retrain_discharge()

    def retrain_mortality(self) -> None:
        rt = self.config.retrain_type
        with tracking.run(f"retrain_mortality_{rt}", model="mortality", **self._common_tags()):
            data = loader.load_pkl(self._processed("lstm_last_48h_train.pkl"))
            outcome = loader.load_pkl(self._processed("icu_expire_flag_train.pkl"))

            stay_ids = list(data.keys())
            train_ids, test_ids = split_stays(stay_ids, seed=self.config.seed)
            X_train, y_train = build_xy_rolling(data, outcome, train_ids)
            X_test, y_test = build_xy_rolling(data, outcome, test_ids)

            X_train = normalizer.fit_or_load_and_transform(
                X_train, self._norm(self.config.mort_normalizer), load_existing=True, save=False,
            )
            X_test = normalizer.fit_or_load_and_transform(
                X_test, self._norm(self.config.mort_normalizer), load_existing=True, save=False,
            )

            X_tr, y_tr, X_te, y_te, X_val, y_val = prepare_train_val_test(
                X_train, X_test, y_train, y_test, seed=self.config.seed,
            )
            X_val_all = np.concatenate([X_te, X_val])
            y_val_all = np.concatenate([y_te, y_val])

            from pads.models.mortality import compile_mortality_model

            model = load_or_create_model(
                self.config.retrain_type, "mortality",
                self._model(self.config.retrain_mort_model), X_tr.shape,
            )
            model = compile_mortality_model(model, self.config.learning_rate_mort)
            trainer.fit(
                model, "RETRAIN_MORT", X_tr, y_tr, X_val_all, y_val_all,
                kind="mortality",
                log_dir=self._model("tensorflow_logs"),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                early_stopping_patience=self.config.early_stopping_patience,
            )
            # Per-epoch metrics are streamed by MLflowEpochLogger inside trainer.fit.
            model_path = self._model(self.config.inference_mort_model)
            save_model(model, model_path)
            tracking.log_artifact(model_path, artifact_path="mortality_model")

    def retrain_discharge(self) -> None:
        rt = self.config.retrain_type
        with tracking.run(f"retrain_discharge_{rt}", model="discharge", **self._common_tags()):
            data = loader.load_pkl(self._processed("lstm_disch_3point_48h_train.pkl"))
            outcome = loader.load_pkl(self._processed("outcome_disch_3point_48h_train.pkl"))

            stay_ids = [s for s in data if s in outcome]
            train_ids, test_ids = split_stays(stay_ids, seed=self.config.seed)
            X_train, y_train = build_xy_discharge(data, outcome, train_ids)
            X_test, y_test = build_xy_discharge(data, outcome, test_ids)

            X_train = normalizer.fit_or_load_and_transform(
                X_train, self._norm(self.config.disch_normalizer), load_existing=True, save=False,
            )
            X_test = normalizer.fit_or_load_and_transform(
                X_test, self._norm(self.config.disch_normalizer), load_existing=True, save=False,
            )

            X_tr, y_tr, X_te, y_te, X_val, y_val = prepare_train_val_test(
                X_train, X_test, y_train, y_test, seed=self.config.seed,
            )
            X_val_all = np.concatenate([X_te, X_val])
            y_val_all = np.concatenate([y_te, y_val])

            from pads.models.discharge import compile_discharge_model

            model = load_or_create_model(
                self.config.retrain_type, "discharge",
                self._model(self.config.retrain_disch_model), X_tr.shape,
            )
            model = compile_discharge_model(model, self.config.learning_rate_disch)
            trainer.fit(
                model, "RETRAIN_DISCH", X_tr, y_tr, X_val_all, y_val_all,
                kind="discharge",
                log_dir=self._model("tensorflow_logs"),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                early_stopping_patience=self.config.early_stopping_patience,
            )
            # Per-epoch metrics are streamed by MLflowEpochLogger inside trainer.fit.
            model_path = self._model(self.config.inference_disch_model)
            save_model(model, model_path)
            tracking.log_artifact(model_path, artifact_path="discharge_model")

    # --- step 3: metrics on the test split ---------------------------------
    def calculate_metrics(self, *, threshold_method: ThresholdMethod = "min_distance") -> None:
        rt = self.config.retrain_type
        parent_id = tracking.load_parent_run_id(self.config.base_path)
        with tracking.run(
            f"calculate_metrics_{rt}",
            parent_run_id=parent_id,
            step="calculate_metrics",
            **self._common_tags(),
        ):
            disch_ds = {
                "data": loader.load_pkl(self._processed("lstm_disch_3point_48h_test.pkl")),
                "disch_outcome": loader.load_pkl(self._processed("outcome_disch_3point_48h_test.pkl")),
            }
            mort_ds = {
                "data": loader.load_pkl(self._processed("lstm_last_48h_test.pkl")),
                "mortality_outcome": loader.load_pkl(self._processed("icu_expire_flag_test.pkl")),
            }

            mort_out = self._predict_mortality(mort_ds, self.config.inference_mort_model)
            disch_out = self._predict_discharge(disch_ds, self.config.inference_disch_model)

            mort_pred = mort_out["y_pred"][:, 1]
            mort_gt = mort_out["y_true"][:, 0]
            disch_pred = disch_out["y_pred"][:, 1]
            disch_gt = np.concatenate(disch_out["y_true"], axis=0).ravel()

            th_mort, th_disch = plot_roc_combined(
                mort_pred, mort_gt, disch_pred, disch_gt,
                out_dir=self._results("images"),
                inference_type=None, threshold_method=threshold_method, save=True,
            )
            m_metrics = evaluate(mort_gt, mort_pred, th_mort)
            d_metrics = evaluate(disch_gt, disch_pred, th_disch)
            tracking.log_metrics({
                "test/mort_auc": m_metrics.auc,
                "test/mort_f1": m_metrics.f1,
                "test/mort_precision": m_metrics.precision,
                "test/mort_recall": m_metrics.recall,
                "test/disch_auc": d_metrics.auc,
                "test/disch_f1": d_metrics.f1,
                "test/disch_precision": d_metrics.precision,
                "test/disch_recall": d_metrics.recall,
                "test/threshold_mort": float(th_mort),
                "test/threshold_disch": float(th_disch),
            })
            params = {
                "th_mort": float(th_mort),
                "th_disch": float(th_disch),
                "min_prob": float(mort_pred.min()),
                "max_prob": float(mort_pred.max()),
            }
            params_path = self._processed("model_parameters_test.json")
            loader.save_json(params, params_path)
            tracking.log_artifact(params_path, artifact_path="metrics")
            tracking.log_artifact(self._results("images", "roc_combined.png"),
                                  artifact_path="metrics")

            # Per-sample test predictions, so the app can redraw the ROC in JS
            # (long format: mortality and discharge test sets differ in length).
            # Logged as both CSV (human-readable) and Parquet (the app reads this
            # — smaller download + faster parse).
            metrics_df = pd.DataFrame({
                "model": ["mortality"] * len(mort_pred) + ["discharge"] * len(disch_pred),
                "prob": np.concatenate([mort_pred, disch_pred]),
                "gt": np.concatenate([mort_gt, disch_gt]),
            })
            metrics_df.to_csv(self._results("results_metrics.csv"), index=False)
            metrics_df.to_parquet(self._results("results_metrics.parquet"), index=False)
            tracking.log_artifact(self._results("results_metrics.csv"), artifact_path="metrics")
            tracking.log_artifact(self._results("results_metrics.parquet"), artifact_path="metrics")

    # --- step 4: inference + errors ----------------------------------------
    def run_inference(
        self,
        data_filename: str,
        *,
        test_type: TestType | None = None,
        threshold_method: ThresholdMethod = "min_distance",
    ) -> pd.DataFrame:
        test_type = test_type or self.config.test_type
        rt = self.config.retrain_type
        parent_id = tracking.load_parent_run_id(self.config.base_path)
        with tracking.run(
            f"inference_{rt}_{test_type}",
            parent_run_id=parent_id,
            step="inference",
            **self._common_tags(data_filename),
        ):
            tracking.log_params({"threshold_method": threshold_method,
                                 "test_type_active": test_type})
            validate_dataset(self._data(data_filename))

            medians = loader.load_json(self._processed("medians_48h.json"))
            dataset = self._build_inference_dataset(data_filename, test_type, medians)

            disch_out = self._predict_discharge(dataset, self.config.inference_disch_model)
            mort_out = self._predict_mortality(dataset, self.config.inference_mort_model)

            mort_pred = mort_out["y_pred"][:, 1]
            mort_gt = mort_out["y_true"][:, 0]
            disch_pred = disch_out["y_pred"][:, 1]
            disch_gt = np.hstack(disch_out["y_true"])

            # Reuse the decision thresholds chosen on the test split (the run that
            # produced roc_combined.png). Picking them from the inference data would
            # leak its labels and inflate the metrics, so we never recompute here.
            th_mort, th_disch = self._load_test_thresholds()
            plot_roc_combined(
                mort_pred, mort_gt, disch_pred, disch_gt,
                out_dir=self._results("images"),
                inference_type=test_type, threshold_method=threshold_method,
                fixed_thresholds=(th_mort, th_disch), save=True,
            )
            m_metrics = evaluate(mort_gt, mort_pred, th_mort)
            d_metrics = evaluate(disch_gt, disch_pred, th_disch)
            params = {
                "th_mort": float(th_mort),
                "th_disch": float(th_disch),
                "min_prob": float(mort_pred.min()),
                "max_prob": float(mort_pred.max()),
            }
            params_file = self._results("model_parameters_inference.json")
            loader.save_json(params, params_file)

            errors = compute_errors(
                self._data(data_filename), dataset,
                mort_pred, mort_gt, disch_pred, disch_gt,
                params, out_path=self._results("results_inference.csv"),
            )
            # Parquet mirror of results_inference.csv — the app reads this for the
            # ROC/error charts (smaller download + faster parse than the CSV).
            errors.to_parquet(self._results("results_inference.parquet"), index=False)
            plot_error(errors, out_dir=self._results("images"))

            tracking.log_metrics({
                f"inf/{test_type}/mort_auc": m_metrics.auc,
                f"inf/{test_type}/mort_f1": m_metrics.f1,
                f"inf/{test_type}/mort_precision": m_metrics.precision,
                f"inf/{test_type}/mort_recall": m_metrics.recall,
                f"inf/{test_type}/disch_auc": d_metrics.auc,
                f"inf/{test_type}/disch_f1": d_metrics.f1,
                f"inf/{test_type}/disch_precision": d_metrics.precision,
                f"inf/{test_type}/disch_recall": d_metrics.recall,
                f"inf/{test_type}/mean_error": float(errors["error"].mean()),
                f"inf/{test_type}/critical_error_rate": float((errors["error"] == 3).mean()),
            })
            tracking.log_artifact(params_file, artifact_path="inference")
            tracking.log_artifact(self._results("results_inference.csv"), artifact_path="inference")
            tracking.log_artifact(self._results("results_inference.parquet"), artifact_path="inference")
            for img in ("roc_combined", "barplot_error", "heatmap_error"):
                p = self._results("images", f"{img}_{test_type}.png" if img == "roc_combined" else f"{img}.png")
                tracking.log_artifact(p, artifact_path="inference")
            return errors

    # --- internal: prediction wrappers --------------------------------------
    def _predict_mortality(self, dataset: dict, model_filename: str) -> dict:
        tmp_x = [dataset["data"][sid][:, ::-1, 1:] for sid in tqdm(dataset["data"])]
        tmp_y = [
            np.full((dataset["data"][sid].shape[0], 1), dataset["mortality_outcome"][sid])
            for sid in dataset["data"]
        ]
        X = np.vstack(tmp_x)
        y = np.vstack(tmp_y)
        X = normalizer.fit_or_load_and_transform(
            X, self._norm(self.config.mort_normalizer), load_existing=True, save=False,
        )
        X = np.nan_to_num(X)
        model = load_model(self._model(model_filename))
        y_pred = self._ensure_finite_predictions(model.predict(X), model_filename)
        return {"y_true": y, "y_pred": y_pred}

    def _predict_discharge(self, dataset: dict, model_filename: str) -> dict:
        X = np.vstack(list(dataset["data"].values()))[:, :, 1:]
        y = [np.array(item) for item in dataset["disch_outcome"].values()]
        X = normalizer.fit_or_load_and_transform(
            X, self._norm(self.config.disch_normalizer), load_existing=True, save=False,
        )
        X = np.nan_to_num(X)
        model = load_model(self._model(model_filename))
        y_pred = self._ensure_finite_predictions(model.predict(X), model_filename)
        return {"y_true": y, "y_pred": y_pred}

    @staticmethod
    def _ensure_finite_predictions(y_pred: np.ndarray, model_filename: str) -> np.ndarray:
        """Fail clearly if a model emits NaN/Inf predictions (diverged weights).

        Without this, the non-finite scores reach sklearn's roc_curve and raise
        an opaque 'Input contains NaN'. A saved model with NaN weights means its
        training diverged — see trainer.assert_finite_weights.
        """
        n_bad = int((~np.isfinite(y_pred)).sum())
        if n_bad:
            raise ValueError(
                f"Model '{model_filename}' produced {n_bad}/{y_pred.size} non-finite "
                f"predictions — its weights diverged during training (NaN). Retrain it "
                f"with real/larger data or a different --retrain_type; the synthetic "
                f"dataset is for smoke tests only."
            )
        return y_pred

    # --- internal: dataset builders -----------------------------------------
    def _build_window_dataset(self, data_filename: str, split: str, medians: dict) -> dict:
        df = loader.load_raw_dataset(self._data(data_filename))
        train_stays = loader.load_stays(self._processed(f"{split}_stays.txt"))
        df = df[df["stay_id"].isin(train_stays)]
        df = df[df["los"] >= N_TIME_OFFSETS].copy()
        df["disch_48h"] = disch_label_series(df)
        df = preprocess(df, medians)

        data = build_rolling_windows(df)
        return {
            "data": data,
            "mortality_outcome": stay_to_mortality_outcome(df),
            "disch_outcome": disch_outcomes_by_stay(df),
        }

    def _fit_mortality_normalizer(self, data: dict[int, np.ndarray]) -> None:
        # mortality normalizer is fit on time-reversed rolling windows (drop stay_id col)
        X = np.vstack([arr[:, ::-1, 1:] for arr in data.values()])
        normalizer.fit_and_save(X, self._norm(self.config.mort_normalizer))

    def _fit_discharge_normalizer(self, data: dict[int, np.ndarray]) -> None:
        X = np.vstack(list(data.values()))[:, :, 1:]
        normalizer.fit_and_save(X, self._norm(self.config.disch_normalizer))

    def _build_mortality_dataset(self, data_filename: str, split: str, medians: dict) -> None:
        df = loader.load_raw_dataset(self._data(data_filename))
        df = df[df["stay_id"].isin(loader.load_stays(self._processed(f"{split}_stays.txt")))]
        df = preprocess(df, medians)
        df = df[df["los"] >= N_TIME_OFFSETS].copy()

        data = build_rolling_windows(df)
        last_48h = {sid: arr[-1:] for sid, arr in data.items()}
        outcomes = stay_to_mortality_outcome(df)

        loader.save_pkl(last_48h, self._processed(f"lstm_last_48h_{split}.pkl"))
        loader.save_pkl(outcomes, self._processed(f"icu_expire_flag_{split}.pkl"))

    def _build_discharge_dataset(self, data_filename: str, split: str, medians: dict) -> None:
        df = loader.load_raw_dataset(self._data(data_filename))
        df = df[df["stay_id"].isin(loader.load_stays(self._processed(f"{split}_stays.txt")))]
        df = preprocess(df, medians)
        df = df[df["los"] >= N_TIME_OFFSETS].copy()
        df["half_los"] = np.floor(df["los"] / 2)
        df["disch_48h"] = disch_label_series(df)

        all_data, outcomes = build_discharge_windows(df)
        loader.save_pkl(all_data, self._processed(f"lstm_disch_3point_48h_{split}.pkl"))
        loader.save_pkl(outcomes, self._processed(f"outcome_disch_3point_48h_{split}.pkl"))

    def _build_inference_dataset(
        self, data_filename: str, test_type: TestType, medians: dict
    ) -> dict:
        df = loader.load_raw_dataset(self._data(data_filename))
        df = df[df["stay_id"].isin(loader.load_stays(self._processed("test_stays.txt")))]
        df = preprocess(df, medians)
        df = df[df["los"] >= N_TIME_OFFSETS].copy()
        df["disch_48h"] = disch_label_series(df)

        data = build_rolling_windows(df)
        disch_out = disch_outcomes_by_stay(df)
        mort_out = stay_to_mortality_outcome(df)

        n = N_TIME_OFFSETS
        if test_type == "last_48h":
            data = {k: v[-(n + 1):] for k, v in data.items()}
            disch_out = {k: v[-(n + 1):] for k, v in disch_out.items()}
        elif test_type == "last_96h":
            data = {k: v[-(2 * n + 1):] for k, v in data.items()}
            disch_out = {k: v[-(2 * n + 1):] for k, v in disch_out.items()}
        elif test_type == "first_48h":
            data = {k: v[:n] for k, v in data.items()}
            disch_out = {k: v[:n] for k, v in disch_out.items()}

        return {"data": data, "mortality_outcome": mort_out, "disch_outcome": disch_out}
