import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import contextlib
import joblib
from tensorflow import keras
import numpy as np
from pandas.api.types import is_numeric_dtype
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, roc_curve, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from matplotlib import pyplot as plt

PARALLEL = True
N_TIME_OFFSETS = 48
TH = 0.019882

NAN_RULES = {
    "pao2fio2ratio_novent": "zero",
    "pao2fio2ratio_vent": "zero",
    "gcs_min": "prev_value",
    "rate_norepinephrine": "zero",
    "rate_epinephrine": "zero",
    "rate_dopamine": "zero",
    "rate_dobutamine": "zero",
    "meanbp_min": "prev_value",
    "bilirubin_max": "prev_value",
    "platelet_min": "prev_value",
    "creatinine_max": "prev_value",
}

OUTLIER_CORRECTION = {
    "rate_epinephrine": (0, 10),
    "rate_norepinephrine": (0, 5),
    "rate_dopamine": (0, 50),
    "rate_dobutamine": (0, 40),
    "meanbp_min": (10, 150),
    "pao2fio2ratio_novent": (50, 600),
    "pao2fio2ratio_vent": (50, 600),
    "bilirubin_max": (0.1, 70),
    "creatinine_max": (0.2, 20),
    "platelet_min": (5, 2000),
}

MANDATORY_COLUMNS = [
    "stay_id",
    "hr",
    "rate_epinephrine",
    "rate_norepinephrine",
    "rate_dopamine",
    "rate_dobutamine",
    "meanbp_min",
    "pao2fio2ratio_novent",
    "pao2fio2ratio_vent",
    "gcs_min",
    "bilirubin_max",
    "creatinine_max",
    "platelet_min",
    "admission_age",
    "icu_expire_flag",
    "admission_type_Medical",
    "admission_type_ScheduledSurgical",
    "admission_type_UnscheduledSurgical",
    "charlson_comorbidity_index",
]

FEATURES = [
    "rate_epinephrine",
    "rate_norepinephrine",
    "rate_dopamine",
    "rate_dobutamine",
    "meanbp_min",
    "pao2fio2ratio_novent",
    "pao2fio2ratio_vent",
    "gcs_min",
    "bilirubin_max",
    "creatinine_max",
    "platelet_min",
    "admission_age",
    "charlson_comorbidity_index",
    "admission_type_Medical",
    "admission_type_ScheduledSurgical",
    "admission_type_UnscheduledSurgical",
]

def check_dataset(dataset_filename, base_path):
    
    print(f"Checking {dataset_filename} dataset completness")

    data_folder_path = os.path.join(base_path, "data/")
    data_path = os.path.join(data_folder_path, dataset_filename)
    df = pd.read_csv(data_path, index_col=0)

    for c in MANDATORY_COLUMNS:
        assert c in df.columns, f"Column {c} is mandatory."
        # assert isinstance(df[c].dtype, type(np.dtype("float64"))) or isinstance(df[c].dtype, type(np.dtype("int")))
        assert is_numeric_dtype(df[c]), f"Column {c} must be numeric."

    print("Check done!")

def apply_nan_rules(df, rules):
    """
    Applies specified rules to fill NaN values in a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing NaN values to be filled.
    rules : dict
        A dictionary mapping column names to fill strategies. Supported strategies:
            - "zero": Fill NaNs with 0.
            - "prev_value": Fill NaNs with the previous value within each 'stay_id' group (forward fill).
            - any other value: Fill NaNs with the specified value.
    Returns
    -------
    pandas.DataFrame
        The DataFrame with NaN values filled according to the provided rules.
    """

    fill_with_zeros = [k for k, v in rules.items() if v == "zero"]
    fill_prev_value = [k for k, v in rules.items() if v == "prev_value"]
    fill_with_value = [
        (k, v) for k, v in rules.items() if v not in ["zero", "prev_value"]
    ]

    for c in tqdm(fill_with_zeros):
        df.loc[:, c] = df[c].fillna(0)
    for c in tqdm(fill_prev_value):
        g_ffill = df.groupby("stay_id")[[c]].fillna(method="ffill", inplace=False)
        df.loc[:, c] = g_ffill[c]
    for c, v in tqdm(fill_with_value):
        df.loc[:, c] = df[c].fillna(v)
    return df

def correct_outliers(data):
    """
    Corrects outliers in the specified features of the input DataFrame by clipping their values to predefined bounds.
    Args:
        data (pandas.DataFrame): The input DataFrame containing features to correct.
    Returns:
        pandas.DataFrame: The DataFrame with outliers in specified features clipped to their respective bounds.
    Notes:
        The function relies on the global dictionary OUTLIER_CORRECTION, which should map feature names to (lower_bound, upper_bound) tuples.
    """

    for feature, bounds in OUTLIER_CORRECTION.items():
        lower_bound, upper_bound = bounds
        data[feature] = data[feature].clip(lower_bound, upper_bound)
    return data

def _generate_offset(df, features, time_column="hour", offset=1):
    """
    Modifies the input DataFrame by offsetting the specified time column and renaming feature columns.
    Args:
        df (pd.DataFrame): The input DataFrame to modify.
        features (list of str): List of feature column names to rename with the offset.
        time_column (str, optional): Name of the time column to offset. Defaults to "hour".
        offset (int, optional): The value to add to the time column and append to feature names. Defaults to 1.
    Returns:
        pd.DataFrame: The modified DataFrame with updated time and feature columns.
    """

    df.loc[:, f"{time_column}_{offset}"] = df[time_column]
    df.loc[:, time_column] += offset
    renaming = {f: f"{f}_{offset}" for f in features}
    df.rename(columns=renaming, inplace=True)
    return df

def _get_sliced_dataframe(df, n_time_offsets, features, time_column="hour"):
    """
    Generates a sliced dataframe by merging multiple time-offset versions of the input dataframe.
    For each time offset in the range [0, n_time_offsets), the function creates a modified version
    of the input dataframe using the `_generate_offset` function. These offset dataframes are then
    merged sequentially on the "stay_id" and the specified time column. The final merged dataframe
    has the time column dropped before being returned.
    Args:
        df (pd.DataFrame): The input dataframe containing the data to be sliced.
        n_time_offsets (int): The number of time offsets to generate and merge.
        features (list): List of feature column names to be included in the offset dataframes.
        time_column (str, optional): The name of the time column to use for offsets and merging. Defaults to "hour".
    Returns:
        pd.DataFrame: The final merged dataframe with sliced features across time offsets, excluding the time column.
    """

    final_data = None
    for i in range(n_time_offsets):
        aux = _generate_offset(df.copy(), features, offset=i, time_column=time_column)
        if final_data is None:
            final_data = aux.copy()
        else:
            final_data = pd.merge(final_data, aux, on=["stay_id", time_column])
    final_data.drop(columns=[time_column], inplace=True)
    return final_data

def _get_numpy_matrix(df, fixed_cols, features, n_time_offsets):
    """
    Converts a pandas DataFrame into a 3D NumPy matrix for time series modeling.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing both fixed and time-dependent features.
        fixed_cols (list of str): List of column names in `df` that are fixed (do not vary with time).
        features (list of str): List of feature names that vary with time and are suffixed by time offset.
        n_time_offsets (int): Number of time offsets (timesteps) to include for each sample.
    Returns:
        np.ndarray: A 3D NumPy array of shape (num_samples, n_time_offsets, num_features),
                    where num_features = len(fixed_cols) + len(features).
                    For each timestep, the corresponding time-dependent features are selected
                    using the pattern '{feature}_{timestep}'.
    """

    matrix = np.ndarray((df.shape[0], n_time_offsets, len(fixed_cols) + len(features)))

    for i in range(n_time_offsets):
        timestep_features = [f"{f}_{i}" for f in features]

        matrix[:, i, :] = df[fixed_cols + timestep_features].to_numpy()
    return matrix

def _process_stay(df, stay_id, n_time_offsets, features, time_column, fixed_cols):
    """
    Processes data for a single stay by slicing the dataframe and converting it into a numpy matrix.
    Args:
        df (pd.DataFrame): The input dataframe containing stay data.
        stay_id (Any): The identifier for the stay being processed.
        n_time_offsets (int): Number of time offsets to consider for slicing.
        features (List[str]): List of feature column names to include.
        time_column (str): Name of the column representing time.
        fixed_cols (List[str]): List of columns to be treated as fixed (not time-dependent).
    Returns:
        Tuple[Any, np.ndarray]: A tuple containing the stay_id and the processed numpy matrix for the stay.
    """

    sliced_data = _get_sliced_dataframe(
        df[["stay_id", time_column] + features],
        n_time_offsets,
        features,
        time_column=time_column,
    )
    stay_matrix = _get_numpy_matrix(sliced_data, fixed_cols, features, n_time_offsets)
    return (stay_id, stay_matrix)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def generate_test_dataset(dataset_filename, base_path):
    """
    Generates a processed test dataset for ICU stay prediction tasks.
    This function reads a dataset from the specified path, processes each ICU stay by applying outlier correction,
    handling missing values according to predefined rules, and extracting relevant features for each time offset.
    It also computes discharge outcomes within 48 hours and mortality outcomes for each stay.
    Args:
        base_path (str): The base directory path where the dataset file is located.
        dataset_filename (str): The filename of the dataset to be loaded.
    Returns:
        dict: A dictionary containing:
            - "data": A mapping from stay_id to processed feature data for each time offset.
            - "mortality_outcome": A mapping from stay_id to ICU mortality outcome.
            - "disch_outcome": A mapping from stay_id to a list of discharge outcomes within 48 hours for each time offset.
    Notes:
        - Requires global variables: N_TIME_OFFSETS, FEATURES, PARALLEL, NAN_RULES.
        - Utilizes parallel processing if PARALLEL is True.
        - Assumes existence of helper functions: correct_outliers, apply_nan_rules, _process_stay, tqdm_joblib.
    """

    data_folder_path = os.path.join(base_path, "data/")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = pd.read_csv(data_path)
    lista_pacientes = data.stay_id.unique()
    data = data[data.stay_id.isin(lista_pacientes[:20])]

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])
    data = correct_outliers(data)
    data = apply_nan_rules(data, NAN_RULES)
    all_data = []
    if PARALLEL:
        with tqdm_joblib(tqdm(desc="Process df", total=len(stays_to_process))):
            all_data = Parallel(n_jobs=20)(
                delayed(_process_stay)(
                    data[data["stay_id"] == stay_id].copy(),
                    stay_id,
                    N_TIME_OFFSETS,
                    FEATURES,
                    time_column,
                    fixed_cols,
                )
                for stay_id in stays_to_process
            )
    else:
        for stay_id in tqdm(stays_to_process):
            aux = _process_stay(
                data[data["stay_id"] == stay_id].copy(),
                stay_id,
                N_TIME_OFFSETS,
                FEATURES,
                time_column,
                fixed_cols,
            )
            all_data.append(aux)
    all_data = dict(all_data)

    disch_outcomes = {}
    for stay_id in tqdm(stays_to_process):
        disch_outcomes[stay_id] = data[(data["stay_id"] == stay_id) & (data["hr"] >= (N_TIME_OFFSETS - 1))][
            "disch_48h"
        ].values.tolist()
    outcomes = data[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"].to_dict()

    print("Total stays with >={}h of data: {}.".format(N_TIME_OFFSETS, len(stays_to_process)))
    return {
        "data": all_data,
        "mortality_outcome": outcomes,
        "disch_outcome": disch_outcomes,
    }

def _normalize(base_path, X, load=False, save=True, filename="", model=""):
    """
    Normalizes the input data `X` using MinMaxScaler, with options to load or save the scaler.
    Parameters
    ----------
    base_path : str
        The base directory path for saving or loading the scaler.
    X : np.ndarray
        The input data array of shape (n_samples, 48, 16) to be normalized.
    load : bool, optional
        If True, loads an existing scaler from file. If False, fits a new scaler on the data. Default is False.
    save : bool, optional
        If True, saves the fitted scaler to file. Default is True.
    filename : str, optional
        The filename for saving or loading the scaler. If empty, a default filename is used.
    Returns
    -------
    np.ndarray
        The normalized data array with the same shape as the input `X`.
    """

    if len(filename) > 0:
        fn = os.path.join(base_path, filename)
    else:
        if model == "discharge":
            fn = os.path.join(base_path, "mimic_iv_normalizer_disch.pkl")
        elif model == "mortality":
            fn = os.path.join(base_path, "mimic_iv_normalizer.pkl")
        else:
            raise ValueError(f"Unknown model type: {model}")

    if load:
        norm = pickle.load(open(fn, "rb"))
    else:
        norm = MinMaxScaler()
        norm = norm.fit(X[:, 0, :])
        if save:
            pickle.dump(norm, open(fn, "wb"))
    tmp = X.shape[0]
    X = X[:, :, :].reshape((tmp * 48, 16))
    X = norm.transform(X)
    X = X.reshape((tmp, 48, 16))
    return X

def softmax_temperature(x, temperature=1.0):
    """
    Applies the softmax function with temperature scaling to the input tensor.
    Args:
        x: Input tensor.
        temperature (float, optional): Temperature parameter for scaling logits.
            Higher values produce softer probability distributions. Default is 1.0.
    Returns:
        Tensor: Softmax probabilities computed along the last axis of the input tensor.
    """

    exp_x = keras.backend.exp(x / temperature)
    return exp_x / keras.backend.sum(exp_x, axis=-1, keepdims=True)

def get_mortality_predictions(base_path, dataset, mortality_model_filename):
    """
    Generates mortality predictions using a pre-trained LSTM model.
    Args:
        base_path (str): The base directory path for loading normalization parameters.
        dataset (dict): A dictionary containing patient data and mortality outcomes.
            Expected keys:
                - "data": dict mapping stay_id to patient data arrays.
                - "mortality_outcome": dict mapping stay_id to mortality outcome (0 or 1).
        mortality_model_filename (str): Path to the saved Keras LSTM model file.
    Returns:
        dict: A dictionary with the following keys:
            - "y_true": Ground truth mortality outcomes as a numpy array.
            - "y_pred": Predicted mortality probabilities from the model as a numpy array.
    """

    model_folder_path = os.path.join(base_path, "models/")
    model_path = os.path.join(model_folder_path, mortality_model_filename)

    normalizer_name = "mimic_iv_normalizer.pkl"
    normalizer_folder_path = os.path.join(base_path, "normalizers/")

    tmp_x = []
    tmp_y = []
    for stay_id in tqdm(dataset["data"]):
        tmp_x.append(dataset["data"][stay_id][:, ::-1, 1:])
        tmp_y.append(np.ones((dataset["data"][stay_id].shape[0], 1)) * dataset["mortality_outcome"][stay_id])
    X = np.vstack(tmp_x)
    y = np.vstack(tmp_y)
    X = _normalize(
        normalizer_folder_path,
        X,
        load=True,
        save=False,
        filename=normalizer_name,
        model="mortality"
    )

    X = np.nan_to_num(X)
    model_lstm = keras.models.load_model(model_path, custom_objects={'softmax_temperature': softmax_temperature})
    return {"y_true": y, "y_pred": model_lstm.predict(X)}

def get_discharge_predictions(base_path, dataset, discharge_model_filename):
    """
    Generates discharge outcome predictions using a pre-trained LSTM model.
    Args:
        base_path (str): The base directory path for loading normalization parameters.
        dataset (dict): A dictionary containing patient data and discharge outcomes.
            Expected keys are "data" (feature arrays) and "disch_outcome" (outcome arrays).
        discharge_model_filename (str): Path to the saved Keras LSTM model file.
    Returns:
        dict: A dictionary with the following keys:
            - "y_true": List of true discharge outcome arrays.
            - "y_pred": Numpy array of predicted discharge outcomes from the model.
    """

    model_folder_path = os.path.join(base_path, "models/")
    model_path = os.path.join(model_folder_path, discharge_model_filename)

    normalizer_name = "mimic_iv_normalizer_disch.pkl"
    normalizer_folder_path = os.path.join(base_path, "normalizers/")

    X = np.vstack(list(dataset["data"].values()))[:, :, 1:]
    y = [np.array(item) for item in list(dataset["disch_outcome"].values())]
    X = _normalize(
        normalizer_folder_path,
        X,
        load=True,
        save=False,
        filename=normalizer_name,
        model="discharge"
    )
    X = np.nan_to_num(X)
    model_lstm = keras.models.load_model(model_path, custom_objects={'softmax_temperature': softmax_temperature})
    return {"y_true": y, "y_pred": model_lstm.predict(X)}

def run_test_pipeline(dataset_filename, base_path, mortality_model_filename, disch_model_filename):
    check_dataset(dataset_filename, base_path)
    dataset = generate_test_dataset(dataset_filename, base_path)

    mortality_outcome = get_mortality_predictions(base_path, dataset, mortality_model_filename)
    discharge_outcome = get_discharge_predictions(base_path, dataset, disch_model_filename)
    return {"mortality_outcome": mortality_outcome, "discharge_outcome": discharge_outcome}

def _plot_roc(base_path, image_name, groundtruth, predictions, **kwargs):

    # path
    results_folder_path = os.path.join(base_path, "results/")
    os.makedirs(results_folder_path, exist_ok=True)
    image_path = os.path.join(results_folder_path, image_name)

    save_flag = kwargs.pop('save', False)

    fp, tp, _ = roc_curve(groundtruth, predictions)
    auc = roc_auc_score(groundtruth, predictions)

    plt.plot(100 * fp, 100 * tp, label=f'(AUC={auc:.2f})',
             linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.legend()

    if save_flag:
        plt.savefig(f'{image_path}.png', bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    result = run_test_pipeline("jx_data.csv", "./", "lstm_mortality_model.keras", "lstm_discharge_model.keras")
    
    mortality_pred = result['mortality_outcome']['y_pred'][:, 1]
    mortality_binary = (mortality_pred > .5).astype(int)
    mortality_groundtruth = result['mortality_outcome']['y_true'][:, 0]
    print(mortality_pred)
    print(np.unique(mortality_binary, return_counts=True))
    print(np.unique(mortality_groundtruth, return_counts=True))
    print(confusion_matrix(mortality_groundtruth, mortality_binary))
    _plot_roc(
        base_path="./",
        image_name="roc_mortality",
        groundtruth=mortality_groundtruth,
        predictions=mortality_pred,
        color="tab:red",
        save=True
    )

    # discharge_pred = result['discharge_outcome']['y_pred'][:, 1]
    # discharge_groundtruth = np.hstack(result['discharge_outcome']['y_true'])
    # _plot_roc(
    #     base_path="./",
    #     image_name="roc_discharge",
    #     groundtruth=discharge_groundtruth,
    #     predictions=discharge_groundtruth,
    #     color="tab:red",
    #     save=True
    # )