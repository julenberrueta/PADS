import argparse
import asyncio
import contextlib
import gc
import importlib
import json
import logging
import os
import pickle
import sys
import warnings

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import keras.layers as layers
from keras.losses import BinaryFocalCrossentropy
from keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import pytz
from typing import Dict, Tuple, Any, List, Optional, Union

from .discharge_model import _prepare_data as _prepare_disch_data
from .discharge_model import _get_weights as _get_disch_weights
from .mortality_model import _prepare_data as _prepare_mort_data
from .mortality_model import _get_weights as _get_mort_weights

PARALLEL = True
N_TIME_OFFSETS = 48
LEARNING_RATE_DISCH = 1e-5
LEARNING_RATE_MORT = 1e-5

IMPUTE_FIRST_ROW = [
    "gcs_min",
    "meanbp_min",
    "bilirubin_max",
    "platelet_min",
    "creatinine_max",
]

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

def check_dataset(base_path: str, dataset_filename: str):

    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    if not os.path.isfile(data_path):
        print(f"Error: The file '{data_path}' does not exist.")
    else:
        df = pd.read_csv(data_path)

        for c in MANDATORY_COLUMNS:
            assert c in df.columns, f"Column {c} is mandatory."
            assert is_numeric_dtype(df[c]), f"Column {c} must be numeric."

def _get_data(base_path: str, dataset_filename: str) -> pd.DataFrame:
    """
    Loads and processes patient data from the specified base path.
    This function performs the following steps:
    1. Reads the main patient dataset from 'mimic_iv_nan_corrected.csv'.
    2. Sets the 'outcome' column to the value of 'icu_expire_flag'.
    3. Excludes patients listed in 'patients_ltsv.csv'.
    4. Logs the number of patients before and after exclusion.
    5. Extracts and saves outcome dictionaries for 'outcome', 'icu_expire_flag', and 'hospital_expire_flag'
       as pickle files in the base path.
    6. Calculates the length of stay (LOS) for each patient based on the maximum 'hr' value and adds it to the data.
    Args:
        base_path (str): The directory path containing the required CSV files.
    Returns:
        pandas.DataFrame: The processed patient data with updated columns and excluded patients.
    """

    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)
    data = pd.read_csv(data_path)
    data.loc[:, "outcome"] = data["icu_expire_flag"]

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)

    return data

def _correct_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects outliers in the specified features of the input DataFrame by clipping their values to predefined bounds.
    Args:
        df (pandas.DataFrame): The input DataFrame containing features to correct.
    Returns:
        pandas.DataFrame: The DataFrame with outliers in specified features clipped to their respective bounds.
    Notes:
        The function relies on the global dictionary OUTLIER_CORRECTION, which should map feature names to (lower_bound, upper_bound) tuples.
    """

    for feature, bounds in OUTLIER_CORRECTION.items():
        lower_bound, upper_bound = bounds
        df[feature] = df[feature].clip(lower_bound, upper_bound)
    return df

def _impute_first_row(df: pd.DataFrame, medians_dict: dict) -> pd.DataFrame:
    """
    Imputes missing values in the first row (hr == 0) per stay_id using the provided global medians.

    Parameters:
        df: Original DataFrame with potential NaNs
        medians_dict: Dictionary containing global medians for the columns

    Returns:
        Imputed DataFrame
    """

    cols = list(medians_dict.keys())
    mask_hr0 = df['hr'] == 0

    for col in cols:
        df.loc[mask_hr0, col] = df.loc[mask_hr0, col].fillna(medians_dict[col])

    return df

def _apply_nan_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
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


def _generate_offset(df: pd.DataFrame, features: List[str], time_column: Optional[str] = "hr", offset: Optional[int] = 1) -> pd.DataFrame:
    """
    Modifies the input DataFrame by offsetting the specified time column and renaming feature columns.
    Args:
        df (pd.DataFrame): The input DataFrame to modify.
        features (list of str): List of feature column names to rename with the offset.
        time_column (str, optional): Name of the time column to offset. Defaults to "hr".
        offset (int, optional): The value to add to the time column and append to feature names. Defaults to 1.
    Returns:
        pd.DataFrame: The modified DataFrame with updated time and feature columns.
    """

    df.loc[:, f"{time_column}_{offset}"] = df[time_column]
    df.loc[:, time_column] += offset
    renaming = {f: f"{f}_{offset}" for f in features}
    df.rename(columns=renaming, inplace=True)
    return df

def _get_sliced_dataframe(df: pd.DataFrame, n_time_offsets: int, features: List[str], time_column: Optional[str] = "hr") -> pd.DataFrame:
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
        time_column (str, optional): The name of the time column to use for offsets and merging. Defaults to "hr".
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

def _get_numpy_matrix(df: pd.DataFrame, fixed_cols: List[str], features: List[str], n_time_offsets: int) -> np.ndarray:
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

def _process_stay(df: pd.DataFrame, stay_id: Any, n_time_offsets: int, features: List[str], time_column: str, fixed_cols: List[str]):
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

def _process_discharge_dataset(data: pd.DataFrame) -> Dict[dict, dict]:
    """
    Processes a patient discharge dataset to extract multiple time-based datapoints for each ICU stay.
    The function selects up to five specific time windows for each patient stay:
        - First available datapoint (first 48 hours)
        - Second datapoint (hours 24-71)
        - Intermediate datapoint before halfway point of stay
        - Intermediate datapoint after halfway point of stay
        - Final datapoint (last 48 hours of stay)
    For each stay, it extracts a fixed set of features and outcomes for each time window, reshaping the data
    into arrays suitable for machine learning tasks.
    Args:
        data (pd.DataFrame): Input DataFrame containing patient time-series data with columns for time offsets,
            features, and outcomes.
    Returns:
        tuple:
            all_data (dict): Dictionary mapping stay_id to a numpy array of shape (n_datapoints, 48, 17),
                where n_datapoints is the number of time windows available for the stay.
            outcomes (dict): Dictionary mapping stay_id to a numpy array of shape (n_datapoints, 1),
                containing the discharge outcome for each time window.
    """

    n_time_offsets = 48

    # first available datapoint
    first_datapoint = data[(data["los"] >= n_time_offsets) & (data["hr"] < n_time_offsets)].copy()

    second_datapoint = data[
        (data["los"] >= n_time_offsets) & (data["hr"] < n_time_offsets + 24) & (data["hr"] >= 24)
    ].copy()

    # intermediate datapoint (half of the stay)
    int_bef_datapoint = data[
        (data["half_los"] >= n_time_offsets)
        & (data["hr"] < data["half_los"])
        & (data["hr"] >= data["half_los"] - n_time_offsets)
    ].copy()

    int_aft_datapoint = data[
        (data["half_los"] >= n_time_offsets)
        & (data["hr"] >= data["half_los"])
        & (data["hr"] < data["half_los"] + n_time_offsets)
    ].copy()

    # final datapoint (end of the stay)
    final_datapoint = data[(data["los"] >= n_time_offsets) & (data["hr"] > data["los"] - n_time_offsets)].copy()

    n_time_offsets = 48
    fixed_cols = ["stay_id"]

    outcomes = {}
    all_data = {}
    for stay_id in tqdm(set(first_datapoint["stay_id"])):
        all_data[stay_id] = (
            first_datapoint.loc[first_datapoint["stay_id"] == stay_id, fixed_cols + FEATURES]
            .to_numpy()
            .reshape(1, 48, 17)
        )
        outcomes[stay_id] = (
            first_datapoint.loc[first_datapoint["stay_id"] == stay_id, "disch_48h"].to_numpy()[-1:].reshape(-1, 1)
        )
    for datapoint in [
        second_datapoint,
        int_bef_datapoint,
        int_aft_datapoint,
        final_datapoint,
    ]:
        for stay_id in tqdm(set(datapoint["stay_id"])):
            if datapoint[datapoint["stay_id"] == stay_id].shape[0] != 48:
                continue
            aux = datapoint.loc[datapoint["stay_id"] == stay_id, fixed_cols + FEATURES].to_numpy().reshape(1, 48, 17)
            aux_outcome = datapoint.loc[datapoint["stay_id"] == stay_id, "disch_48h"].to_numpy()[-1:].reshape(-1, 1)
            if stay_id in all_data:
                all_data[stay_id] = np.vstack([all_data[stay_id], aux])
                outcomes[stay_id] = np.vstack([outcomes[stay_id], aux_outcome])
            else:
                all_data[stay_id] = aux
                outcomes[stay_id] = aux_outcome
    return all_data, outcomes

def _normalize(base_path: str, X: np.ndarray, load: Optional[bool] = False, save: Optional[bool] = True, filename: Optional[str] = "", model: Optional[str] = "") -> np.ndarray:
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

def _get_retrain_test_split_disch(data: Dict[Any, np.ndarray], outcome: Dict[Any, np.ndarray]):
    
    stay_ids = list(data.keys())
    valid_stay_ids = [sid for sid in data if sid in outcome]
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    print(f"{len(train_stays)} patients for retraining discharge model")

    X_train = np.ndarray((len(train_stays), 48, 16))
    y_train = np.ndarray((len(train_stays), 1))
    X_test = np.ndarray((len(test_stays), 48, 16))
    y_test = np.ndarray((len(test_stays), 1))

    offset = 0
    for stay_id in tqdm(train_stays):
        if stay_id not in data:
            continue
        for sample in range(data[stay_id].shape[0]):
            X_train[offset:offset+1, :, :] = data[stay_id][sample, :, 1:]
            y_train[offset:offset+1, :] = outcome[stay_id][sample]
            offset += 1

    offset = 0
    for stay_id in tqdm(test_stays):
        if stay_id not in data:
            continue
        for sample in range(data[stay_id].shape[0]):
            X_test[offset:offset+1, :, :] = data[stay_id][sample, :, 1:]
            y_test[offset:offset+1, :] = outcome[stay_id][sample]
            offset += 1

    return X_train, X_test, y_train, y_test

def _get_retrain_test_split_mort(data: Dict[Any, np.ndarray], outcome: Dict[Any, np.ndarray]):
    
    stay_ids = list(data.keys())
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    X_train = np.ndarray((len(train_stays), 48, 16))
    y_train = np.ndarray((len(train_stays), 1))
    X_test = np.ndarray((len(test_stays), 48, 16))
    y_test = np.ndarray((len(test_stays), 1))

    offset = 0
    for stay_id in tqdm(train_stays):
        if stay_id not in data:
            continue
        X_train[offset:offset+1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_train[offset:offset+1, :] = outcome[stay_id]
        offset += 1

    offset = 0
    for stay_id in tqdm(test_stays):
        if stay_id not in data:
            continue
        X_test[offset:offset+1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_test[offset:offset+1, :] = outcome[stay_id]
        offset += 1

    return X_train, X_test, y_train, y_test

class SoftmaxTemperature(layers.Layer):
    """
    A custom Keras layer that applies the softmax activation function with a configurable temperature.
    The temperature parameter controls the smoothness of the output probabilities:
    - Higher temperature produces a softer probability distribution.
    - Lower temperature makes the distribution more peaked.
    Args:
        temperature (float): The temperature value to scale the inputs before applying softmax. Default is 1.0.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    Methods:
        call(inputs): Applies the temperature-scaled softmax to the input tensor.
        get_config(): Returns the configuration of the layer for serialization.
    Example:
        layer = SoftmaxTemperature(temperature=0.5)
    """

    def __init__(self, temperature=1.0, **kwargs):
        super(SoftmaxTemperature, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        exp_x = keras.backend.exp(inputs / self.temperature)
        return exp_x / keras.backend.sum(exp_x, axis=-1, keepdims=True)

    def get_config(self):
        config = super(SoftmaxTemperature, self).get_config()
        config.update({"temperature": self.temperature})
        return config

def softmax_temperature(x: Union[tf.Tensor, float], temperature: float = 1.0) -> tf.Tensor:
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

def _create_lstm_disch_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Creates and compiles an LSTM-based neural network model for classification tasks.
    Args:
        input_shape (tuple): Shape of the input data, typically (batch_size, timesteps, features).
    Returns:
        keras.Model: A compiled Keras Sequential model with LSTM and dense layers.
    Model Architecture:
        - LSTM layer with 50 units.
        - Dense layer with 50 units and ReLU activation.
        - Batch normalization.
        - Dropout layer with 0.3 rate.
        - Dense layer with 20 units and ReLU activation.
        - Dropout layer with 0.3 rate.
        - Dense output layer with 2 units (logits).
        - SoftmaxTemperature layer for temperature-scaled softmax.
    Compilation:
        - Loss: Binary crossentropy.
        - Optimizer: Adam.
        - Metrics: accuracy, AUC, precision, recall, categorical accuracy, weighted F1 score.
    """

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, input_shape=input_shape[1:]))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2, name="logits"))
    model.add(SoftmaxTemperature(temperature=1))
    return model

def _create_lstm_mort_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Creates and compiles an LSTM-based Keras Sequential model for binary classification.
    The model architecture includes:
        - LSTM layer with 100 units
        - Dense layers with ReLU activation
        - Batch normalization and dropout for regularization
        - Output Dense layer with 2 units (logits)
        - Custom softmax activation with temperature scaling
    The model is compiled with:
        - Binary focal cross-entropy loss with class balancing
        - Adam optimizer
        - Weighted metrics: accuracy, AUC, precision, recall, categorical accuracy, and weighted F1 score
    Args:
        input_shape (tuple): Shape of the input data, including batch size and time steps.
    Returns:
        keras.Model: Compiled Keras Sequential model.
    """

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=100, input_shape=input_shape[1:]))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2, name="logits"))
    model.add(SoftmaxTemperature(temperature=1))
    return model

def _fit_and_eval_model_retrain(model,
                        model_name,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        model_type,
                        epochs=1000,
                        batch_size=100):

    logger = TensorBoard(log_dir=f'./models/tensorflow_logs/{model_name}', write_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True)

    model_checkpoint = ModelCheckpoint(
        f'./models/tensorflow_logs/{model_name}_best_model.keras',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    if model_type == "discharge":

        weights = _get_disch_weights(y_train)

        _ = model.fit(
            X_train,
            y_train,
            class_weight=weights,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_data=(X_val, y_val),
            callbacks=[logger, early_stopping, model_checkpoint]
        )

    elif model_type == "mortality":

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train[:, 0])
        weights = {i: weight for i, weight in enumerate(class_weights)}

        weights[0] = weights[0] * 0.8
        weights[1] = weights[1] * 1.5

        _ = model.fit(
            X_train,
            y_train,
            class_weight=weights,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_data=(X_val, y_val),
            callbacks=[logger, early_stopping, model_checkpoint]
        )

def _plot_roc(base_path, image_name, groundtruth, predictions, **kwargs):

    results_folder_path = os.path.join(base_path, "results")
    os.makedirs(results_folder_path, exist_ok=True)
    image_path = os.path.join(results_folder_path, image_name)

    save_flag = kwargs.pop('save', False)

    fp, tp, thresholds = roc_curve(groundtruth, predictions)
    auc = roc_auc_score(groundtruth, predictions)

    # Optimal threshold (Youden's J)
    j_scores = tp - fp
    j_max_idx = np.argmax(j_scores)
    optimal_fp = fp[j_max_idx]
    optimal_tp = tp[j_max_idx]
    optimal_thresh = thresholds[j_max_idx]

    # Binary predictions at optimal threshold
    bin_preds = (predictions >= optimal_thresh).astype(int)

    # Metrics
    accuracy = accuracy_score(groundtruth, bin_preds)
    precision = precision_score(groundtruth, bin_preds, zero_division=0)
    recall = recall_score(groundtruth, bin_preds, zero_division=0)
    f1 = f1_score(groundtruth, bin_preds, zero_division=0)

    # Plot ROC
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(100 * fp, 100 * tp, label=f'AUC = {auc:.2f}', linewidth=2, **kwargs)
    ax.scatter(100 * optimal_fp, 100 * optimal_tp, color='blue',
               label=f'Optimal threshold = {optimal_thresh:.4f}', zorder=5)

    ax.set_xlabel('False positives [%]')
    ax.set_ylabel('True positives [%]')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.set_aspect('equal')

    # Legend
    legend_elements = [
        Line2D([0], [0], color='C0', lw=2, label=f'AUC = {auc:.2f}'),
        Line2D([0], [0], marker='o', color='w', label=f'Optimal threshold = {optimal_thresh:.4f}',
               markerfacecolor='blue', markersize=8)
    ]
    legend = ax.legend(handles=legend_elements, loc='center left',
                       bbox_to_anchor=(1.02, 0.55), borderaxespad=0.)

    metrics_text = (
        f"$\\bf{{Accuracy}}$: {accuracy:.2f}\n"
        f"$\\bf{{Precision}}$: {precision:.2f}\n"
        f"$\\bf{{Recall}}$: {recall:.2f}\n"
        f"$\\bf{{F1}}$: {f1:.2f}\n"
        f"$\\bf{{AUC}}$: {auc:.2f}"
    )

    plt.text(1.02, 0.05, metrics_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round", facecolor='#f0f0f0', edgecolor='gray'))

    plt.tight_layout()

    if save_flag:
        plt.savefig(f'{image_path}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return optimal_thresh

def _get_optimal_threshold(method, y_true, y_pred):
    fp, tp, thresholds = roc_curve(y_true, y_pred)

    if method == 'youden':
        j_scores = tp - fp
        idx = np.argmax(j_scores)
    elif method == 'min_distance':
        distances = np.sqrt(fp**2 + (1 - tp)**2)
        idx = np.argmin(distances)
    elif method == 'precision_recall':
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        fscore = 2 * precision * recall / (precision + recall + 1e-8)
        idx = np.argmax(fscore)
        return pr_thresholds[idx]
    else:
        raise ValueError(f"Unknown threshold_method: {method}")

    return thresholds[idx]

def _plot_roc_combined(base_path,
                       mortality_pred, mortality_groundtruth,
                       discharge_pred, discharge_groundtruth,
                       threshold_method='youden', save=True, **kwargs):

    save_folder = os.path.join(base_path, "results", "images")
    os.makedirs(save_folder, exist_ok=True)

    # --- Mortality ---
    auc_mort = roc_auc_score(mortality_groundtruth, mortality_pred)
    fp_mort, tp_mort, thresholds_mort = roc_curve(mortality_groundtruth, mortality_pred)
    optimal_thresh_mort = _get_optimal_threshold(threshold_method, mortality_groundtruth, mortality_pred)
    bin_mort = (mortality_pred >= optimal_thresh_mort).astype(int)

    acc_mort = accuracy_score(mortality_groundtruth, bin_mort)
    prec_mort = precision_score(mortality_groundtruth, bin_mort, zero_division=0)
    rec_mort = recall_score(mortality_groundtruth, bin_mort, zero_division=0)
    f1_mort = f1_score(mortality_groundtruth, bin_mort, zero_division=0)

    opt_idx_mort = np.argmin(np.abs(thresholds_mort - optimal_thresh_mort))
    opt_fp_mort = fp_mort[opt_idx_mort]
    opt_tp_mort = tp_mort[opt_idx_mort]

    # --- Discharge ---
    auc_disch = roc_auc_score(discharge_groundtruth, discharge_pred)
    fp_disch, tp_disch, thresholds_disch = roc_curve(discharge_groundtruth, discharge_pred)
    optimal_thresh_disch = _get_optimal_threshold(threshold_method, discharge_groundtruth, discharge_pred)
    bin_disch = (discharge_pred >= optimal_thresh_disch).astype(int)

    acc_disch = accuracy_score(discharge_groundtruth, bin_disch)
    prec_disch = precision_score(discharge_groundtruth, bin_disch, zero_division=0)
    rec_disch = recall_score(discharge_groundtruth, bin_disch, zero_division=0)
    f1_disch = f1_score(discharge_groundtruth, bin_disch, zero_division=0)

    opt_idx_disch = np.argmin(np.abs(thresholds_disch - optimal_thresh_disch))
    opt_fp_disch = fp_disch[opt_idx_disch]
    opt_tp_disch = tp_disch[opt_idx_disch]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(100 * fp_mort, 100 * tp_mort, color='#d4453b', linewidth=2,
            label=f'Mortality (AUC = {auc_mort:.2f})')
    ax.scatter(100 * opt_fp_mort, 100 * opt_tp_mort, color='#d4453b', edgecolor='black', zorder=5)

    ax.plot(100 * fp_disch, 100 * tp_disch, color='#298fcf', linewidth=2,
            label=f'Discharge (AUC = {auc_disch:.2f})')
    ax.scatter(100 * opt_fp_disch, 100 * opt_tp_disch, color='#298fcf', edgecolor='black', zorder=5)

    ax.set_title(f"ROC Curve - Mortality & Discharge\nThreshold Method: {threshold_method}", fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positives [%]')
    ax.set_ylabel('True Positives [%]')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend(loc='center right', fontsize=10)

    # --- Metrics Table ---
    metrics_text = (
        "           Accuracy  Precision  Recall  F1 Score  AUC  Threshold\n"
        f"Mortality  {acc_mort:8.2f}  {prec_mort:9.2f}  {rec_mort:6.2f}  {f1_mort:8.2f}  {auc_mort:4.2f}  {optimal_thresh_mort:.4f}\n"
        f"Discharge  {acc_disch:8.2f}  {prec_disch:9.2f}  {rec_disch:6.2f}  {f1_disch:8.2f}  {auc_disch:4.2f}  {optimal_thresh_disch:.4f}"
    )

    fig.text(0.5, -0.01, metrics_text, ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(facecolor='#f0f0f0', edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save == True:
        plt.savefig(os.path.join(save_folder, f"roc_combined.png"), bbox_inches='tight')
    plt.close()

    return optimal_thresh_mort, optimal_thresh_disch

def _plot_inference(df_prob, base_path, test_type="last_96h", save=False):
    """
    Plots adjusted mortality probabilities over time per patient using `df_prob` which includes:
    - mortality_prob, disch_prob, prob_mortality, color_group, stay_id, hr
    Saves one image per patient as {stay_id}_{test_type}.png inside base_path/results/images/
    """

    save_folder = os.path.join(base_path, "results", "images")

    for stay_id, group_df in df_prob.groupby("stay_id"):
        plt.figure(figsize=(10, 5))

        group_df = group_df.copy()
        group_df['normalized'] = group_df['normalized'].fillna(50)
        group_df['color_group'] = group_df['color_group'].fillna('GRIS')

        max_hr = group_df.hr.max()

        plt.scatter(group_df['hr'], group_df['normalized'],
                    c=group_df['color_group'].map({
                        "EXITUS <48h": "#f43543",
                        "EXITUS >48h": "#ff9e30",
                        "ALIVE <48h": "#3ab830",
                        "ALIVE >48h": "#1097d8"
                    }),
                    s=50, alpha=0.85)

        plt.axvline(x=max_hr-48, color='black', linestyle='--', linewidth=1.5, label='48h before discharge')
        plt.axhline(y=50, color='gray', linestyle=':', linewidth=1.2)

        plt.xlabel("hr (relative to discharge)", fontsize=12)
        plt.ylabel("Adjusted Mortality Probability (%)", fontsize=12)

        final_status = "ALIVE" if group_df['mortality_groundtruth'].iloc[-1] == 0 else "EXITUS"
        title_color = "#3ab830" if final_status == "ALIVE" else "#f43543"
        plt.title(f"Stay {stay_id} - {test_type}", fontsize=14, color=title_color)

        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.legend()

        if save:
            image_path = os.path.join(save_folder, f"{test_type}/{stay_id}.png")
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def _plot_error(df: pd.DataFrame, base_path: str):
    
    save_folder = os.path.join(base_path, "results", "images")
    
    color_map = {
        0.0: 'lightgreen',
        1.0: 'lightyellow',
        2.0: 'lightsalmon',
        3.0: 'lightcoral'
    }

    bar_data = df['error'].value_counts(normalize=True).reset_index()
    bar_data['index'] = bar_data['index'].astype('int')
    bar_data.sort_values("index", ascending=True, inplace=True)
    bar_data['error'] = bar_data['error'].round(2)
    mean_error = df['error'].mean()

    bar_data['color'] = bar_data['index'].map(color_map).fillna('gray')

    plt.figure(figsize=(8, 6))
    bars = plt.bar(bar_data['index'].astype(str), bar_data['error'], color=bar_data['color'], edgecolor='black', alpha=0.6)

    plt.title('Error Percentage per Error Group')
    plt.xlabel('Group')
    plt.ylabel('Percentage [%]')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 6), ha='center', va='bottom')
        
    plt.text(2.4, 0.55, f'Mean Error: {mean_error:.6f}', fontsize=10, ha='left')

    plt.xlim(-0.5, len(bar_data) - 0.5)
    plt.savefig(os.path.join(save_folder, f"barplot_error.png"))
    plt.close()

    plot_data = df.groupby(['real_color_group','color_group','error']).size().reset_index()
    plot_data.rename(columns={0:"count"}, inplace=True)

    heatmap_data = plot_data.pivot_table(index='real_color_group', columns='color_group', values='count', aggfunc='sum')

    color_data = plot_data.pivot_table(index='real_color_group', columns='color_group', values='error', aggfunc='first')

    max_count = heatmap_data.max().max()
    circle_size = heatmap_data / max_count * 12000

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            count = heatmap_data.iloc[i, j]
            if pd.notna(count):
                error_value = color_data.iloc[i, j]
                color = color_map[error_value]
                ax.scatter(j, i, s=circle_size.iloc[i, j], color=color, alpha=0.6, edgecolor='black')

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            count = heatmap_data.iloc[i, j]
            if pd.notna(count):
                ax.text(j, i, int(count), color='black', ha='center', va='center', fontsize=10)

    ax.set_xlim(-0.5, len(heatmap_data.columns) - 0.5)
    ax.set_ylim(len(heatmap_data.index) - 0.5, -0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.grid(False)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(save_folder, f"heatmap_error.png"))
    plt.close()

def generate_medians_json(base_path: str, dataset_filename: str):
    
    data_folder_path = os.path.join(base_path, "data")

    data = _get_data(base_path, dataset_filename)
    data_48h = data[data.hr <= 48]
    medians_48h = data_48h[IMPUTE_FIRST_ROW].median().to_dict()

    with open(os.path.join(data_folder_path, "medians_48h.json"), "w") as f:
        json.dump(medians_48h, f, indent=4)

def generate_train_test_split_file(base_path: str, dataset_filename: str):
    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = _get_data(base_path, dataset_filename)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

    stay_ids = data.stay_id.unique()
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    with open(os.path.join(data_folder_path, 'train_stays.txt'), 'w') as f:
        for stay in train_stays:
            f.write(f"{stay}\n")

    with open(os.path.join(data_folder_path, 'test_stays.txt'), 'w') as f:
        for stay in test_stays:
            f.write(f"{stay}\n")

def generate_normalizer_dataset(base_path: str, dataset_filename: str):
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
        - Assumes existence of helper functions: _correct_outliers, _impute_first_row, _apply_nan_rules, _process_stay, tqdm_joblib.
    """

    data_folder_path = os.path.join(base_path, "data")

    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, 'train_stays.txt'), 'r') as f:
        train_stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(train_stays)]

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    data = _correct_outliers(data)
    data = _impute_first_row(data, medians_48h)
    data = _apply_nan_rules(data, NAN_RULES)
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

def generate_mortality_normalizer(base_path: str, dataset: dict):

    normalizer_folder_path = os.path.join(base_path, "normalizers")

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
        load=False,
        save=True,
        filename="custom_normalizer.pkl",
    )

def generate_discharge_normalizer(base_path: str, dataset: dict):

    normalizer_folder_path = os.path.join(base_path, "normalizers")

    X = np.vstack(list(dataset["data"].values()))[:, :, 1:]
    y = [np.array(item) for item in list(dataset["disch_outcome"].values())]
    X = _normalize(
        normalizer_folder_path,
        X,
        load=False,
        save=True,
        filename="custom_normalizer_disch.pkl",
    )

def generate_retrain_mortality_dataset(base_path: str, dataset_filename: str, data_type: str):
    """
    Generates and saves datasets for mortality prediction from ICU stay data.
    This function processes ICU stay data to create datasets suitable for LSTM-based mortality prediction.
    It filters stays with sufficient length, processes each stay (optionally in parallel), and saves the
    processed data and outcome labels as pickle files.
    Args:
        base_path (str): The base directory path where input data is located and output files will be saved.
    Saves:
        - lstm_last_48h_train.pkl: Dictionary mapping stay_id to the last 48 hours of data for each stay.
        - icu_expire_flag_train.pkl: Dictionary mapping stay_id to ICU expire flag.
        - lstm_last_48h_test.pkl: Dictionary mapping stay_id to the last 48 hours of data for each stay.
        - icu_expire_flag_test.pkl: Dictionary mapping stay_id to ICU expire flag.
    Prints:
        - The total number of stays with sufficient data processed.
    Note:
        Requires global variables and helper functions such as N_TIME_OFFSETS, FEATURES, PARALLEL,
        _get_data, _process_stay, tqdm_joblib, tqdm, Parallel, delayed, os, and pickle to be defined elsewhere.
    """

    data_folder_path = os.path.join(base_path, "data")
    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    if data_type == "train":
        with open(os.path.join(data_folder_path, 'train_stays.txt'), 'r') as f:
            stays = [int(line.strip()) for line in f]

    elif data_type == "test":
        with open(os.path.join(data_folder_path, 'test_stays.txt'), 'r') as f:
            stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(stays)]

    data = _correct_outliers(data)
    data = _impute_first_row(data, medians_48h)
    data = _apply_nan_rules(data, NAN_RULES)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    stays_to_process = set(data["stay_id"])

    all_data = []
    last_48h = []
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
            last_48h = {stay_id: data[-1:, :, :] for stay_id, data in all_data}
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
            last_48h.append((aux[0], aux[1][-1:, :, :]))

    all_data = dict(all_data)
    last_48h = dict(last_48h)
    outcomes = data[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"].to_dict()

    print("Total stays with >={}h of data: {}.".format(N_TIME_OFFSETS, len(stays_to_process)))

    if data_type == "train":
        with open(os.path.join(data_folder_path, "lstm_last_48h_train.pkl"), "wb") as fp:
            pickle.dump(last_48h, fp)
        with open(os.path.join(data_folder_path, "icu_expire_flag_train.pkl"), "wb") as fp:
            pickle.dump(outcomes, fp)

    elif data_type == "test":
        with open(os.path.join(data_folder_path, "lstm_last_48h_test.pkl"), "wb") as fp:
            pickle.dump(last_48h, fp)
        with open(os.path.join(data_folder_path, "icu_expire_flag_test.pkl"), "wb") as fp:
            pickle.dump(outcomes, fp)

def generate_retrain_discharge_dataset(base_path: str, dataset_filename: str, data_type: str):
    """
    Generates and saves discharge prediction datasets for patients based on their length of stay (LOS) and other features.
    This function processes patient data to create a dataset for predicting discharge within the next 48 hours.
    It filters patients with sufficient LOS, computes additional features, and labels time points where discharge
    is imminent. The processed dataset and corresponding outcomes are saved as pickle files in the specified base path.
    Args:
        base_path (str): The directory path where the input data is located and output files will be saved.
    Saves:
        lstm_disch_3point_48h_train.pkl: Pickled dataset for LSTM discharge prediction retrain.
        outcome_disch_3point_48h_train.pkl: Pickled outcomes for discharge prediction retrain.
        lstm_disch_3point_48h_test.pkl: Pickled dataset for LSTM discharge prediction test.
        outcome_disch_3point_48h_test.pkl: Pickled outcomes for discharge prediction test.
    """

    data_folder_path = os.path.join(base_path, "data")
    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    if data_type == "train":
        with open(os.path.join(data_folder_path, 'train_stays.txt'), 'r') as f:
            stays = [int(line.strip()) for line in f]

    elif data_type == "test":
        with open(os.path.join(data_folder_path, 'test_stays.txt'), 'r') as f:
            stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(stays)]

    data = _correct_outliers(data)
    data = _impute_first_row(data, medians_48h)
    data = _apply_nan_rules(data, NAN_RULES)

    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "half_los"] = np.floor(data["los"] / 2)
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - 47, "disch_48h"] = 1

    if data_type == "train":
        all_data, outcomes = _process_discharge_dataset(data)
        with open(os.path.join(data_folder_path, "lstm_disch_3point_48h_train.pkl"), "wb") as fp:
            pickle.dump(all_data, fp)
        with open(os.path.join(data_folder_path, "outcome_disch_3point_48h_train.pkl"), "wb") as fp:
            pickle.dump(outcomes, fp)

    elif data_type == "test":
        all_data, outcomes = _process_discharge_dataset(data)
        with open(os.path.join(data_folder_path, "lstm_disch_3point_48h_test.pkl"), "wb") as fp:
            pickle.dump(all_data, fp)
        with open(os.path.join(data_folder_path, "outcome_disch_3point_48h_test.pkl"), "wb") as fp:
            pickle.dump(outcomes, fp)

def retrain_mortality_model(base_path: str, model_filename: str, normalizer_name: Optional[str] = "", retrain_type: Optional[str] = "full"):
    
    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    data_folder_path = os.path.join(base_path, "data")
    normalizer_folder_path = os.path.join(base_path, "normalizers")

    try:
        with open(os.path.join(data_folder_path, "lstm_last_48h_train.pkl"), "rb") as fp:
            data = pickle.load(fp)

        with open(os.path.join(data_folder_path, "icu_expire_flag_train.pkl"), "rb") as fp:
            outcome = pickle.load(fp)

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f" Error loading dataset files: {e}", file=sys.stderr)
        print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
        print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
        sys.exit(1)
        
    X_train, X_test, y_train, y_test =_get_retrain_test_split_mort(data, outcome)
    X_train = \
        _normalize(
            normalizer_folder_path,
            X_train,
            load=True,
            save=False,
            filename=normalizer_name,
            model="mortality"
        )
    X_test = \
        _normalize(
            normalizer_folder_path,
            X_test,
            load=True,
            save=False,
            filename=normalizer_name,
            model="mortality"
        )

    X_train, y_train_v2, X_test, y_test_v2, X_val, y_val, y_val_v2 = \
        _prepare_mort_data(X_train, X_test, y_train, y_test)

    X_val = np.concatenate([X_test, X_val], axis=0)
    y_val_v2 = np.concatenate([y_test_v2, y_val_v2], axis=0)

    # Retrain all the layers
    if retrain_type=="full":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

    # Freeze LSTM layer
    elif retrain_type=="dense":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        model_lstm.layers[0].trainable = False

    # Freeze all layers except the first one (LSTM layer)
    elif retrain_type=="lstm":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

        for layer in model_lstm.layers[1:]:
            layer.trainable = False

    # Train model from scratch
    elif retrain_type=="zero":
        model_lstm = _create_lstm_mort_model(X_train.shape)
    
    low_lr_optimizer = Adam(learning_rate=LEARNING_RATE_MORT)

    model_lstm.compile(
        loss=BinaryFocalCrossentropy(apply_class_balancing=True),
        optimizer=low_lr_optimizer,
        weighted_metrics=[
            "accuracy",
            "AUC",
            "precision",
            "recall",
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.F1Score(name="f1_score", average="weighted"),
        ],
    )

    _fit_and_eval_model_retrain(model_lstm, 'RETRAIN_MORT', 
                            X_train,
                            y_train_v2,
                            X_val, 
                            y_val_v2, 
                            model_type='mortality',
                            epochs=1000, 
                            batch_size=100
                            )

    model_lstm.save(os.path.join(model_path, 'RETRAINED_' + retrain_type + '_' + model_filename))

def retrain_discharge_model(base_path: str, model_filename: str, normalizer_name: Optional[str] = "", retrain_type: Optional[str] = "full"):

    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    data_folder_path = os.path.join(base_path, "data")
    normalizer_folder_path = os.path.join(base_path, "normalizers")
    
    try:
        with open(os.path.join(data_folder_path, "lstm_disch_3point_48h_train.pkl"), "rb") as fp:
            data = pickle.load(fp)

        with open(os.path.join(data_folder_path, "outcome_disch_3point_48h_train.pkl"), "rb") as fp:
            outcome = pickle.load(fp)

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f" Error loading dataset files: {e}", file=sys.stderr)
        print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
        print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
        sys.exit(1)
        
    X_train, X_test, y_train, y_test =_get_retrain_test_split_disch(data, outcome)
    X_train = \
        _normalize(
            normalizer_folder_path,
            X_train,
            load=True,
            save=False,
            filename=normalizer_name,
            model="discharge"
        )
    X_test = \
        _normalize(
            normalizer_folder_path,
            X_test,
            load=True,
            save=False,
            filename=normalizer_name,
            model="discharge"
        )

    X_train, y_train_v2, X_test, y_test_v2, X_val, y_val, y_val_v2 = \
        _prepare_disch_data(X_train, X_test, y_train, y_test)

    X_val = np.concatenate([X_test, X_val], axis=0)
    y_val_v2 = np.concatenate([y_test_v2, y_val_v2], axis=0)

    # Retrain all the layers
    if retrain_type=="full":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

    # Freeze LSTM layer
    elif retrain_type=="dense":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        model_lstm.layers[0].trainable = False
    
    # Freeze all layers except the first one (LSTM layer)
    elif retrain_type=="lstm":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

        for layer in model_lstm.layers[1:]:
            layer.trainable = False

    # Train model from scratch
    elif retrain_type=="zero":
        model_lstm = _create_lstm_disch_model(X_train.shape)
    
    low_lr_optimizer = Adam(learning_rate=LEARNING_RATE_DISCH)

    model_lstm.compile(
        loss="binary_crossentropy",
        optimizer=low_lr_optimizer,
        weighted_metrics=[
            "accuracy",
            "AUC",
            "precision",
            "recall",
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.F1Score(name="f1_score", average="weighted"),
        ],
    )

    _fit_and_eval_model_retrain(model_lstm, 'RETRAIN_DISCH', 
                            X_train,
                            y_train_v2,
                            X_val, 
                            y_val_v2, 
                            model_type='discharge',
                            epochs=1000, 
                            batch_size=100
                            )

    model_lstm.save(os.path.join(model_path, 'RETRAINED_' + retrain_type + '_' + model_filename))

def generate_inference_dataset(base_path: str, dataset_filename: str, test_type: Optional[str] = "inference"):
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
        - Assumes existence of helper functions: _correct_outliers, _impute_first_row, _apply_nan_rules, _process_stay, tqdm_joblib.
    """

    if test_type not in ["retrain", "inference", "last_48h", "last_96h", "first_48h"]:
        raise ValueError(f"Invalid test_type: '{test_type}'. Must be 'retrain', 'inference', 'last_48h', 'last_96h' or 'first_48h'.")

    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    with open(os.path.join(data_folder_path, 'test_stays.txt'), 'r') as f:
        test_stays = [int(line.strip()) for line in f]
    data = data[data.stay_id.isin(test_stays)]

    # lista_pacientes = data.stay_id.unique()
    # data = data[data.stay_id.isin(lista_pacientes[:20])]

    data = _correct_outliers(data)
    data = _impute_first_row(data, medians_48h)
    data = _apply_nan_rules(data, NAN_RULES)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

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

    if test_type == "last_48h":
        all_data = {k: v[-48-1:] for k, v in all_data.items()}
        disch_outcomes = {k: v[-48-1:] for k, v in disch_outcomes.items()}
    elif test_type == "last_96h":
        all_data = {k: v[-96-1:] for k, v in all_data.items()}
        disch_outcomes = {k: v[-96-1:] for k, v in disch_outcomes.items()}
    elif test_type == "first_48h":
        all_data = {k: v[:48] for k, v in all_data.items()}
        disch_outcomes = {k: v[:48] for k, v in disch_outcomes.items()}      

    print("Total stays with >={}h of data: {}.".format(N_TIME_OFFSETS, len(stays_to_process)))

    return {
        "data": all_data,
        "mortality_outcome": outcomes,
        "disch_outcome": disch_outcomes,
    }

def get_test_dataset(base_path: str, normalizer_filename: str, data_type: str):

    data_folder_path = os.path.join(base_path, "data")

    if data_type == "mortality":
        try:
            with open(os.path.join(data_folder_path, "lstm_disch_3point_48h_test.pkl"), "rb") as fp:
                data = pickle.load(fp)

            with open(os.path.join(data_folder_path, "icu_expire_flag_test.pkl"), "rb") as fp:
                outcome = pickle.load(fp)

            return {
                "data": data,
                "mortality_outcome": outcome,
            }

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f" Error loading dataset files: {e}", file=sys.stderr)
            print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
            print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
            sys.exit(1)

    if data_type == "discharge":
        try:
            with open(os.path.join(data_folder_path, "lstm_disch_3point_48h_test.pkl"), "rb") as fp:
                data = pickle.load(fp)

            with open(os.path.join(data_folder_path, "outcome_disch_3point_48h_test.pkl"), "rb") as fp:
                outcome = pickle.load(fp)

            return {
                "data": data,
                "disch_outcome": outcome,
            }

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f" Error loading dataset files: {e}", file=sys.stderr)
            print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
            print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
            sys.exit(1)

def generate_production_dataset(base_path: str, dataset_filename: str):
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
        - Assumes existence of helper functions: _correct_outliers, _impute_first_row, _apply_nan_rules, _process_stay, tqdm_joblib.
    """

    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    data = _correct_outliers(data)
    data = _impute_first_row(data, medians_48h)
    data = _apply_nan_rules(data, NAN_RULES)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

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

    print("Total stays with >={}h of data: {}.".format(N_TIME_OFFSETS, len(stays_to_process)))

    return {
        "data": all_data
    }

def get_mortality_predictions(base_path: str, dataset: dict, mortality_model_filename: str, normalizer_filename: str, test_type: str) -> Dict[np.ndarray, np.ndarray]:
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

    model_folder_path = os.path.join(base_path, "models")
    model_path = os.path.join(model_folder_path, mortality_model_filename)
    normalizer_folder_path = os.path.join(base_path, "normalizers")

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
        filename=normalizer_filename,
        model="mortality"
    )

    X = np.nan_to_num(X)
    model_lstm = keras.models.load_model(model_path, custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
    return {"y_true": y, "y_pred": model_lstm.predict(X)}

def get_discharge_predictions(base_path: str, dataset: Optional[dict], discharge_model_filename: str, normalizer_filename: str, test_type: str) -> Dict[np.ndarray, np.ndarray]:
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

    model_folder_path = os.path.join(base_path, "models")
    model_path = os.path.join(model_folder_path, discharge_model_filename)
    normalizer_folder_path = os.path.join(base_path, "normalizers")

    X = np.vstack(list(dataset["data"].values()))[:, :, 1:]
    y = [np.array(item) for item in list(dataset["disch_outcome"].values())]

    X = _normalize(
        normalizer_folder_path,
        X,
        load=True,
        save=False,
        filename=normalizer_filename,
        model="mortality"
    )

    X = np.nan_to_num(X)
    model_lstm = keras.models.load_model(model_path, custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
    return {"y_true": y, "y_pred": model_lstm.predict(X)}

def process_predictions(mortality_pred: np.ndarray, discharge_pred: np.ndarray, mortality_groundtruth: np.ndarray, dataset: dict, base_path: str):
            
    data_folder_path = os.path.join(base_path, "data")

    with open(os.path.join(data_folder_path, f"model_parameters_test.json"), "r") as f:
        model_parameters = json.load(f)

    opt_th_mort = model_parameters['th_mort']
    opt_th_disch = model_parameters['th_disch']
    min_prob = model_parameters['min_prob']
    max_prob = model_parameters['max_prob']

    df_prob = pd.DataFrame({
        'mortality_prob': mortality_pred,
        'disch_prob': discharge_pred
    })

    df_prob['range'] = np.where(df_prob['mortality_prob']<opt_th_mort, (opt_th_mort-min_prob), max_prob-opt_th_mort)
    df_prob['substract'] = np.where(df_prob['mortality_prob']<opt_th_mort, df_prob['mortality_prob']-min_prob, df_prob['mortality_prob']-opt_th_mort)
    df_prob['normalized'] = np.where(df_prob['mortality_prob']<opt_th_mort, ((df_prob['substract']/df_prob['range'])/2)*100, ((df_prob['substract']/df_prob['range']+1)/2)*100)

    df_prob['disch_prob_cat'] = np.where(df_prob['disch_prob']>opt_th_disch, 1, 0)
    df_prob['mortality_prob_cat'] = np.where(df_prob['mortality_prob']>opt_th_mort, 1, 0)

    df_prob['color_group'] = np.where((df_prob['mortality_prob_cat']==1) & (df_prob['disch_prob_cat']==1), 'EXITUS <48h',
                             np.where((df_prob['mortality_prob_cat']==1) & (df_prob['disch_prob_cat']==0), 'EXITUS >48h',
                                     np.where((df_prob['mortality_prob_cat']==0) & (df_prob['disch_prob_cat']==0), 'ALIVE >48h', 'ALIVE <48h')))

    stay_ids = [
        stay_id
        for stay_id, arr in dataset['data'].items()
        for _ in range(len(arr))
    ]
    df_prob['stay_id'] = stay_ids
    df_prob['hr'] = df_prob.groupby('stay_id').cumcount() + 48
    df_prob['mortality_groundtruth'] = mortality_groundtruth

    df_prob.to_pickle(os.path.join(data_folder_path, "probs.pkl"))

    return df_prob

def calculate_errors(base_path, dataset_filename, disch_dataset, mort_dataset, mortality_pred, mortality_groundtruth, discharge_pred, discharge_groundtruth, th_mort, th_disch, min_prob, max_prob):
    
    data_folder_path = os.path.join(base_path, "data")
    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, 'test_stays.txt'), 'r') as f:
        test_stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(test_stays)]
    data = data.drop_duplicates(["stay_id", "hr"], keep="last").reset_index(drop=True)
    
    exitus = data[['stay_id','icu_expire_flag']].drop_duplicates()

    stay_ids = []

    for stay_id in disch_dataset['data'].keys():
        num_rows = disch_dataset['data'][stay_id].shape[0]
        stay_ids.extend([stay_id] * num_rows)

    df = pd.DataFrame({
    'stay_id': stay_ids,
    'mortality_prob': mortality_pred,
    'mortality_gt': mortality_groundtruth,
    'disch_prob': discharge_pred,
    'disch_gt': discharge_groundtruth,
    })
    
    df['range'] = np.where(df['mortality_prob']<th_mort, (th_mort-min_prob), max_prob-th_mort)
    df['substract'] = np.where(df['mortality_prob']<th_mort, df['mortality_prob']-min_prob, df['mortality_prob']-th_mort)
    df['normalized'] = np.where(df['mortality_prob']<th_mort, ((df['substract']/df['range'])/2)*100, ((df['substract']/df['range']+1)/2)*100)

    df = pd.merge(df, exitus[['stay_id','icu_expire_flag']], on='stay_id', how='left')
    df['disch_prob_cat'] = np.where(df['disch_prob']>th_disch, 1, 0)
    df['mortality_prob_cat'] = np.where(df['mortality_prob']>th_mort, 1, 0)

    df['color_group'] = np.where((df['mortality_prob_cat']==1) & (df['disch_prob_cat']==1), 'EXITUS <48h',
                                    np.where((df['mortality_prob_cat']==1) & (df['disch_prob_cat']==0), 'EXITUS >48h',
                                            np.where((df['mortality_prob_cat']==0) & (df['disch_prob_cat']==0), 'ALIVE >48h', 'ALIVE <48h')))
        
    df['real_color_group'] = np.where((df['icu_expire_flag']==1) & (df['disch_gt']==1), 'EXITUS <48h',
                            np.where((df['icu_expire_flag']==1) & (df['disch_gt']==0), 'EXITUS >48h',
                                    np.where((df['icu_expire_flag']==0) & (df['disch_gt']==0), 'ALIVE >48h', 'ALIVE <48h')))
    
    df['error'] = np.where(((df['real_color_group']=='EXITUS <48h') & (df['color_group']=='ALIVE <48h')) | (
        (df['color_group']=='EXITUS <48h') & (df['real_color_group']=='ALIVE <48h')), 3,
        np.where(((df['real_color_group']=='EXITUS <48h') & (df['color_group']=='ALIVE >48h')) | (
            (df['color_group']=='EXITUS <48h') & (df['real_color_group']=='ALIVE >48h')) | (
            (df['color_group']=='EXITUS >48h') & (df['real_color_group']=='ALIVE <48h')) | (
            (df['real_color_group']=='EXITUS <48h') & (df['color_group']=='ALIVE >48h')) | (
            (df['real_color_group']=='EXITUS >48h') & (df['color_group']=='ALIVE <48h')), 2, 
        np.where(((df['real_color_group']=='EXITUS <48h') & (df['color_group']=='EXITUS >48h')) | (
            (df['color_group']=='EXITUS <48h') & (df['real_color_group']=='EXITUS >48h')) | (
            (df['color_group']=='ALIVE >48h') & (df['real_color_group']=='ALIVE <48h')) | (
            (df['real_color_group']=='ALIVE >48h') & (df['color_group']=='ALIVE <48h')) | (
            (df['real_color_group']=='ALIVE >48h') & (df['color_group']=='EXITUS >48h')) | (
            (df['color_group']=='ALIVE >48h') & (df['real_color_group']=='EXITUS >48h')), 1, 0)))
    
    df['max_error_possible'] = np.where(df['real_color_group'].isin(['EXITUS <48h','ALIVE <48h']), 3, 2)

    df.to_csv(os.path.join(data_folder_path, "final_result.csv"), index=False)
    
    return df

def run_pipeline(dataset_filename:str, mode: str):

    if mode == "generate_files":
        generate_medians_json(BASE_PATH, dataset_filename)
        generate_train_test_split_file(BASE_PATH, dataset_filename)
        dataset = generate_normalizer_dataset(BASE_PATH, dataset_filename)
        generate_mortality_normalizer(BASE_PATH, dataset)
        generate_discharge_normalizer(BASE_PATH, dataset)

    elif mode == "generate_retrain_data":
        generate_retrain_mortality_dataset(BASE_PATH, dataset_filename, "train")
        generate_retrain_mortality_dataset(BASE_PATH, dataset_filename, "test")
        generate_retrain_discharge_dataset(BASE_PATH, dataset_filename, "train")
        generate_retrain_discharge_dataset(BASE_PATH, dataset_filename, "test")

    elif mode == "retrain_models":
        # retrain_mortality_model(BASE_PATH, RETRAIN_MORT_MODEL, MORT_NORMALIZER, RETRAIN_TYPE)
        retrain_discharge_model(BASE_PATH, RETRAIN_DISCH_MODEL, DISCH_NORMALIZER, RETRAIN_TYPE)

    elif mode == "calculate_metrics":

        data_folder_path = os.path.join(BASE_PATH, "data")
            
        disch_dataset = get_test_dataset(BASE_PATH, DISCH_NORMALIZER, "discharge")
        mort_dataset = get_test_dataset(BASE_PATH, MORT_NORMALIZER, "mortality")

        discharge_outcome = get_discharge_predictions(BASE_PATH, disch_dataset, INFERENCE_DISCH_MODEL, DISCH_NORMALIZER, TEST_TYPE)
        mortality_outcome = get_mortality_predictions(BASE_PATH, mort_dataset, INFERENCE_MORT_MODEL, MORT_NORMALIZER, TEST_TYPE)

        mortality_pred = mortality_outcome['y_pred'][:, 1]
        mortality_groundtruth = mortality_outcome['y_true'][:, 0]

        discharge_pred = discharge_outcome['y_pred'][:, 1]
        discharge_groundtruth = np.concatenate(discharge_outcome['y_true'], axis=0).ravel()

        opt_th_mort, opt_th_disch = _plot_roc_combined(BASE_PATH,
                                            mortality_pred, mortality_groundtruth, 
                                            discharge_pred, discharge_groundtruth, 
                                            threshold_method='min_distance',
                                            save=True
                                        )

        min_prob = np.min(mortality_pred)
        max_prob = np.max(mortality_pred)

        model_parameters = {
            'th_disch': float(opt_th_disch),
            'th_mort': float(opt_th_mort),
            'min_prob': float(min_prob),
            'max_prob': float(max_prob),
        }

        with open(os.path.join(data_folder_path, f"model_parameters_test.json"), "w") as f:
            json.dump(model_parameters, f, indent=4)
        
        df_prob = process_predictions(mortality_pred, discharge_pred, mortality_groundtruth, mort_dataset, BASE_PATH)

        error_results = calculate_errors(BASE_PATH, dataset_filename, disch_dataset, mort_dataset, mortality_pred, mortality_groundtruth, discharge_pred, discharge_groundtruth, opt_th_mort, opt_th_disch, min_prob, max_prob)
        _plot_error(error_results, BASE_PATH)

    elif mode == "inference":
    
        check_dataset(BASE_PATH, dataset_filename)
        dataset = generate_inference_dataset(BASE_PATH, dataset_filename, TEST_TYPE)

        discharge_outcome = get_discharge_predictions(BASE_PATH, dataset, INFERENCE_DISCH_MODEL, DISCH_NORMALIZER, TEST_TYPE)
        mortality_outcome = get_mortality_predictions(BASE_PATH, dataset, INFERENCE_MORT_MODEL, MORT_NORMALIZER, TEST_TYPE)
    
        mortality_pred = mortality_outcome['y_pred'][:, 1]
        mortality_groundtruth = mortality_outcome['y_true'][:, 0]

        discharge_pred = discharge_outcome['y_pred'][:, 1]
        discharge_groundtruth = np.hstack(discharge_outcome['y_true'])

        df_prob = process_predictions(mortality_pred, discharge_pred, mortality_groundtruth, dataset, BASE_PATH)
        _plot_inference(df_prob, BASE_PATH, test_type=TEST_TYPE, save=True)

# project root path
BASE_PATH='./'

# normalizers used for retrain and evaluation
MORT_NORMALIZER='mimic_iv_normalizer.pkl' # custom_normalizer.pkl
DISCH_NORMALIZER='mimic_iv_normalizer_disch.pkl' # custom_normalizer_disch.pkl

# retrain type (zero, full, dense, lstm)
RETRAIN_TYPE='zero'

# models name that we want to retrain
RETRAIN_MORT_MODEL='lstm_mortality_model.keras'
RETRAIN_DISCH_MODEL='lstm_disch_model.keras'

# models that we want to evaluate in inference mode
INFERENCE_MORT_MODEL='RETRAINED_zero_lstm_mortality_model.keras'
INFERENCE_DISCH_MODEL='RETRAINED_zero_lstm_disch_model.keras'

# inference type (full, last_48h, last_98h, first_48h)
TEST_TYPE='full'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LSTM test pipeline")
    parser.add_argument("--csv_filename", type=str, help="CSV dataset filename (e.g., jx_data.csv)")
    parser.add_argument("--mode", type=str, help="Mode type (generate_files, generate_retrain_data, retrain_models, calculate_metrics, inference)")
    args = parser.parse_args()

    # Check if both arguments are provided
    if not args.csv_filename or not args.mode:
        parser.error("Both --csv_filename and --mode are required arguments.")

    result = run_pipeline(args.csv_filename, args.mode)