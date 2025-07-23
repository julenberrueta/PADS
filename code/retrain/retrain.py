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
from keras.optimizers import Adam
import keras.layers as layers
from keras.losses import BinaryFocalCrossentropy
from sklearn.model_selection import train_test_split
import argparse
import sys
import json

from ..inference.inference import check_dataset, _process_stay, tqdm_joblib, softmax_temperature, _normalize, correct_outliers, apply_nan_rules, NAN_RULES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from ..discharge_model import _prepare_data as _prepare_disch_data
from ..discharge_model import _get_weights as _get_disch_weights
from ..mortality_model import _prepare_data as _prepare_mort_data
from ..mortality_model import _get_weights as _get_mort_weights


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

def _get_data(base_path, dataset_filename):
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
    data = pd.read_csv(data_path, index_col=0)
    data.loc[:, "outcome"] = data["icu_expire_flag"]

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)

    data_48h = data[data.hr <= 48]

    medians_48h = data_48h[IMPUTE_FIRST_ROW].median().to_dict()
    with open(os.path.join(data_folder_path, "medians_48h.json"), "w") as f:
        json.dump(medians_48h, f, indent=4)

    return data

def _process_discharge_dataset(data):
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
    features = [
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
    fixed_cols = ["stay_id"]

    outcomes = {}
    all_data = {}
    for stay_id in tqdm(set(first_datapoint["stay_id"])):
        all_data[stay_id] = (
            first_datapoint.loc[first_datapoint["stay_id"] == stay_id, fixed_cols + features]
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
            aux = datapoint.loc[datapoint["stay_id"] == stay_id, fixed_cols + features].to_numpy().reshape(1, 48, 17)
            aux_outcome = datapoint.loc[datapoint["stay_id"] == stay_id, "disch_48h"].to_numpy()[-1:].reshape(-1, 1)
            if stay_id in all_data:
                all_data[stay_id] = np.vstack([all_data[stay_id], aux])
                outcomes[stay_id] = np.vstack([outcomes[stay_id], aux_outcome])
            else:
                all_data[stay_id] = aux
                outcomes[stay_id] = aux_outcome
    return all_data, outcomes

def impute_first_row(df: pd.DataFrame, medians_dict: dict) -> pd.DataFrame:
    """
    Imputes missing values in the specified columns:
    1. Uses the first available value per stay_id if present.
    2. If missing values remain, fills them with the provided global median.
    
    Parameters:
        df: Original DataFrame with potential NaNs
        medians_dict: Dictionary containing global medians for the columns
    
    Returns:
        Imputed DataFrame
    """
    cols = list(medians_dict.keys())

    first_values = (
        df.sort_values('hr')
          .groupby('stay_id')[cols]
          .first()
          .reset_index()
    )

    df_merged = df.merge(first_values, on="stay_id", suffixes=("", "_first"))

    for col in cols:
        df_merged[col] = df_merged[col].fillna(df_merged[f"{col}_first"])
        df_merged.drop(columns=[f"{col}_first"], inplace=True)

    for col in cols:
        df_merged[col] = df_merged[col].fillna(medians_dict[col])

    return df_merged

def generate_train_test_split_file(dataset_filename, base_path):
    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = pd.read_csv(data_path)

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

    stay_ids = data.stay_id.unique()
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    # Guardar en archivos de texto
    with open(os.path.join(data_folder_path, 'train_stays.txt'), 'w') as f:
        for stay in train_stays:
            f.write(f"{stay}\n")

    with open(os.path.join(data_folder_path, 'test_stays.txt'), 'w') as f:
        for stay in test_stays:
            f.write(f"{stay}\n")

def generate_normalizer_dataset(dataset_filename, base_path):
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

    data_folder_path = os.path.join(base_path, "data")
    data_path = os.path.join(data_folder_path, dataset_filename)

    data = pd.read_csv(data_path)

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)

    fixed_cols = ["stay_id"]
    time_column = "hr"
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - (N_TIME_OFFSETS - 1), "disch_48h"] = 1
    stays_to_process = set(data["stay_id"])

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    data = correct_outliers(data)
    data = impute_first_row(data, medians_48h)
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

def generate_mortality_dataset(base_path, dataset_filename):
    """
    Generates and saves datasets for mortality prediction from ICU stay data.
    This function processes ICU stay data to create datasets suitable for LSTM-based mortality prediction.
    It filters stays with sufficient length, processes each stay (optionally in parallel), and saves the
    processed data and outcome labels as pickle files.
    Args:
        base_path (str): The base directory path where input data is located and output files will be saved.
    Saves:
        - mimic_iv_lstm_48h.pkl: Dictionary mapping stay_id to processed time-series data for each stay.
        - mimic_iv_lstm_last_48h.pkl: Dictionary mapping stay_id to the last 48 hours of data for each stay.
        - mimic_iv_outcome.pkl: Dictionary mapping stay_id to outcome label.
        - mimic_iv_icu_expire_flag.pkl: Dictionary mapping stay_id to ICU expire flag.
        - mimic_iv_hospital_expire_flag.pkl: Dictionary mapping stay_id to hospital expire flag.
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

    with open(os.path.join(data_folder_path, 'train_stays.txt'), 'r') as f:
        train_stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(train_stays)]

    data = correct_outliers(data)
    data = impute_first_row(data, medians_48h)
    data = apply_nan_rules(data, NAN_RULES)

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

    print("Total stays with >={}h of data: {}.".format(N_TIME_OFFSETS, len(stays_to_process)))
    with open(os.path.join(data_folder_path, "lstm_48h.pkl"), "wb") as fp:
        pickle.dump(all_data, fp)
    with open(os.path.join(data_folder_path, "lstm_last_48h.pkl"), "wb") as fp:
        pickle.dump(last_48h, fp)

    outcomes = data[["stay_id", "outcome"]].drop_duplicates().set_index("stay_id")["outcome"].to_dict()
    with open(os.path.join(data_folder_path, "outcome.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)
    outcomes = data[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"].to_dict()
    with open(os.path.join(data_folder_path, "icu_expire_flag.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)

def generate_discharge_dataset(base_path, dataset_filename):
    """
    Generates and saves discharge prediction datasets for patients based on their length of stay (LOS) and other features.
    This function processes patient data to create a dataset for predicting discharge within the next 48 hours.
    It filters patients with sufficient LOS, computes additional features, and labels time points where discharge
    is imminent. The processed dataset and corresponding outcomes are saved as pickle files in the specified base path.
    Args:
        base_path (str): The directory path where the input data is located and output files will be saved.
    Saves:
        mimic_iv_lstm_disch_3point_48h.pkl: Pickled dataset for LSTM discharge prediction.
        mimic_iv_outcome_disch_3point_48h.pkl: Pickled outcomes for discharge prediction.
    """

    data_folder_path = os.path.join(base_path, "data")
    data = _get_data(base_path, dataset_filename)

    with open(os.path.join(data_folder_path, "medians_48h.json"), "r") as f:
        medians_48h = json.load(f)

    with open(os.path.join(data_folder_path, 'train_stays.txt'), 'r') as f:
        train_stays = [int(line.strip()) for line in f]

    data = data[data.stay_id.isin(train_stays)]

    data = correct_outliers(data)
    data = impute_first_row(data, medians_48h)
    data = apply_nan_rules(data, NAN_RULES)

    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "half_los"] = np.floor(data["los"] / 2)
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - 47, "disch_48h"] = 1

    all_data, outcomes = _process_discharge_dataset(data)
    with open(os.path.join(data_folder_path, "lstm_disch_3point_48h.pkl"), "wb") as fp:
        pickle.dump(all_data, fp)
    with open(os.path.join(data_folder_path, "outcome_disch_3point_48h.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)

def get_mortality_normalizer(base_path, dataset):

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
        filename="custom_normalizer_mort.pkl",
    )

def get_discharge_normalizer(base_path, dataset):

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

def _get_train_test_split_disch(data, outcome):
    
    # Obtener todos los ids de estancia (stay_id) del diccionario data
    stay_ids = list(data.keys())
    valid_stay_ids = [sid for sid in data if sid in outcome]

    # Dividir los stay_ids en entrenamiento y prueba
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    print(f"{len(train_stays)} patients for retraining discharge model")

    # Inicializar matrices para X_train, y_train, X_test, y_test
    X_train = np.ndarray((len(train_stays), 48, 16))
    y_train = np.ndarray((len(train_stays), 1))
    X_test = np.ndarray((len(test_stays), 48, 16))
    y_test = np.ndarray((len(test_stays), 1))

    # Llenar las matrices de entrenamiento
    offset = 0
    for stay_id in tqdm(train_stays):
        if stay_id not in data:
            continue
        for sample in range(data[stay_id].shape[0]):
            X_train[offset:offset+1, :, :] = data[stay_id][sample, :, 1:]
            y_train[offset:offset+1, :] = outcome[stay_id][sample]
            offset += 1

    print('X Train shape: ', X_train.shape)
    print('Y Train shape: ', y_train.shape)

    # Llenar las matrices de prueba
    offset = 0
    for stay_id in tqdm(test_stays):
        if stay_id not in data:
            continue
        for sample in range(data[stay_id].shape[0]):
            X_test[offset:offset+1, :, :] = data[stay_id][sample, :, 1:]
            y_test[offset:offset+1, :] = outcome[stay_id][sample]
            offset += 1

    print('X Test shape: ', X_test.shape)
    print('Y Test shape: ', y_test.shape)

    return X_train, X_test, y_train, y_test

def _get_train_test_split_mort(data, outcome):
    
    # Obtener todos los ids de estancia (stay_id) del diccionario data
    stay_ids = list(data.keys())

    # Dividir los stay_ids en conjuntos de entrenamiento y prueba
    train_stays, test_stays = train_test_split(stay_ids, test_size=0.2, random_state=42)

    print(f"{len(train_stays)} patients for retraining mortality model")

    # Inicializar matrices para X_train, y_train, X_test, y_test
    X_train = np.ndarray((len(train_stays), 48, 16))
    y_train = np.ndarray((len(train_stays), 1))
    X_test = np.ndarray((len(test_stays), 48, 16))
    y_test = np.ndarray((len(test_stays), 1))

    # Llenar las matrices de entrenamiento
    offset = 0
    for stay_id in tqdm(train_stays):
        if stay_id not in data:
            continue
        X_train[offset:offset+1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_train[offset:offset+1, :] = outcome[stay_id]
        offset += 1

    print('X Train shape: ', X_train.shape)
    print('Y Train shape: ', y_train.shape)

    # Llenar las matrices de prueba
    offset = 0
    for stay_id in tqdm(test_stays):
        if stay_id not in data:
            continue
        X_test[offset:offset+1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_test[offset:offset+1, :] = outcome[stay_id]
        offset += 1

    print('X Test shape: ', X_test.shape)
    print('Y Test shape: ', y_test.shape)

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


def _create_lstm_disch_model(input_shape):
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
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(),
        weighted_metrics=[
            "accuracy",
            "AUC",
            "precision",
            "recall",
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.F1Score(name="f1_score", average="weighted"),
        ],
    )

    model.summary()
    return model

def _create_lstm_mort_model(input_shape):
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

    #     model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(10, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    #     model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.add(keras.layers.Dense(2, name="logits"))
    # model.add(keras.layers.Activation(softmax_temperature))
    model.add(SoftmaxTemperature(temperature=1))
    model.compile(
        # loss="binary_crossentropy",
        loss=BinaryFocalCrossentropy(apply_class_balancing=True),
        optimizer=Adam(),
        weighted_metrics=[
            "accuracy",
            "AUC",
            "precision",
            "recall",
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            keras.metrics.F1Score(name="f1_score", average="weighted"),
        ],
    )

    model.summary()

    return model

def _fit_and_eval_model_retrain(model,
                        model_name,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        epochs=1000,
                        batch_size=100):

    logger = TensorBoard(
        log_dir=f'./models/tensorflow_logs/{model_name}', write_graph=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        f'./models/tensorflow_logs/{model_name}_best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    _ = model.fit(
        X_train,
        y_train,
        class_weight=_get_disch_weights(y_train),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=[logger, early_stopping, model_checkpoint]
    )

def retrain_mortality_model(base_path, model_filename, normalizer_name="", retrain_type="full"):
    
    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    data_path = os.path.join(base_path, "data")
    normalizer_path = os.path.join(base_path, "normalizers")

    try:
        with open(os.path.join(data_path, "lstm_last_48h.pkl"), "rb") as fp:
            data = pickle.load(fp)

        with open(os.path.join(data_path, "icu_expire_flag.pkl"), "rb") as fp:
            outcome = pickle.load(fp)

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f" Error loading dataset files: {e}", file=sys.stderr)
        print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
        print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
        sys.exit(1)
        
    X_train, X_test, y_train, y_test =_get_train_test_split_mort(data, outcome)
    X_train = \
        _normalize(
            normalizer_path,
            X_train,
            load=True,
            save=False,
            filename=normalizer_name,
            model="mortality"
        )
    X_test = \
        _normalize(
            normalizer_path,
            X_test,
            load=True,
            save=False,
            filename=normalizer_name,
            model="mortality"
        )

    X_train, y_train_v2, X_test, y_test_v2, X_val, y_val, y_val_v2 = \
        _prepare_disch_data(X_train, X_test, y_train, y_test)

    X_val = np.concatenate([X_test, X_val], axis=0)
    y_val_v2 = np.concatenate([y_test_v2, y_val_v2], axis=0)

    if retrain_type=="full":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

    elif retrain_type=="dense":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        # Freeze LSTM layer
        model_lstm.layers[0].trainable = False

    elif retrain_type=="lstm":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        # Freeze all layers except the first one (LSTM layer)
        for layer in model_lstm.layers[1:]:
            layer.trainable = False
    elif retrain_type=="zero":
        model_lstm = _create_lstm_mort_model(X_train.shape)
    
    low_lr_optimizer = Adam(learning_rate=LEARNING_RATE_MORT)

    model_lstm.compile(loss='binary_crossentropy',
        optimizer=low_lr_optimizer,
        metrics=['accuracy', 'AUC'])

    _fit_and_eval_model_retrain(model_lstm, f'RETRAIN_MORT', X_train,
                            y_train_v2,
                            X_val, y_val_v2, epochs=1000, batch_size=100)

    model_lstm.save(os.path.join(model_path, 'RETRAINED_' + retrain_type + '_' + model_filename))

def retrain_discharge_model(base_path, model_filename, normalizer_name="", retrain_type="full"):

    model_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, "results")
    data_path = os.path.join(base_path, "data")
    normalizer_path = os.path.join(base_path, "normalizers")
    
    try:
        with open(os.path.join(data_path, "lstm_disch_3point_48h.pkl"), "rb") as fp:
            data = pickle.load(fp)

        with open(os.path.join(data_path, "outcome_disch_3point_48h.pkl"), "rb") as fp:
            outcome = pickle.load(fp)

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f" Error loading dataset files: {e}", file=sys.stderr)
        print(" Please run the retraining pipeline to generate the datasets:", file=sys.stderr)
        print("   python -m code.retrain.retrain <dataset_filename> --generate_datasets", file=sys.stderr)
        sys.exit(1)
        
    X_train, X_test, y_train, y_test =_get_train_test_split_disch(data, outcome)
    X_train = \
        _normalize(
            normalizer_path,
            X_train,
            load=True,
            save=False,
            filename=normalizer_name,
            model="discharge"
        )
    X_test = \
        _normalize(
            normalizer_path,
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

    if retrain_type=="full":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})

    elif retrain_type=="dense":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        # Freeze LSTM layer
        model_lstm.layers[0].trainable = False

    elif retrain_type=="lstm":
        model_lstm = keras.models.load_model(os.path.join(model_path, model_filename), custom_objects={'softmax_temperature': softmax_temperature, 'SoftmaxTemperature': SoftmaxTemperature})
        # Freeze all layers except the first one (LSTM layer)
        for layer in model_lstm.layers[1:]:
            layer.trainable = False
    elif retrain_type=="zero":
        model_lstm = _create_lstm_disch_model(X_train.shape)
    
    low_lr_optimizer = Adam(learning_rate=LEARNING_RATE_DISCH)

    model_lstm.compile(loss='binary_crossentropy',
        optimizer=low_lr_optimizer,
        metrics=['accuracy', 'AUC'])

    _fit_and_eval_model_retrain(model_lstm, 'RETRAIN_DISCH', 
                            X_train,
                            y_train_v2,
                            X_val, 
                            y_val_v2, 
                            epochs=1000, 
                            batch_size=100
                            )

    model_lstm.save(os.path.join(model_path, 'RETRAINED_' + retrain_type + '_' + model_filename))

def run_retrain_pipeline(dataset_filename, base_path, normalizer_disch, normalizer_mort, mortality_model_filename, disch_model_filename, retrain_models=False, generate_datasets=False, generate_normalizers=False, retrain_type="full"):

    if generate_datasets:
        generate_train_test_split_file(dataset_filename, base_path)
        check_dataset(dataset_filename, base_path)
        generate_discharge_dataset(base_path, dataset_filename)
        generate_mortality_dataset(base_path, dataset_filename)

    if generate_normalizers:
        check_dataset(dataset_filename, base_path)
        dataset = generate_normalizer_dataset(dataset_filename, base_path)
        get_mortality_normalizer(base_path, dataset)
        get_discharge_normalizer(base_path, dataset)
        
    if retrain_models:
        retrain_discharge_model(base_path, disch_model_filename, normalizer_name=normalizer_disch, retrain_type=retrain_type)
        retrain_mortality_model(base_path, mortality_model_filename, normalizer_name=normalizer_mort, retrain_type=retrain_type)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run LSTM retrain pipeline")
    parser.add_argument("csv_filename", type=str, help="CSV dataset filename (e.g., jx_data.csv)")
    parser.add_argument("--retrain_models", action="store_true", help="Whether to retrain or not the models")
    parser.add_argument("--generate_datasets", action="store_true", help="Whether to generate the datasets before training")
    parser.add_argument("--generate_normalizers", action="store_true", help="Whether to generate the normalizers")
    parser.add_argument("--retrain_type", type=str, default="full", help="Retraining type")
    args = parser.parse_args()

    result = run_retrain_pipeline(
        args.csv_filename,
        './',
        'mimic_iv_normalizer_disch.pkl',
        'mimic_iv_normalizer.pkl',
        'lstm_mortality_model.keras',
        'lstm_discharge_model.keras',
        retrain_models=args.retrain_models,
        generate_datasets=args.generate_datasets,
        generate_normalizers=args.generate_normalizers,
        retrain_type=args.retrain_type
    )
