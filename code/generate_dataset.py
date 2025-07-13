import contextlib
import logging
import os
import pickle

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .get_data import NAN_RULES, apply_nan_rules, correct_outliers

LOGGER = logging.getLogger(__name__)


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
PARALLEL = True
N_TIME_OFFSETS = 48


def _get_data(base_path):
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

    data = pd.read_csv(os.path.join(base_path, "mimic_iv_nan_corrected.csv"), index_col=0)
    data.loc[:, "outcome"] = data["icu_expire_flag"]

    exclude_patients = pd.read_csv(os.path.join(base_path, "patients_ltsv.csv"))
    LOGGER.debug("There are {:,.0f} patients".format(len(set(data["stay_id"]))))
    data = data[~data["stay_id"].isin(exclude_patients["stay_id"])].copy()
    LOGGER.debug("We will be using {:,.0f} patients".format(len(set(data["stay_id"]))))

    outcomes = data[["stay_id", "outcome"]].drop_duplicates().set_index("stay_id")["outcome"].to_dict()
    with open(os.path.join(base_path, "mimic_iv_outcome.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)

    outcomes = data[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"].to_dict()
    with open(os.path.join(base_path, "mimic_iv_icu_expire_flag.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)

    outcomes = (
        data[["stay_id", "hospital_expire_flag"]]
        .drop_duplicates()
        .set_index("stay_id")["hospital_expire_flag"]
        .to_dict()
    )
    with open(os.path.join(base_path, "mimic_iv_hospital_expire_flag.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)

    los = data.groupby("stay_id")["hr"].max().to_dict()
    data.loc[:, "los"] = data["stay_id"].map(los)
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


def generate_mortality_dataset(base_path):
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

    data = _get_data(base_path)
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
    with open(os.path.join(base_path, "mimic_iv_lstm_48h.pkl"), "wb") as fp:
        pickle.dump(all_data, fp)
    with open(os.path.join(base_path, "mimic_iv_lstm_last_48h.pkl"), "wb") as fp:
        pickle.dump(last_48h, fp)

    outcomes = data[["stay_id", "outcome"]].drop_duplicates().set_index("stay_id")["outcome"].to_dict()
    with open(os.path.join(base_path, "mimic_iv_outcome.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)
    outcomes = data[["stay_id", "icu_expire_flag"]].drop_duplicates().set_index("stay_id")["icu_expire_flag"].to_dict()
    with open(os.path.join(base_path, "mimic_iv_icu_expire_flag.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)
    outcomes = (
        data[["stay_id", "hospital_expire_flag"]]
        .drop_duplicates()
        .set_index("stay_id")["hospital_expire_flag"]
        .to_dict()
    )
    with open(os.path.join(base_path, "mimic_iv_hospital_expire_flag.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)


def generate_discharge_dataset(base_path):
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

    data = _get_data(base_path)
    data = data[(data["los"] >= N_TIME_OFFSETS)].copy()
    data.loc[:, "half_los"] = np.floor(data["los"] / 2)
    data.loc[:, "disch_48h"] = 0
    data.loc[data["hr"] >= data["los"] - 47, "disch_48h"] = 1

    all_data, outcomes = _process_discharge_dataset(data)
    with open(os.path.join(base_path, "mimic_iv_lstm_disch_3point_48h.pkl"), "wb") as fp:
        pickle.dump(all_data, fp)
    with open(os.path.join(base_path, "mimic_iv_outcome_disch_3point_48h.pkl"), "wb") as fp:
        pickle.dump(outcomes, fp)


def generate_test_dataset(base_path, dataset_filename):
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

    data = pd.read_csv(os.path.join(base_path, dataset_filename))
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
