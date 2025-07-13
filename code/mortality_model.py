import datetime
import os
import pickle

import keras.layers as layers
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.losses import BinaryFocalCrossentropy
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tqdm import tqdm

N_FOLDS = 5


def _get_data(base_path):
    """
    Loads patient data and outcome labels from specified files, excluding stays listed in a CSV file.
    Args:
        base_path (str): The directory path containing the data files.
    Returns:
        tuple: A tuple containing:
            - data (dict): Dictionary of patient data, with excluded stays removed.
            - outcome (dict): Dictionary of patient outcome flags, with excluded stays removed.
    Files read:
        - mimic_iv_lstm_last_48h.pkl: Pickle file containing patient data.
        - mimic_iv_icu_expire_flag.pkl: Pickle file containing patient outcome flags.
        - patients_ltsv.csv: CSV file listing stay IDs to exclude.
    Side Effects:
        Prints the number of stays before and after exclusion.
    """

    with open(os.path.join(base_path, "mimic_iv_lstm_last_48h.pkl"), "rb") as fp:
        data = pickle.load(fp)

    with open(os.path.join(base_path, "mimic_iv_icu_expire_flag.pkl"), "rb") as fp:
        outcome = pickle.load(fp)

    exclude_stays = pd.read_csv(os.path.join(base_path, "patients_ltsv.csv"))
    exclude_stays = set(exclude_stays["stay_id"])
    print(len(data.keys()))
    for stay in exclude_stays:
        if stay in data:
            data.pop(stay)
        if stay in outcome:
            outcome.pop(stay)
    print(len(data.keys()))
    return data, outcome


def _get_folds(base_path, n_splits, generate=False, outcome=None, stay_ids=None):
    """
    Generates or loads stratified cross-validation folds for patient stays.
    Parameters
    ----------
    base_path : str
        The base directory path where the fold CSV file will be saved or loaded from.
    n_splits : int
        Number of folds for stratified cross-validation.
    generate : bool, optional
        If True, generates new folds and saves them to CSV. If False, loads existing folds from CSV.
    outcome : dict, optional
        Dictionary mapping stay IDs to outcome values. Required if generate is True.
    stay_ids : list or array-like, optional
        List of stay IDs to include in the folds. Required if generate is True.
    Returns
    -------
    folds : pandas.DataFrame
        DataFrame containing stay IDs, train/test split, and fold index.
    """

    if generate:
        assert outcome is not None
        assert stay_ids is not None
        aux = pd.DataFrame.from_dict(outcome, orient="index")
        aux = aux.reset_index().rename(columns={"index": "stay_id", 0: "outcome"})
        aux = aux[aux["stay_id"].isin(stay_ids)].copy()

        skf = StratifiedKFold(n_splits=n_splits, random_state=42)

        folds = pd.DataFrame()
        fold_ix = 0
        for train_index, test_index in skf.split(aux["stay_id"], aux["outcome"]):
            train_stays = list(aux.iloc[train_index]["stay_id"])
            test_stays = list(aux.iloc[test_index]["stay_id"])
            folds = pd.concat(
                [
                    folds,
                    pd.DataFrame(
                        [(stay, "train", fold_ix) for stay in train_stays],
                        columns=["stay_id", "train_test", "fold"],
                    ),
                ]
            )

            folds = pd.concat(
                [
                    folds,
                    pd.DataFrame(
                        [(stay, "test", fold_ix) for stay in test_stays],
                        columns=["stay_id", "train_test", "fold"],
                    ),
                ]
            )
            fold_ix += 1
        folds.to_csv(os.path.join(base_path, f"mimic_iv_{n_splits}fold.csv"), index=False)
    else:
        folds = pd.read_csv(os.path.join(base_path, f"mimic_iv_{n_splits}fold.csv"))
    return folds


def _get_train_test_split(data, outcome, folds, fold_ix):
    """
    Splits the dataset into training and testing sets for a specific fold.
    Parameters
    ----------
    data : dict
        Dictionary mapping stay_id to patient data arrays of shape (n_timesteps, n_features).
    outcome : dict
        Dictionary mapping stay_id to outcome labels.
    folds : pandas.DataFrame
        DataFrame containing fold assignments with columns 'fold', 'train_test', and 'stay_id'.
    fold_ix : int
        The fold index to use for splitting.
    Returns
    -------
    X_train : np.ndarray
        Training data array of shape (num_train_stays, 48, 16).
    X_test : np.ndarray
        Testing data array of shape (num_test_stays, 48, 16).
    y_train : np.ndarray
        Training outcome labels of shape (num_train_stays, 1).
    y_test : np.ndarray
        Testing outcome labels of shape (num_test_stays, 1).
    """

    train_stays = folds.loc[(folds["fold"] == fold_ix) & (folds["train_test"] == "train"), "stay_id"]
    test_stays = folds.loc[(folds["fold"] == fold_ix) & (folds["train_test"] == "test"), "stay_id"]

    # # allocate the total size
    X_train = np.ndarray((len(train_stays), 48, 16))
    y_train = np.ndarray((len(train_stays), 1))

    # start updating the arrays
    offset = 0
    for stay_id in tqdm(train_stays):
        if stay_id not in data:
            continue
        X_train[offset : offset + 1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_train[offset : offset + 1, :] = outcome[stay_id]
        offset += 1
    print("X Train shape: ", X_train.shape)
    print("Y Train shape: ", y_train.shape)

    X_test = np.ndarray((len(test_stays), 48, 16))
    y_test = np.ndarray((len(test_stays), 1))

    # start updating the arrays
    offset = 0
    for stay_id in tqdm(test_stays):
        if stay_id not in data:
            continue
        X_test[offset : offset + 1, :, :] = data[stay_id][-1, ::-1, 1:]
        y_test[offset : offset + 1, :] = outcome[stay_id]
        offset += 1
    print("X Test shape: ", X_test.shape)
    print("Y Test shape: ", y_test.shape)

    return X_train, X_test, y_train, y_test


def _normalize(base_path, X, load=False, save=True, filename=""):
    """
    Normalizes the input data `X` using MinMaxScaler, with options to load or save the scaler.
    Parameters
    ----------
    base_path : str
        The base directory path for saving/loading the scaler.
    X : np.ndarray
        The input data array of shape (batch_size, 48, 16) to be normalized.
    load : bool, optional
        If True, loads an existing scaler from file. If False, fits a new scaler. Default is False.
    save : bool, optional
        If True, saves the fitted scaler to file. Default is True.
    filename : str, optional
        The filename for saving/loading the scaler. If empty, defaults to 'mimic_iv_normalizer.pkl' in `base_path`.
    Returns
    -------
    np.ndarray
        The normalized data array with the same shape as the input `X`.
    """

    fn = filename if len(filename) > 0 else os.path.join(base_path, "mimic_iv_normalizer.pkl")
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


def _prepare_data(X_train, X_test, y_train, y_test):
    """
    Splits the test set into validation and test subsets, handles NaN values, and converts target arrays to one-hot encoded format.
    Parameters
    ----------
    X_train : np.ndarray
        Training feature data of shape (n_samples_train, ...).
    X_test : np.ndarray
        Test feature data of shape (n_samples_test, ...).
    y_train : np.ndarray
        Training target data of shape (n_samples_train, 1).
    y_test : np.ndarray
        Test target data of shape (n_samples_test, 1).
    Returns
    -------
    X_train : np.ndarray
        Training feature data with NaNs replaced by zeros.
    y_train_v2 : np.ndarray
        One-hot encoded training target data of shape (n_samples_train, 2).
    X_test : np.ndarray
        Test feature data (after validation split) with NaNs replaced by zeros.
    y_test_v2 : np.ndarray
        One-hot encoded test target data of shape (n_samples_test / 2, 2).
    X_val : np.ndarray
        Validation feature data with NaNs replaced by zeros.
    y_val : np.ndarray
        Validation target data.
    y_val_v2 : np.ndarray
        One-hot encoded validation target data of shape (n_samples_test / 2, 2).
    """

    mask = [False] * X_test.shape[0]
    val_idx = np.random.randint(0, X_test.shape[0], size=int(np.floor(X_test.shape[0] / 2)))
    for i in val_idx:
        mask[i] = True

    X_val = X_test[mask, :, :]
    y_val = y_test[mask]

    X_test = X_test[np.invert(mask), :, :]
    y_test = y_test[np.invert(mask)]
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    y_train_v2 = np.hstack([y_train, y_train])
    y_train_v2[:, 0] = np.where(y_train == 0, 1, 0)[:, 0]
    y_train_v2[:, 1] = np.where(y_train == 1, 1, 0)[:, 0]

    y_val_v2 = np.hstack([y_val, y_val])
    y_val_v2[:, 0] = np.where(y_val == 0, 1, 0)[:, 0]
    y_val_v2[:, 1] = np.where(y_val == 1, 1, 0)[:, 0]

    y_test_v2 = np.hstack([y_test, y_test])
    y_test_v2[:, 0] = np.where(y_test == 0, 1, 0)[:, 0]
    y_test_v2[:, 1] = np.where(y_test == 1, 1, 0)[:, 0]

    return X_train, y_train_v2, X_test, y_test_v2, X_val, y_val, y_val_v2


def _plot_roc(name, labels, predictions, **kwargs):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for given labels and predictions.
    Args:
        name (str): Name to display in the plot legend.
        labels (array-like): True binary labels (0 or 1).
        predictions (array-like): Predicted scores or probabilities.
        **kwargs: Additional keyword arguments passed to `plt.plot`.
            - 'save' (bool, optional): If True, saves the plot as a PNG file with a timestamped filename.
    Displays:
        ROC curve with false positive rate (x-axis) and true positive rate (y-axis), including the AUC in the legend.
        Optionally saves the plot if 'save' is set to True.
    """

    fp, tp, _ = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    plt.plot(
        100 * fp,
        100 * tp,
        label="{} (AUC={:.2f})".format(name, auc),
        linewidth=2,
        **kwargs,
    )
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")
    if kwargs.get("save", False):
        plt.savefig("{}.png".format(datetime.datetime.now().isoformat()))


def _get_weights(y_train):
    """
    Calculates class weights for a multi-label classification problem.
    Args:
        y_train (np.ndarray): A 2D numpy array where each column represents a class label (binary: 0 or 1).
    Returns:
        dict: A dictionary mapping class indices (0 and 1) to their computed weights,
              where the weight for each class is the ratio of the total number of samples
              to the number of positive samples for that class.
    Note:
        Assumes y_train has at least two columns, each corresponding to a class.
    """

    return {
        0: y_train.shape[0] / (y_train[:, 0].sum()),
        1: (y_train.shape[0] / (y_train[:, 1].sum())),
    }


def softmax_temperature(x, temperature=1.0):
    """
    Applies the softmax function with temperature scaling to the input tensor.
    Args:
        x: Input tensor.
        temperature (float, optional): Temperature parameter for scaling logits.
            Higher values produce softer probability distributions. Default is 1.0.
    Returns:
        Tensor: Softmax probabilities of the input tensor, scaled by the temperature.
    """

    exp_x = keras.backend.exp(x / temperature)
    return exp_x / keras.backend.sum(exp_x, axis=-1, keepdims=True)


class SoftmaxTemperature(layers.Layer):
    """
    A custom Keras layer that applies the softmax activation function with a configurable temperature.
    The temperature parameter controls the "sharpness" of the softmax output:
    - Higher temperature (>1.0) produces softer probability distributions.
    - Lower temperature (<1.0) produces more peaked distributions.
    Args:
        temperature (float): The temperature value to scale the logits before applying softmax. Default is 1.0.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    Methods:
        call(inputs): Applies temperature-scaled softmax to the input tensor.
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


def _create_lstm_model(input_shape):
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


def _fit_and_eval_model(model, model_name, X_train, y_train, X_test, y_test, epochs=1000, batch_size=100):
    """
    Fits a Keras model to training data and evaluates it on test data, using class weighting and early stopping.
    Parameters:
        model (keras.Model): The Keras model to train.
        model_name (str): Name of the model, used for logging.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test target data.
        epochs (int, optional): Number of training epochs. Defaults to 1000.
        batch_size (int, optional): Size of training batches. Defaults to 100.
    Returns:
        None
    """

    logger = keras.callbacks.TensorBoard(log_dir=f"./tensorflow_logs/{model_name}", write_graph=True)

    early_stop = EarlyStopping(monitor="val_recall", patience=100, mode="max")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train[:, 0])
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    class_weights_dict[0] = class_weights_dict[0] * 0.8
    class_weights_dict[1] = class_weights_dict[1] * 1.5

    _ = model.fit(
        X_train,
        y_train,
        class_weight=class_weights_dict,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        # validation_split=0.1,
        validation_data=(X_test, y_test),
        # shuffle=True,
        callbacks=[logger, early_stop],
    )


def _plot_results(model_lstm, X_train, y_train, X_test, y_test):
    """
    Evaluates and visualizes the performance of a given LSTM model on training and test datasets.
    This function:
    - Predicts outcomes on the test set and prints a classification report.
    - Computes baseline predictions for both training and test sets.
    - Plots ROC curves for training and test predictions.
    - Displays a normalized confusion matrix for test predictions.
    Args:
        model_lstm: Trained LSTM model with a `predict` method.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.
    Returns:
        None
    """

    y_pred = model_lstm.predict(X_test, batch_size=64, verbose=1)[:, 1]
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print(classification_report(y_test, y_pred))

    train_predictions_baseline = model_lstm.predict(X_train, batch_size=100)[:, 1]
    test_predictions_baseline = model_lstm.predict(X_test, batch_size=100)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_roc("Train Baseline", y_train, train_predictions_baseline)
    _plot_roc("Test Baseline", y_test, test_predictions_baseline, linestyle="--", save=True)
    plt.legend(loc="lower right")

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def train_mortality_model(base_path):
    """
    Trains an LSTM-based mortality prediction model using data from the specified base path.
    The function performs the following steps:
    1. Loads and preprocesses the data and outcome labels.
    2. Retrieves cross-validation folds.
    3. Iterates over each fold to:
        - Split data into training and testing sets.
        - Normalize the features for each fold.
        - Further prepares the data for model training and validation.
        - Creates and trains an LSTM model.
        - Evaluates the model on the test set.
        - Saves the trained model for each fold.
    Args:
        base_path (str): Path to the directory containing the data and configuration files.
    Returns:
        None
    """

    print("Get data")
    data, outcome = _get_data(base_path)
    print("Data obtained")
    folds = _get_folds(base_path, N_FOLDS)
    print(folds)
    print("Get folds")
    print("Folds obtained")
    for fold_ix in tqdm(range(N_FOLDS)):
        X_train, X_test, y_train, y_test = _get_train_test_split(data, outcome, folds, fold_ix)
        X_train = _normalize(
            base_path,
            X_train,
            load=False,
            save=True,
            filename=os.path.join(base_path, f"mimic_iv_normalizer_f{fold_ix}.pkl"),
        )
        X_test = _normalize(
            base_path,
            X_test,
            load=True,
            save=False,
            filename=os.path.join(base_path, f"mimic_iv_normalizer_f{fold_ix}.pkl"),
        )

        X_train, y_train_v2, X_test, y_test_v2, X_val, y_val, y_val_v2 = _prepare_data(X_train, X_test, y_train, y_test)

        model_lstm = _create_lstm_model(X_train.shape)
        _fit_and_eval_model(
            model_lstm,
            f"LSTM_{fold_ix}",
            X_train,
            y_train_v2,
            X_test,
            y_test_v2,
            epochs=1000,
            batch_size=100,
        )
        today = datetime.date.today().strftime("%Y%m%d")
        print("Saving model")
        model_lstm.save(f"./lstm_model_{fold_ix}_of_{N_FOLDS}_{today}.keras")


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

    tmp_x = []
    tmp_y = []
    for stay_id in tqdm(dataset["data"]):
        tmp_x.append(dataset["data"][stay_id][:, ::-1, 1:])
        tmp_y.append(np.ones((dataset["data"][stay_id].shape[0], 1)) * dataset["mortality_outcome"][stay_id])
    X = np.vstack(tmp_x)
    y = np.vstack(tmp_y)
    X = _normalize(
        base_path,
        X,
        load=True,
        save=False,
        filename=os.path.join(base_path, "mimic_iv_normalizer.pkl"),
    )

    X = np.nan_to_num(X)
    print(X.shape, y.shape)
    model_lstm = keras.models.load_model(mortality_model_filename)
    return {"y_true": y, "y_pred": model_lstm.predict(X)}
