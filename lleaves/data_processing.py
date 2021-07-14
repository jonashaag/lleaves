import json
import os
from typing import List

import numpy as np

try:
    from pandas import DataFrame as pd_DataFrame
except ImportError:

    class pd_DataFrame:
        """Dummy class for pandas.DataFrame."""

        pass


def _dataframe_to_ndarray(data, pandas_categories: List[List]):
    """
    Converts the given dataframe into a 2D numpy array, without checking dimensions.
    Docstring ist nicht so hilfreich. Hier passiert viel mehr als dass ein df nach numpy
    konvertiert wird (sonst wäre es einfach "return df.values").
    :param data: 2D pandas dataframe.
    :param pandas_categories: list of lists. For each column a list of all categories in this column.
        The ordering of columns and of categories within each column should match the training dataset.
        Beispiel wäre hilfreich.
    :return: 2D np.ndarray, dtype float64 or float32
    """
    cat_cols = list(data.select_dtypes(include=["category"]).columns)
    if len(cat_cols) != len(pandas_categories):
        raise ValueError(
            "The categorical columns in the dataset don't match the categorical columns during training!"
            f"Train had {len(pandas_categories)} categorical columns, data has {len(cat_cols)}"
        )
    for col, category in zip(cat_cols, pandas_categories):
        # we use set_categories to get the same (category -> code) mapping that we used during train
        if list(data[col].cat.categories) != list(category):
            data[col] = data[col].cat.set_categories(category)
    if len(cat_cols):  # cat_cols is list
        data = data.copy()
        # apply (category -> code) mapping. Categories become floats
        data[cat_cols] = (
            data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        )
    data = data.values
    if data.dtype != np.float64 and data.dtype != np.float32:
        data = data.astype(np.float64)
    return data


def _list_to_ndarray(data):
    # Finde diese Funktion nicht sinnvoll, sie ist trivial und verschlechtert das
    # Error reporting (Grund warum es nicht konvertiert werden kann wird gelöscht)
    try:
        data = np.array(data, dtype=np.float64)
    except BaseException:
        raise ValueError("Cannot convert data list to appropriate np array")
    return data


def data_to_ndarray(data, pandas_categorical):
    """
    Hier fehlt wieder Info wie genau "data" aussieht.
    Es reicht, die Parameter in dieser Funktion zu dokumentieren (die Funktionen
    oben sind ja private)

    :param data: Pandas df, numpy 2D array or Python list.
    :param pandas_categorical: list of lists. For each categorical column in dataframe, a list of its categories.
    :return: numpy ndarray
    """
    if isinstance(data, np.ndarray):
        data = data
    elif isinstance(data, pd_DataFrame):
        data = _dataframe_to_ndarray(data, pandas_categorical)
    elif isinstance(data, list):
        data = _list_to_ndarray(data)
    else:
        raise ValueError(
            f"Expecting numpy.ndarray, pandas.DataFrame or Python list, got {type(data)}"
        )

    return data


def ndarray_to_1Darray(data):
    """
    Takes a 2D numpy array, flattens it and converts to float64 if necessary
    :param data: 2D numpy array.
    :return: (1D numpy array (dtype float64), number of rows in original data)
    """
    # könnte man nicht sowas machen wie data.astype(float64).ravel?
    # dieses "n_predictions" gehört nicht in diese Funktion.
    n_predictions = data.shape[0]
    if data.dtype == np.float64:
        data = np.array(data.reshape(data.size), dtype=np.float64, copy=False)
    else:
        data = np.array(data.reshape(data.size), dtype=np.float64)
    return data, n_predictions


def extract_pandas_traintime_categories(file_path):
    """
    Scan the model.txt from the back to extract the 'pandas_categorical' field.

    This is a list of lists that stores the ordering of categories from the pd.DataFrame used for training.
    Storing this list is necessary as LightGBM encodes categories as integer indices and we need to guarantee that
    the mapping (<category string> -> <integer idx>) is the same during inference as it was during training.

    Example (pandas categoricals were present in training):
    pandas_categorical:[["a", "b", "c"], ["b", "c", "d"], ["w", "x", "y", "z"]]
    Example (no pandas categoricals during training):
    pandas_categorical:[] OR pandas_categorical=null

    LightGBM generates this list of lists like so:
      pandas_categorical = [list(df[col].cat.categories) for col in df.select_dtypes(include=['category']).columns]
    and stores it via json.dump

    :param file_path: path to model.txt
    :return: list of list. For each pd.categorical column encountered during training, a list of the categories.
    """
    pandas_key = "pandas_categorical:"
    max_offset = os.path.getsize(file_path)
    stepsize = min(1024, max_offset - 1)
    current_offset = stepsize
    lines = []
    # seek backwards from end of file until we have two lines
    # the (pen)ultimate line should be pandas_categorical:XXX
    with open(file_path, "rb") as f:
        while len(lines) < 2 and current_offset < max_offset:
            if current_offset > max_offset:
                current_offset = max_offset
            # read <current_offset>-many Bytes from end of file
            f.seek(-current_offset, os.SEEK_END)
            lines = f.readlines()
            current_offset *= 2

    # pandas_categorical has to be present in the ultimate or penultimate line. Else the model.txt is malformed.
    if len(lines) >= 2:
        last_line = lines[-1].decode().strip()
        if not last_line.startswith(pandas_key):
            last_line = lines[-2].decode().strip()
        if last_line.startswith(pandas_key):
            return json.loads(last_line[len(pandas_key) :])
    raise ValueError("Ill formatted model file!")


def extract_num_feature(file_path):
    """
    Extract number of features expected by this model as 'max_feature_idx' + 1
    :param file_path: path to model.txt
    :return: the number of features expected by this model.
    """
    with open(file_path, "r") as f:
        line = f.readline()
        while line and not line.startswith("max_feature_idx"):
            line = f.readline()

        if line.startswith("max_feature_idx"):
            n_args = int(line.split("=")[1]) + 1
        else:
            raise ValueError("Ill formatted model file!")
    return n_args
