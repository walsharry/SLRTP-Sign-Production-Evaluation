import json, pickle

from pathlib import Path


def load_json(file_path):
    """
    Load a JSON file from the given file path.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(path, 'r') as file:
        data = json.load(file)

    return data


def load_pickle(file_path):
    """
    Load a pickle file from the given file path.

    :param file_path: Path to the pickle file.
    :return: Loaded pickle data.
    """
    import pickle

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data