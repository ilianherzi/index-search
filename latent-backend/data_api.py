import json
import hashlib
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Any
from typing_extensions import TypeAlias
import logging

logging.basicConfig(
    level=logging.DEBUG,
)

TextSource: TypeAlias = str
JSON: TypeAlias = "dict[Any, Any]"


@dataclass(frozen=True)
class PassageIndex:
    # the name of the original text file like "Harry Potter book 1"
    text_src: TextSource
    # The passage should be a single sentence
    passage: str
    # The context should be multiple sentences: NOTE this is memory inefficent.
    contextual_passage: str
    # The starting sentence of the passage
    sentence_index_start: int
    # The ending sentence of the passage
    sentence_index_end: int
    # An optional name reference to how the passage was split up. If the rules
    # change for how a name was split then this will break.
    sentence_splitter_name: Optional[str] = None

    @classmethod
    def from_json(cls, json_dict: JSON) -> "PassageIndex":
        return PassageIndex(
            **json_dict,
        )


# Database APIs
def array_to_index(arr: np.ndarray) -> bytes:
    hashed_key = hashlib.sha256(arr.tobytes()).hexdigest()
    return hashed_key


def load_database(
    database_path: str,
) -> "tuple[Optional[np.ndarray], dict[bytes, str]]":
    if not os.path.exists(database_path):
        os.mkdir(database_path)
        logging.info(f"Database does not exist at {database_path}, creating empty one.")
        return None, {}

    with open(os.path.join(database_path, "database.json"), "r") as f:
        database = json.load(f)
    keys = np.load(os.path.join(database_path, "keys.npy"))
    return keys, database


def save_database(
    keys: np.ndarray,
    database: "dict[tuple, str]",
    database_path: str,
) -> None:
    np.save(os.path.join(database_path, "keys.npy"), keys)
    with open(os.path.join(database_path, "database.json"), "w") as f:
        json.dump(database, f)
