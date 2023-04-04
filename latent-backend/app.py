from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

import numpy as np
import os
from typing import Callable
from tqdm import tqdm
import re
from dataclasses import asdict
from .embedding import (
    chat_gpt_embedding,
    sentence_transformer_embedding,
)
from .data_api import load_database, save_database, array_to_index
from .similarity import cosine_similarity
from .data_api import PassageIndex

# DONT do this (for debugging)
# exceeded monnies quota.
os.environ["OPENAI_API_KEY"] = ""

logging.basicConfig(
    level=logging.DEBUG,
)

app = Flask(__name__)
CORS(app)

DATABASE = "./database/"
DIMENSION = 384
EPS = 1e-5


@app.after_request
def add_cors_headers(response):
    # Allow requests from any origin
    response.headers["Access-Control-Allow-Origin"] = "*"
    # Allow the following HTTP methods
    response.headers["Access-Control-Allow-Methods"] = "POST"
    # Allow the following headers
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _split(text: str) -> "list[str]":
    pattern = r"[.\n?!]"
    sentences = [s for s in re.split(pattern, text) if s]
    return sentences


def sentence_level_embedding(
    text_src: str,
    text: str,
    dimension_size: int,
    embedder: "Callable[[str, int], np.ndarray]",
    database: "dict[Any, Any]",
    num_sentences_per_passage: int = 5,
) -> "tuple[list[hash_keys], list[keys], list[PassageIndex]":
    all_sentences = _split(text)
    keys = []
    map_passages = {}

    def _sentences_to_passage(sentences: "list[str]") -> "tuple[bytes, np.ndarray]":
        key = embedder(
            ". ".join(sentences),
            dimension_size=dimension_size,
        )
        hash_key = array_to_index(key)
        if hash_key in database:
            logging.info("Found key in database, assuming text has been indexed.")
            return None, {}

        return hash_key, key

    if num_sentences_per_passage > len(all_sentences):
        hash_key, key = _sentences_to_passage(all_sentences)
        passage_index = PassageIndex(
            text_src,
            passage=". ".join(all_sentences),
            contextual_passage=all_sentences,
            sentence_index_start=0,
            sentence_index_end=len(all_sentences),
            sentence_splitter_name="_split",
        )
        return [key], {hash_key: asdict(passage_index)}

    for index in tqdm(
        range(
            num_sentences_per_passage // 2,
            len(all_sentences) - num_sentences_per_passage // 2,
            num_sentences_per_passage,
        ),
        total=len(all_sentences) // num_sentences_per_passage,
    ):
        sentences = all_sentences[
            index
            - num_sentences_per_passage // 2 : index
            + num_sentences_per_passage // 2
        ]
        hash_key, key = _sentences_to_passage(sentences)
        keys.append(key)
        map_passages[hash_key] = asdict(
            PassageIndex(
                text_src,
                passage=all_sentences[index],
                contextual_passage=sentences,
                sentence_index_start=index,
                sentence_index_end=index + num_sentences_per_passage,
                sentence_splitter_name="_split",
            )
        )
    return keys, map_passages


def _index_all_data(
    request_data: str,
    database_path: str,
    embedder: "Callable[[str], np.ndarray]",
) -> None:
    passage_key = embedder(request_data)  # (D, )
    assert (
        passage_key.shape[0] == DIMENSION
    ), "Embedding dimension not configured properly."
    keys, database = load_database(database_path=database_path)
    hashed_key = array_to_index(passage_key)
    if hashed_key in database:
        logging.info("Passage already exists, skipping")
        return
    keys = (
        np.concatenate(
            [
                keys,
                passage_key[None, ...],
            ],
            axis=0,
        )
        if keys is not None
        else passage_key[None, ...]
    )
    database[hashed_key] = request_data
    save_database(
        keys,
        database,
        database_path,
    )


def _index_data(
    data_source: str,
    request_data: str,
    database_path: str,
    dimension_size: int,
    embedder: "Callable[[str], np.ndarray]",
    num_sentences_per_passage: int = 5,
) -> None:
    keys, database = load_database(database_path=database_path)
    passage_keys, map_passages = sentence_level_embedding(
        text_src=data_source,
        text=request_data,
        dimension_size=dimension_size,
        embedder=embedder,
        num_sentences_per_passage=num_sentences_per_passage,
        database=database,
    )
    if passage_keys is None:  # already indexed
        return

    keys = (
        np.concatenate(
            [keys, np.stack(passage_keys)],
            axis=0,
        )
        if keys is not None
        else np.stack(passage_keys)
    )
    database.update(map_passages)
    save_database(
        keys,
        database,
        database_path,
    )


def _search_data(
    request_data: str,
    database_path: str,
    embedder: "Callable[[str], np.ndarray]",
    k: int = 1,
) -> "list[str]":
    query_embedding = embedder(request_data)  # (D, )
    assert (
        query_embedding.shape[0] == DIMENSION
    ), "Embedding dimension not configured properly."
    # At this point keys is a (P, D) where P is the number of passages
    # Database is a vector hash mapping from the vector to text.
    keys, database = load_database(database_path=database_path)
    passages = []
    most_similar_keys = cosine_similarity(keys, query_embedding, k=k)
    for key in most_similar_keys:
        most_similar_passage = database[array_to_index(key)]
        passages.append(PassageIndex.from_json(most_similar_passage))
    return [p for p in passages if len(p.passage) > 5]  # NOTE: hotfix for bad indexing


@app.route("/", methods=["GET"])
def home() -> str:
    return "Backend server connected. Check API for possible routes."


@app.route("/index", methods=["POST"])
def index() -> str:
    request_data = request.get_json()  # parse the JSON data
    _index_data(
        data_source=request_data["file"],
        request_data=request_data["payload"],
        database_path=DATABASE,
        embedder=sentence_transformer_embedding,
        dimension_size=DIMENSION,
        num_sentences_per_passage=5,
    )
    return "Data indexed successfully."


@app.route("/query", methods=["POST"])
def query() -> "dict[str, Any]":
    request_data = request.get_json()
    text_query = request_data["query"]
    passages: "list[Passages]" = _search_data(
        text_query,
        DATABASE,
        sentence_transformer_embedding,
        k=500,
    )
    cleaned_passages = passages[-10:]  # NOTE: hotfix for bad indexing
    return {
        "passages": [re.sub(r"\W+", " ", p.passage).lower() for p in cleaned_passages],
        "contextual_passage": [
            re.sub(r"\W+", " ", ". ".join(p.contextual_passage)).lower()
            for p in cleaned_passages
        ],
        "num_passages": len(cleaned_passages),
    }
