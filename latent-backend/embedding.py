from sentence_transformers import SentenceTransformer
import numpy as np
import openai


def _pad(
    vector: np.ndarray,
    dim_size: int,
    value=0.0,
) -> np.ndarray:
    return np.pad(
        vector,
        (0, dim_size - vector.shape[-1]),
        "constant",
        constant_values=value,
    )


def sentence_transformer_embedding(
    text: str,
    dimension_size: int = 384,
) -> np.ndarray:
    """Map a text passage to a tensor vector

    Args:
        text: A N long sentence
        dimensions: Size of the vector dimension. If it's greater than the
            model output the result will be zero-padded.

    Returns:
        np.ndarray: (D)

    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(text)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, ...]
    return _pad(embeddings.mean(axis=0), dimension_size)


def chat_gpt_embedding(text: str, dimension_size: int = 1536) -> np.ndarray:
    # platform.openai.com/docs/guides/embeddings/what-are-embeddings
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
