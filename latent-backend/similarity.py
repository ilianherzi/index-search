import numpy as np

EPS = 1e-5


def cosine_similarity(
    keys: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 1,
) -> "list[np.ndarray]":
    """Cosine similarity on a database of keys.

    Args:
        keys: (M, D).
        query_embedding: (D).
        k: number of keys to return

    Returns:
        np.ndarray: One of the M keys.
    """
    # (M, D) -> (M, 1)
    similarities = (keys / (np.linalg.norm(keys, axis=1) + EPS)[..., None]) @ (
        query_embedding / (np.linalg.norm(query_embedding, axis=0) + EPS)
    )
    topk_indicies = np.argpartition(similarities, -k)[-k:]
    selected_keys = keys[topk_indicies]
    return selected_keys
