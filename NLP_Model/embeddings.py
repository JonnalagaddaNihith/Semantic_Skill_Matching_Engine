import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def embed_pairs(encoder, df, batch_size=64, cache_dir=None):
    texts = pd.concat(
        [df["text_a"], df["text_b"]], ignore_index=True
    ).astype(str)

    # Encode each unique text once, reducing redundant computation.
    unique_texts = texts.drop_duplicates().tolist()

    # Check cache
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_hash = hashlib.md5(str(unique_texts).encode()).hexdigest()[:16]
        cache_file = cache_dir / f"embeddings_{cache_hash}.npz"

    if cache_file and cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        vectors = np.load(cache_file)['vectors']
    else:
        vectors = encoder.encode(
            unique_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if cache_file:
            np.savez(cache_file, vectors=vectors)
            print(f"Cached embeddings to {cache_file}")

    vector_map = {
        text: vector for text, vector in zip(unique_texts, vectors)
    }

    a = np.vstack([vector_map[t] for t in df["text_a"]])
    b = np.vstack([vector_map[t] for t in df["text_b"]])

    # Calculate cosine similarity between pairs
    cos_sim = cosine_similarity(a, b).diagonal().reshape(-1, 1)

    # Preserve both text meanings and their pairwise relationship.
    dense_features = np.concatenate(
        [a, b, np.abs(a - b), a * b, cos_sim],
        axis=1,
    ).astype(np.float32)

    return dense_features
