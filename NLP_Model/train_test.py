import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity


SEED = 42
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_data(path):
    df = pd.read_csv(path)

    required = [
        "task_type",
        "text_a",
        "text_b",
        "relationship_label",
        "generation_group_id",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=required).copy()

    for column in required:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["text_a"] != "")
        & (df["text_b"] != "")
        & (df["relationship_label"] != "")
        & (df["generation_group_id"] != "")
    ].reset_index(drop=True)

    if df["relationship_label"].nunique() < 2:
        raise ValueError("At least two relationship classes are required.")

    # Load competencies for domain information
    competencies_path = Path(path).parent / "competencies.csv"
    if competencies_path.exists():
        competencies_df = pd.read_csv(competencies_path)
        competency_to_domain = dict(zip(
            competencies_df["competency_id"],
            competencies_df["domain"]
        ))
        df["domain_a"] = df["entity_a_id"].map(competency_to_domain).fillna("unknown")
        df["domain_b"] = df["entity_b_id"].map(competency_to_domain).fillna("unknown")
        df["same_domain"] = (df["domain_a"] == df["domain_b"]).astype(int)
    else:
        df["domain_a"] = "unknown"
        df["domain_b"] = "unknown"
        df["same_domain"] = 0

    return df


def split_data(df):
    groups = df["generation_group_id"].astype(str)

    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )
    trainval_idx, test_idx = next(
        split1.split(df, groups=groups)
    )

    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1765,
        random_state=SEED + 1,
    )
    train_idx, val_idx = next(
        split2.split(
            trainval,
            groups=trainval["generation_group_id"].astype(str),
        )
    )

    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    group_sets = [
        set(part["generation_group_id"])
        for part in (train, val, test)
    ]

    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise ValueError("Group leakage detected between data splits.")

    return train, val, test


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


def make_features(dense, tasks, task_vectorizer, domains_a, domains_b, same_domain, fit=False):
    task_records = [{"task_type": str(t)} for t in tasks]
    domain_a_records = [{"domain": str(d)} for d in domains_a]
    domain_b_records = [{"domain": str(d)} for d in domains_b]

    if fit:
        task_features = task_vectorizer.fit_transform(task_records)
        domain_vectorizer = DictVectorizer(sparse=True)
        domain_a_features = domain_vectorizer.fit_transform(domain_a_records)
        domain_b_features = domain_vectorizer.transform(domain_b_records)
    else:
        task_features = task_vectorizer.transform(task_records)
        domain_vectorizer = task_vectorizer.domain_vectorizer
        domain_a_features = domain_vectorizer.transform(domain_a_records)
        domain_b_features = domain_vectorizer.transform(domain_b_records)

    # Stack all features
    same_domain_matrix = csr_matrix(same_domain.values.reshape(-1, 1))
    return hstack(
        [csr_matrix(dense), task_features, domain_a_features, domain_b_features, same_domain_matrix],
        format="csr",
    ), domain_vectorizer


def evaluate(model, features, df, class_names):
    actual = df["relationship_label"].astype(str).to_numpy()
    predictions = model.predict(features)

    return {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(
            f1_score(
                actual, predictions,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual, predictions,
                labels=class_names,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            actual,
            predictions,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="../Synthetic_DataSet/DataSet/semantic_pairs.csv",
    )
    parser.add_argument("--output", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--classifier",
        choices=["logreg", "svm", "rf"],
        default="logreg",
        help="Classifier type: logreg, svm, or rf"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with grid search"
    )
    parser.add_argument(
        "--cache-dir",
        default="embeddings_cache",
        help="Directory to cache embeddings"
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    df = load_data(data_path)
    train, val, test = split_data(df)

    print(f"Total rows: {len(df)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")
    print("\nTraining labels:")
    print(train["relationship_label"].value_counts().to_string())

    print(f"\nLoading encoder: {ENCODER_NAME}")
    print("The first run may download the pretrained model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(ENCODER_NAME, device=device)

    print("\nEncoding training pairs...")
    train_dense = embed_pairs(encoder, train, args.batch_size, args.cache_dir)

    print("\nEncoding validation pairs...")
    val_dense = embed_pairs(encoder, val, args.batch_size, args.cache_dir)

    print("\nEncoding test pairs...")
    test_dense = embed_pairs(encoder, test, args.batch_size, args.cache_dir)

    task_vectorizer = DictVectorizer(sparse=True)

    train_features, domain_vectorizer = make_features(
        train_dense, train["task_type"], task_vectorizer,
        train["domain_a"], train["domain_b"], train["same_domain"],
        fit=True
    )
    task_vectorizer.domain_vectorizer = domain_vectorizer
    val_features, _ = make_features(
        val_dense, val["task_type"], task_vectorizer,
        val["domain_a"], val["domain_b"], val["same_domain"]
    )
    test_features, _ = make_features(
        test_dense, test["task_type"], task_vectorizer,
        test["domain_a"], test["domain_b"], test["same_domain"]
    )

    # Select classifier based on argument
    if args.classifier == "logreg":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'max_iter': [2000],
            'class_weight': ['balanced'],
            'random_state': [SEED]
        }
    elif args.classifier == "svm":
        classifier = SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=SEED,
            probability=False
        )
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf'],
            'class_weight': ['balanced'],
            'random_state': [SEED]
        }
    else:  # rf
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced'],
            'random_state': [SEED],
            'n_jobs': [-1]
        }

    print(f"\nTraining {args.classifier} classifier...")

    if args.tune:
        print("Performing hyperparameter tuning with grid search...")
        grid_search = GridSearchCV(
            classifier,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(train_features, train["relationship_label"].astype(str))
        classifier = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    else:
        classifier.fit(
            train_features,
            train["relationship_label"].astype(str),
        )

    classes = list(classifier.classes_)

    validation_metrics = evaluate(
        classifier, val_features, val, classes
    )
    test_metrics = evaluate(
        classifier, test_features, test, classes
    )

    print("\nValidation metrics:")
    print(json.dumps({
        k: v for k, v in validation_metrics.items()
        if k != "classification_report"
    }, indent=2))

    print("\nTest metrics:")
    print(json.dumps({
        k: v for k, v in test_metrics.items()
        if k != "classification_report"
    }, indent=2))

    bundle = {
        "classifier": classifier,
        "classifier_type": args.classifier,
        "task_vectorizer": task_vectorizer,
        "domain_vectorizer": domain_vectorizer,
        "encoder_name": ENCODER_NAME,
        "classes": classes,
        "model_version": "0.3.0",
        "feature_format": (
            "embedding_a, embedding_b, absolute_difference, "
            "elementwise_product, cosine_similarity, task_type, "
            "domain_a, domain_b, same_domain"
        ),
    }

    model_path = output_dir / "semantic_matching_model.joblib"
    joblib.dump(bundle, model_path)

    metrics = {
        "model": "Sentence Transformer + Logistic Regression",
        "encoder": ENCODER_NAME,
        "device": device,
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
        "split_strategy": "GroupShuffleSplit",
        "split_sizes": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "note": (
            "Metrics measure agreement with synthetic heuristic labels. "
            "They do not establish real-world matching accuracy."
        ),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {model_path.resolve()}")
    print(f"Metrics saved to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
