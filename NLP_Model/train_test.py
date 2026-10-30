import argparse
import json
import numpy as np
from pathlib import Path
import joblib
import sklearn
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from data import load_data, split_data, SEED
from embeddings import embed_pairs
from features import make_features
from models import SkillDataset, train_neural_network

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="../Synthetic_DataSet/DataSet/semantic_pairs.csv",
    )
    parser.add_argument("--output", default="../artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(ENCODER_NAME, device=device)
    train_dense = embed_pairs(encoder, train, args.batch_size, args.cache_dir)
    val_dense = embed_pairs(encoder, val, args.batch_size, args.cache_dir)
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

    # Train ensemble model (RF + Neural Network)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_to_idx = {label: idx for idx, label in enumerate(sorted(train["relationship_label"].unique()))}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    )
    rf_classifier.fit(train_features, train["relationship_label"].astype(str))

    # Train Neural Network
    nn_classifier = train_neural_network(
        train_features, train["relationship_label"].astype(str),
        val_features, val["relationship_label"].astype(str),
        label_to_idx, num_classes, device=device
    )

    classes = list(label_to_idx.keys())

    # Ensemble predictions (soft voting)
    def ensemble_predict(features):
        rf_probs = rf_classifier.predict_proba(features)
        nn_dataset = SkillDataset(features, None, label_to_idx)
        nn_loader = DataLoader(nn_dataset, batch_size=256, shuffle=False)
        nn_probs_list = []
        nn_classifier.eval()
        with torch.no_grad():
            for f, _ in nn_loader:
                f = f.to(device)
                outputs = nn_classifier(f)
                probs = torch.softmax(outputs, dim=1)
                nn_probs_list.append(probs.cpu().numpy())
        nn_probs = np.vstack(nn_probs_list)

        # Weighted average (RF gets 0.6, NN gets 0.4)
        combined_probs = 0.6 * rf_probs + 0.4 * nn_probs
        predictions = [classes[np.argmax(p)] for p in combined_probs]
        return predictions

    val_predictions = ensemble_predict(val_features)
    test_predictions = ensemble_predict(test_features)

    val_actual = val["relationship_label"].astype(str).to_numpy()
    test_actual = test["relationship_label"].astype(str).to_numpy()

    validation_metrics = {
        "rows": int(val.shape[0]),
        "accuracy": float(accuracy_score(val_actual, val_predictions)),
        "macro_f1": float(f1_score(val_actual, val_predictions, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(val_actual, val_predictions, labels=classes, average="weighted", zero_division=0)),
        "classification_report": classification_report(val_actual, val_predictions, labels=classes, output_dict=True, zero_division=0),
    }

    test_metrics = {
        "rows": int(test.shape[0]),
        "accuracy": float(accuracy_score(test_actual, test_predictions)),
        "macro_f1": float(f1_score(test_actual, test_predictions, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_actual, test_predictions, labels=classes, average="weighted", zero_division=0)),
        "classification_report": classification_report(test_actual, test_predictions, labels=classes, output_dict=True, zero_division=0),
    }

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
        "rf_classifier": rf_classifier,
        "nn_classifier": nn_classifier.cpu().state_dict(),
        "label_to_idx": label_to_idx,
        "task_vectorizer": task_vectorizer,
        "domain_vectorizer": domain_vectorizer,
        "encoder_name": ENCODER_NAME,
        "classes": classes,
        "model_version": "0.5.0",
        "feature_format": (
            "embedding_a, embedding_b, absolute_difference, "
            "elementwise_product, cosine_similarity, task_type, "
            "domain_a, domain_b, same_domain"
        ),
    }

    model_path = output_dir / "semantic_matching_model.joblib"
    joblib.dump(bundle, model_path)

    metrics = {
        "model": "Sentence Transformer + Ensemble (RF + Neural Network)",
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
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
