import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from models import SkillDataset

def evaluate_neural(model, features, df, class_names, label_to_idx, device='cpu'):
    model.eval()
    dataset = SkillDataset(features, df["relationship_label"].astype(str), label_to_idx)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for f, _ in loader:
            f = f.to(device)
            outputs = model(f)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    predictions = [idx_to_label[p] for p in all_predictions]
    actual = df["relationship_label"].astype(str).to_numpy()

    return {
        "rows": int(df.shape[0]),
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


def evaluate(model, features, df, class_names):
    actual = df["relationship_label"].astype(str).to_numpy()
    predictions = model.predict(features)

    return {
        "rows": int(df.shape[0]),
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
