import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

SEED = 42

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

    # Merge PREREQUISITE into RELATED for better classification
    df["relationship_label"] = df["relationship_label"].replace("PREREQUISITE", "RELATED")

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
