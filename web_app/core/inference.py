import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from django.conf import settings


class SkillDataset(Dataset):
    def __init__(self, features, labels, label_to_idx):
        if hasattr(features, 'toarray'):
            self.features = torch.FloatTensor(features.toarray())
        else:
            self.features = torch.FloatTensor(features)
        if labels is not None and len(labels) > 0:
            self.labels = torch.LongTensor([label_to_idx[l] for l in labels])
        else:
            self.labels = None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], 0


class NeuralClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ModelInference:
    def __init__(self, model_path=None, encoder_name='sentence-transformers/all-MiniLM-L6-v2'):
        if model_path is None:
            model_path = settings.MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from {model_path}")
        self.model_bundle = joblib.load(model_path)
        self.encoder = SentenceTransformer(encoder_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self._load_neural_network()
        print("Model loaded successfully")

    def _load_neural_network(self):
        nn_state_dict = self.model_bundle['nn_classifier']
        self.label_to_idx = self.model_bundle['label_to_idx']
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.classes = self.model_bundle['classes']
        self.task_vectorizer = self.model_bundle['task_vectorizer']
        self.domain_vectorizer = self.model_bundle['domain_vectorizer']
        self.rf_classifier = self.model_bundle['rf_classifier']

        num_classes = len(self.classes)
        input_dim = self._get_input_dim()

        self.nn_classifier = NeuralClassifier(input_dim, num_classes)
        self.nn_classifier.load_state_dict(nn_state_dict)
        self.nn_classifier.to(self.device)
        self.nn_classifier.eval()

    def _get_input_dim(self):
        return self.rf_classifier.n_features_in_

    def _embed_pair(self, text_a, text_b):
        texts = [text_a, text_b]
        vectors = self.encoder.encode(
            texts,
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        a = vectors[0].reshape(1, -1)
        b = vectors[1].reshape(1, -1)

        cos_sim = cosine_similarity(a, b).diagonal().reshape(-1, 1)

        dense_features = np.concatenate(
            [a, b, np.abs(a - b), a * b, cos_sim],
            axis=1,
        ).astype(np.float32)

        return dense_features

    def _make_features(self, dense, task_type='skill_matching'):
        task_records = [{"task_type": str(task_type)}]
        domain_a_records = [{"domain": "unknown"}]
        domain_b_records = [{"domain": "unknown"}]

        task_features = self.task_vectorizer.transform(task_records)
        domain_a_features = self.domain_vectorizer.transform(domain_a_records)
        domain_b_features = self.domain_vectorizer.transform(domain_b_records)

        same_domain_matrix = csr_matrix(np.array([[0]]))

        features = hstack(
            [csr_matrix(dense), task_features, domain_a_features, domain_b_features, same_domain_matrix],
            format="csr",
        )

        return features

    def predict(self, text_a, text_b, task_type='skill_matching'):
        dense = self._embed_pair(text_a, text_b)
        features = self._make_features(dense, task_type)

        rf_probs = self.rf_classifier.predict_proba(features)

        nn_dataset = SkillDataset(features, None, self.label_to_idx)
        nn_loader = DataLoader(nn_dataset, batch_size=1, shuffle=False)
        nn_probs_list = []
        with torch.no_grad():
            for f, _ in nn_loader:
                f = f.to(self.device)
                outputs = self.nn_classifier(f)
                probs = torch.softmax(outputs, dim=1)
                nn_probs_list.append(probs.cpu().numpy())
        nn_probs = np.vstack(nn_probs_list)

        combined_probs = 0.6 * rf_probs + 0.4 * nn_probs
        prediction_idx = np.argmax(combined_probs[0])
        prediction = self.classes[prediction_idx]

        probabilities = {
            self.classes[i]: float(combined_probs[0][i])
            for i in range(len(self.classes))
        }

        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': float(combined_probs[0][prediction_idx])
        }


_inference_instance = None


def get_inference():
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = ModelInference()
    return _inference_instance
