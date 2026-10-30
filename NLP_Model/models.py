import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        return self.features[idx], 0  # Dummy label for prediction

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

def train_neural_network(train_features, train_labels, val_features, val_labels,
                         label_to_idx, num_classes, device='cpu', epochs=50, batch_size=128):
    train_dataset = SkillDataset(train_features, train_labels, label_to_idx)
    val_dataset = SkillDataset(val_features, val_labels, label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    model = NeuralClassifier(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            break

    model.load_state_dict(best_state)
    return model
