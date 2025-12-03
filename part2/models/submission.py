from __future__ import annotations
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from part2.dataset import ImageDataset



def mil_collate_fn(batch: List[Tuple[torch.Tensor, int]]):
    """
    Each batch item:
        embeddings: (num_patches_i, embed_dim)
        label: int
    We return:
        bags: list[Tensor], each (num_patches, embed_dim)
        labels: Tensor(batch_size)
    """
    bags, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(bags), labels



class AttentionMIL(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # Patch encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Attention: a_i = softmax(v^T tanh(W h_i))
        self.att_V = nn.Linear(hidden_dim, hidden_dim)
        self.att_U = nn.Linear(hidden_dim, 1)

        # Bag classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    
    def forward(self, bags: List[torch.Tensor]):
        device = next(self.parameters()).device
        bag_representations = []

        for bag in bags:
            bag = bag.to(device)

            H = self.encoder(bag)                        
            A = self.att_U(torch.tanh(self.att_V(H)))    
            A = torch.softmax(A.squeeze(1), dim=0)      

            bag_repr = torch.sum(A.unsqueeze(1) * H, dim=0)
            bag_representations.append(bag_repr)

        bag_representations = torch.stack(bag_representations)

        # Classifier
        logits = self.classifier(bag_representations)

        # return probabilities for ROC-AUC
        probs = torch.softmax(logits, dim=1)
        return probs







class Submission(AttentionMIL):
    """
    Assignment requires:
    - class Submission
    - method load_weights(cls, path)
    """

    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.25):
        super().__init__(embed_dim, hidden_dim, num_classes, dropout)

    def save_weights(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout
        }, path)
        print(f"Saved MIL model to {path}")

    @classmethod
    def load_weights(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device)
        model = cls(
            ckpt["embed_dim"],
            ckpt["hidden_dim"],
            ckpt["num_classes"],
            ckpt.get("dropout", 0.25),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"Loaded MIL model from {path}")
        return model



def train_mil(
    dataset_path="/shared/CS461/cs461_assignment2_data/part2/data",
    split="train",
    hidden_dim=256,
    dropout=0.25,
    batch_size=8,
    lr=1e-4,
    weight_decay=1e-5,
    num_epochs=20,
    val_fraction=0.2,
    ckpt_path="ckpts/best_mil.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageDataset(dataset_path, split)

    example_emb, _ = dataset[0]
    embed_dim = example_emb.shape[1]
    num_classes = int(dataset.labels.max() + 1)

    # imbalance
    counts = np.bincount(dataset.labels)
    class_weights = torch.tensor(1 / np.maximum(counts, 1), dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    n_total = len(dataset)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn)

    model = Submission(embed_dim, hidden_dim, num_classes, dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0.0

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0

        for bags, labels in train_loader:
            labels = labels.to(device)
            optim.zero_grad()
            logits = model(bags)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # val
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for bags, labels in val_loader:
                labels = labels.to(device)
                logits = model(bags)
                pred = logits.argmax(1)
                preds += pred.cpu().tolist()
                gts   += labels.cpu().tolist()

        f1 = f1_score(gts, preds, average="macro")
        print(f"Epoch {epoch}/{num_epochs} | train_loss={avg_loss:.4f} | val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_weights(ckpt_path)
            print(f" → New best model saved (F1={best_f1:.4f})")

    print("Training complete.")
    print(f"Best macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    # Training entry point
    train_mil(
        dataset_path="/shared/CS461/cs461_assignment2_data/part2/data",
        split="train",
        hidden_dim=256,
        dropout=0.25,
        batch_size=8,
        lr=1e-4,
        weight_decay=1e-5,
        num_epochs=20,
        val_fraction=0.2,
        ckpt_path="ckpts/best_mil.pt",
    )
