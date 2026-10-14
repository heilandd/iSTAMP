from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def sample_tiles_for_annotation(
    qc_df: pd.DataFrame,
    n_samples: int = 200,
    stratify_by: str | None = None,
    random_state: int = 42,
):
    if stratify_by is None or stratify_by not in qc_df.columns:
        return qc_df.sample(n=min(n_samples, len(qc_df)), random_state=random_state).copy()

    parts = []
    groups = qc_df.groupby(stratify_by, group_keys=False)
    per_group = max(1, n_samples // max(1, qc_df[stratify_by].nunique()))

    for _, g in groups:
        parts.append(g.sample(n=min(per_group, len(g)), random_state=random_state))

    out = pd.concat(parts).drop_duplicates()
    if len(out) > n_samples:
        out = out.sample(n=n_samples, random_state=random_state)
    return out.copy()

def review_tiles_manually(
    sdata,
    review_df: pd.DataFrame,
    image_name: str = "raw_image",
    tile_size: int = 448,
    save_path: str | Path = "tile_labels.csv",
):
    save_path = Path(save_path)

    if save_path.exists():
        existing = pd.read_csv(save_path)
        done_ids = set(existing["point_id"].astype(str))
    else:
        existing = pd.DataFrame()
        done_ids = set()

    rows = []

    for row in review_df.itertuples(index=False):
        point_id = str(row.point_id)
        if point_id in done_ids:
            continue

        tile = extract_tile_array(
            sdata,
            image_name=image_name,
            center_x=int(row.x),
            center_y=int(row.y),
            tile_size=tile_size,
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(tile)
        plt.title(f"{point_id}\n[d]=diagnostic, [n]=non-diagnostic, [s]=skip, [q]=quit")
        plt.axis("off")
        plt.show()

        label = input("Label: ").strip().lower()

        if label == "q":
            break
        elif label == "d":
            y = 1
        elif label == "n":
            y = 0
        else:
            y = None

        rows.append(
            {
                "point_id": point_id,
                "x": int(row.x),
                "y": int(row.y),
                "diagnostic_label": y,
            }
        )

        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(save_path, index=False)

    return pd.read_csv(save_path) if save_path.exists() else pd.DataFrame(rows)


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

class TileDiagnosticDataset(Dataset):
    def __init__(self, sdata, labels_df, image_name="raw_image", tile_size=448, transform=None):
        self.sdata = sdata
        self.labels_df = labels_df.dropna(subset=["diagnostic_label"]).reset_index(drop=True)
        self.image_name = image_name
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        tile = extract_tile_array(
            self.sdata,
            image_name=self.image_name,
            center_x=int(row["x"]),
            center_y=int(row["y"]),
            tile_size=self.tile_size,
        )

        tile = Image.fromarray(tile.astype(np.uint8))
        if self.transform is not None:
            tile = self.transform(tile)

        label = torch.tensor(int(row["diagnostic_label"]), dtype=torch.long)
        return tile, label
def make_diagnostic_model():
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
def train_diagnostic_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss/len(train_loader):.4f}, "
            f"val_loss={val_loss/len(val_loader):.4f}, "
            f"val_acc={correct/total:.3f}"
        )

    return model
def evaluate_diagnostic_model(model, val_loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc="Evaluating", unit="batch"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_true.extend(y.numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    print(classification_report(all_true, all_pred, digits=3))

    try:
        auc = roc_auc_score(all_true, all_prob)
        print("Validation AUC:", round(auc, 3))
    except Exception:
        pass

from pathlib import Path
import torch


def save_diagnostic_model(
    model,
    model_path: str | Path,
    architecture: str = "resnet18",
    image_name: str = "raw_image",
    tile_size: int = 448,
    input_size: int = 224,
):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "architecture": architecture,
        "tile_size": tile_size,
        "input_size": input_size,
        "image_name": image_name,
        "state_dict": model.state_dict(),
    }

    torch.save(payload, model_path)
    print(f"Saved diagnostic model to: {model_path}")   
      
