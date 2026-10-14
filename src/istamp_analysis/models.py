

# Gradient reversal for domain adaptation

import torch
import torch.nn as nn
from torch.autograd import Function
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# Domain-adversarial head
class DomainAdversarialHead(nn.Module):
    def __init__(self, in_dim: int, n_domains: int, hidden_dim: int = 128, grl_alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x):
        x = self.grl(x)
        return self.net(x)


## Attention-based GNN backbone

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
class RamanGATv2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 5,
        edge_dim: int = 1,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        n_domains: int | None = None,
        domain_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.n_domains = n_domains

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = hidden_channels if layer_idx == 0 else hidden_channels
            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
                share_weights=False,
            )
            self.convs.append(conv)

            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.norms.append(nn.Identity())

        # graph embedding combines mean + max pooling
        graph_dim = hidden_channels * 2

        self.pred_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        if n_domains is not None:
            self.domain_head = DomainAdversarialHead(
                in_dim=graph_dim,
                n_domains=n_domains,
                hidden_dim=domain_hidden_dim,
                grl_alpha=grl_alpha,
            )
        else:
            self.domain_head = None

    def forward(self, data, return_attention: bool = False):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        batch = data.batch

        x = self.input_proj(x)

        attention_outputs = []

        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                x_out, (edge_idx_att, alpha) = conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention_outputs.append((edge_idx_att, alpha))
            else:
                x_out = conv(x, edge_index, edge_attr=edge_attr)

            x = norm(x_out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_embedding = torch.cat([graph_mean, graph_max], dim=1)

        pred = self.pred_head(graph_embedding)

        out = {
            "pred": pred,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
        }

        if return_attention:
            out["attention_weights"] = attention_outputs

        if self.domain_head is not None:
            out["domain_logits"] = self.domain_head(graph_embedding)

        return out


import numpy as np

def transform_edge_distance(distance: float, sigma: float = 300.0):
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


## Training
from tqdm import tqdm


import copy
import torch
import torch.nn as nn
from tqdm import tqdm


def train_raman_gnn(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int = 100,
    patience: int = 15,
    lambda_domain: float = 0.1,
    use_domain_adaptation: bool = False,
    return_history: bool = True,
):
    pred_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader, desc=f"Train epoch {epoch+1}", unit="batch"):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data, return_attention=False)
            pred = out["pred"]

            # make sure target shape matches prediction shape
            y = data.y.float().view_as(pred)

            loss_pred = pred_criterion(pred, y)
            loss = loss_pred

            if use_domain_adaptation and "domain_logits" in out and hasattr(data, "domain_label"):
                loss_domain = domain_criterion(out["domain_logits"], data.domain_label.long())
                loss = loss + lambda_domain * loss_domain

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data, return_attention=False)
                pred = out["pred"]

                y = data.y.float().view_as(pred)

                loss_pred = pred_criterion(pred, y)
                loss = loss_pred

                if use_domain_adaptation and "domain_logits" in out and hasattr(data, "domain_label"):
                    loss_domain = domain_criterion(out["domain_logits"], data.domain_label.long())
                    loss = loss + lambda_domain * loss_domain

                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_history:
        return model, history

    return model


def predict_with_attention(model, loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, return_attention=True)

            results.append({
                "pred": out["pred"].cpu(),
                "graph_embedding": out["graph_embedding"].cpu(),
                "node_embeddings": out["node_embeddings"].cpu(),
                "attention_weights": [
                    (ei.cpu(), alpha.cpu()) for ei, alpha in out["attention_weights"]
                ],
                "y_true": data.y.cpu() if hasattr(data, "y") else None,
            })

    return results

### Model V2 With multiple Batch effects

import torch
import torch.nn as nn
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
class BiasAdversarialHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128, grl_alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.grl(x)
        return self.net(x)

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
class RamanGATv2MultiBias(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 5,
        edge_dim: int = 1,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        bias_classes: dict | None = None,   # {"Center": 3, "sample_type": 2, ...}
        bias_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
    ):
        super().__init__()

        self.dropout = dropout
        self.bias_classes = bias_classes or {}

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity())

        graph_dim = hidden_channels * 2

        self.pred_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.bias_heads = nn.ModuleDict()
        for bias_name, n_classes in self.bias_classes.items():
            self.bias_heads[bias_name] = BiasAdversarialHead(
                in_dim=graph_dim,
                n_classes=n_classes,
                hidden_dim=bias_hidden_dim,
                grl_alpha=grl_alpha,
            )

    def forward(self, data, return_attention: bool = False):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if getattr(data, "edge_attr", None) is not None else None
        batch = data.batch

        x = self.input_proj(x)

        attention_outputs = []

        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                x_out, att = conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention_outputs.append(att)
            else:
                x_out = conv(x, edge_index, edge_attr=edge_attr)

            x = norm(x_out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_embedding = torch.cat([graph_mean, graph_max], dim=1)

        pred = self.pred_head(graph_embedding)

        out = {
            "pred": pred,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
        }

        if return_attention:
            out["attention_weights"] = attention_outputs

        if len(self.bias_heads) > 0:
            bias_logits = {}
            for bias_name, head in self.bias_heads.items():
                bias_logits[bias_name] = head(graph_embedding)
            out["bias_logits"] = bias_logits

        return out


import copy
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def train_raman_gnn_multibias(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int = 100,
    patience: int = 15,
    lambda_bias: dict | None = None,
    use_bias_adaptation: bool = True,
    return_history: bool = True,
    save_best_path: str | None = None,
):
    pred_criterion = nn.MSELoss()
    bias_criterion = nn.CrossEntropyLoss()

    if lambda_bias is None:
        lambda_bias = {
            "Center": 0.1,
            "sample_type": 0.1,
            "image_id": 0.1,
            "patient_id": 0.1,
        }

    model = model.to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_weighted": [],
        "val_f1_macro": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data in tqdm(train_loader, desc=f"Train epoch {epoch+1}", unit="batch"):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data, return_attention=False)
            pred = out["pred"]
            y = data.y.float().view_as(pred)

            loss_pred = pred_criterion(pred, y)
            loss = loss_pred

            if use_bias_adaptation and "bias_logits" in out:
                for bias_name, logits in out["bias_logits"].items():
                    safe_attr = f"bias_{bias_name.replace('.', '_')}".replace(" ", "_")
                    if hasattr(data, safe_attr):
                        labels = getattr(data, safe_attr).long()
                        loss_bias = bias_criterion(logits, labels)
                        loss = loss + lambda_bias.get(bias_name, 0.1) * loss_bias

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0

        all_val_pred = []
        all_val_true = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                out = model(data, return_attention=False)
                pred = out["pred"]
                y = data.y.float().view_as(pred)

                loss_pred = pred_criterion(pred, y)
                loss = loss_pred

                if use_bias_adaptation and "bias_logits" in out:
                    for bias_name, logits in out["bias_logits"].items():
                        safe_attr = f"bias_{bias_name.replace('.', '_')}".replace(" ", "_")
                        if hasattr(data, safe_attr):
                            labels = getattr(data, safe_attr).long()
                            loss_bias = bias_criterion(logits, labels)
                            loss = loss + lambda_bias.get(bias_name, 0.1) * loss_bias

                val_loss += loss.item()

                pred_class = torch.argmax(pred, dim=1)
                true_class = torch.argmax(y, dim=1)

                all_val_pred.extend(pred_class.detach().cpu().numpy())
                all_val_true.extend(true_class.detach().cpu().numpy())

        val_loss /= max(1, len(val_loader))

        all_val_pred = np.asarray(all_val_pred)
        all_val_true = np.asarray(all_val_true)

        if len(all_val_true) > 0:
            val_f1_weighted = f1_score(all_val_true, all_val_pred, average="weighted", zero_division=0)
            val_f1_macro = f1_score(all_val_true, all_val_pred, average="macro", zero_division=0)
            val_accuracy = accuracy_score(all_val_true, all_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(all_val_true, all_val_pred)
        else:
            val_f1_weighted = np.nan
            val_f1_macro = np.nan
            val_accuracy = np.nan
            val_balanced_accuracy = np.nan

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_weighted"].append(val_f1_weighted)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_accuracy"].append(val_accuracy)
        history["val_balanced_accuracy"].append(val_balanced_accuracy)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_f1_weighted={val_f1_weighted:.4f}, "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if save_best_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "best_val_loss": best_val_loss,
                        "epoch": epoch + 1,
                        "history": history,
                    },
                    save_best_path,
                )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_history:
        return model, history

    return model











