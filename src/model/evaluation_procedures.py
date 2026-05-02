import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _get_inputs_and_labels(batch, device):
    if "img" in batch:
        x = batch["img"]
    else:
        x = batch["x"]

    labels = batch["label"]

    return x.to(device), labels


def evaluate_model(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    tot_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in valloader:
        x, labels = _get_inputs_and_labels(batch, device)

        with torch.no_grad():
            if isinstance(model, torch.nn.ModuleDict):
                output = model["encoder"](x)
            else:
                output = model(x)

        if model.is_complete_model:
            logits = output.cpu()
        else:
            logits = server_model_proxy.get_logits(embeddings=output)

        labels_cpu = labels.cpu()

        tot_loss += F.cross_entropy(logits, labels_cpu, reduction="sum").item()

        preds = logits.argmax(dim=1)
        all_labels.extend(labels_cpu.numpy())
        all_preds.extend(preds.numpy())

        if logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    metrics = {
        "loss": tot_loss / len(valloader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Extra fraud metrics only for binary classification.
    if len(all_probs) == len(all_labels):
        y_prob = np.asarray(all_probs)

        metrics["precision"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, zero_division=0)
        )

        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auroc"] = 0.0

        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["auprc"] = 0.0

    return metrics


def evaluate_client_and_server_clf_head(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    client_loss, client_corrects = 0., 0
    server_loss, server_corrects = 0., 0

    for batch in valloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            client_embs = model["encoder"](img)
            client_logits = model["clf_head"](client_embs)
            server_logits = server_model_proxy.get_logits(embeddings=client_embs).to(device)

        client_loss += F.cross_entropy(client_logits, labels, reduction="sum").item()
        client_corrects += (client_logits.argmax(dim=1) == labels).sum().item()
        server_loss += F.cross_entropy(server_logits, labels, reduction="sum").item()
        server_corrects += (server_logits.argmax(dim=1) == labels).sum().item()

    return {
        "client_loss": client_loss / len(valloader.dataset),
        "client_accuracy": client_corrects / len(valloader.dataset),
        "loss": server_loss / len(valloader.dataset),
        "accuracy": server_corrects / len(valloader.dataset),
    }


def evaluate_ushaped(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    loss, corrects = 0., 0
    for batch in valloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            client_embs = model["encoder"](img)

        server_embs = server_model_proxy.u_forward_inference(embeddings=client_embs).to(device)

        with torch.no_grad():
            client_logits = model["clf_head"](server_embs)

        loss += F.cross_entropy(client_logits, labels, reduction="sum").item()
        corrects += (client_logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": loss / len(valloader.dataset),
        "accuracy": corrects / len(valloader.dataset),
    }
