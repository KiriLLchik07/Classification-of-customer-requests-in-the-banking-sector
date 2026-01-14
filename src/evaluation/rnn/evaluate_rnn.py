import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, dataloader, device):
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            predictions = logits.argmax(dim=1)

            preds.extend(predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }
