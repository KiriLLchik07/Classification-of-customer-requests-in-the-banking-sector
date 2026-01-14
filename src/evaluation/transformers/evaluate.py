import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_transformer(model_wrapper, dataloader, device):
    model = model_wrapper.model
    model.eval()
    model.to(device)
    
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            targets.extend(batch["labels"].cpu().numpy())

    return {
        "classification_report": classification_report(targets, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(targets, preds)
    }
