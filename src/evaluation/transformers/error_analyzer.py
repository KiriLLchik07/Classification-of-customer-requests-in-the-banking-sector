import pandas as pd
import torch
from tqdm import tqdm

def error_analysis(model, dataloader, texts: list[str], label2intent: dict[int, str], 
                   device: str,top_k: int = 3) -> pd.DataFrame:
    model.eval()
    model.to(device)

    errors = []
    idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Error analysis"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(preds)):
                true_label = batch["labels"][i].item()
                pred_label = preds[i].item()

                if true_label != pred_label:
                    top_probs, top_classes = torch.topk(probs[i], k=top_k)

                    errors.append({
                        "text": texts[idx],
                        "true_label": true_label,
                        "true_intent": label2intent[true_label],
                        "pred_label": pred_label,
                        "pred_intent": label2intent[pred_label],
                        "pred_confidence": probs[i][pred_label].item(),
                        "top_k_intents": [
                            label2intent[c.item()] for c in top_classes
                        ],
                        "top_k_probs": top_probs.cpu().tolist(),
                    })

                idx += 1

    return pd.DataFrame(errors)
