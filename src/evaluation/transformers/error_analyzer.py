import pandas as pd
import torch

def error_analysis(model, dataloader, texts, label_encoder, device, top_k: int = 5):
    model.eval()
    errors = []

    idx = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range (len(probs)):
                true_label = batch['labels'][i].item()
                pred_label = preds[i].item()

                if true_label != pred_label:
                    top_probs, top_classes = torch.topk(probs[i], k=top_k)

                    errors.append({
                        "text": texts[idx],
                        "true_labels": label_encoder.inverse_transform([true_label])[0],
                        "pred_label": label_encoder.inverse_transform([pred_label])[0],
                        "top_k_preds": [
                            label_encoder.inverse_transform([clas.item()])[0] for clas in top_classes
                        ],
                        "top_k_probs": top_probs.cpu().numpy().tolist()
                    })

                idx +=1
    
    return pd.DataFrame(errors)
