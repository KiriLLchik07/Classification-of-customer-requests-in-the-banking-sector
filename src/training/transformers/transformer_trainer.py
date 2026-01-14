import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

class TransformerTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Обучение модели"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
        
        return total_loss/len(dataloader)
    
    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Валидация модели"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                total_loss += outputs.loss.item()
                preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())

        f1_macro = f1_score(targets, preds, average='macro')
        return total_loss / len(dataloader), f1_macro
