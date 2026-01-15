from torch.utils.data import DataLoader
import time
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import mlflow

from src.data.transformer_dataset import TransformerDataset
from src.training.transformers.transformer_trainer import TransformerTrainer

def train_transformer(model_wrapper, train_texts, train_labels, val_texts, val_labels, 
                      epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    
    device = model_wrapper.device
    model = model_wrapper.model.to(device)
    tokenizer = model_wrapper.tokenizer

    train_ds = TransformerDataset(train_texts, train_labels, tokenizer, model_wrapper.max_length)
    val_ds = TransformerDataset(val_texts, val_labels, tokenizer, model_wrapper.max_length)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, 
                                                num_warmup_steps=int(0.1 * total_steps))
    
    trainer = TransformerTrainer(model, optimizer, scheduler, device)

    val_metrics = []
    strart = time.time()
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss, val_f1_macro = trainer.eval_epoch(val_dataloader)
        print(f"Epoch {epoch + 1}: val_f1_macro={val_f1_macro:.4f}")
        val_metrics.append(val_f1_macro)
    training_time = time.time() - strart

    return model, val_metrics, training_time
