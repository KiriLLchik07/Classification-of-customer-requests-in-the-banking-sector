import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from src.data.tokenizer import TextTokenizer
from src.data.dataset import BankingDataset
from src.data.collate import collate_fn

from src.models.rnn.lstm_gru import RNNClassifier
from src.training.train_one_epoch import train_one_epoch
from src.evaluation.evaluate_rnn import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR_ROOT = PROJECT_ROOT / 'data/processed'
MODEL_SAVE_ROOT = PROJECT_ROOT / "models"
MODEL_SAVE_ROOT.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
EPOCHS = 25
MAX_LEN = 80
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LR = 1e-3

def train_eval_rnn_model (model, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, device):
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model=model, dataloader=train_dataloader,
                                    optimizer=optimizer, criterion=criterion, device=device)
        
        metrics_val = evaluate(model, val_dataloader, device)
        print(
            f"Epoch: {epoch} | "
            f"Train loss: {train_loss:.4f} | "
            f"F1-macro (val sample): {metrics_val['f1_macro']:.4f}"
        )

        metrcis_test = evaluate(model, test_dataloader, device)
    return metrcis_test

def save_model (model, tokenizer, path, model_name):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.word2idx
        },
        path / f"rnn_classifier_{model_name}.pt"
    )

train_df = pd.read_csv(DATA_DIR_ROOT / 'train_df.csv')
val_df = pd.read_csv(DATA_DIR_ROOT / 'val_df.csv')
test_df = pd.read_csv(DATA_DIR_ROOT / 'test_df.csv')

X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()
X_val, y_val = val_df['text'].tolist(), val_df['label'].tolist()
X_test, y_test = test_df['text'].tolist(), test_df['label'].tolist()

tokenizer = TextTokenizer()
tokenizer.fit(X_train)

train_dataset = BankingDataset(texts=X_train, labels=y_train, tokenizer=tokenizer, max_length=MAX_LEN)
val_dataset = BankingDataset(texts=X_val, labels=y_val, tokenizer=tokenizer, max_length=MAX_LEN)
test_dataset = BankingDataset(texts=X_test, labels=y_test, tokenizer=tokenizer, max_length=MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model_lstm = RNNClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=len(set(y_train)),
    rnn_type='lstm'
).to(device)

model_gru = RNNClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=len(set(y_train)),
    rnn_type='gru'
).to(device)

optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LR)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

metrcis_lstm = train_eval_rnn_model(
    model_lstm,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer_lstm,
    criterion,
    device
)

print(f"Результаты модели LSTM\n {metrcis_lstm['f1_macro']:.4f}\n")

metrcis_gru = train_eval_rnn_model(
    model_gru,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer_gru,
    criterion,
    device
)

print(f"Результаты модели GRU\n {metrcis_gru['f1_macro']:.4f}\n")

save_model(model_lstm, tokenizer, MODEL_SAVE_ROOT, 'lstm')
print("LSTM модель сохранена!\n")

save_model(model_gru, tokenizer, MODEL_SAVE_ROOT, 'gru')
print("GRU модель сохранена!")
