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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
