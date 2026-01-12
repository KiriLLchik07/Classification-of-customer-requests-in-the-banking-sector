from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

class BankingDistilBERT:
    def __init__(self, model_name: str = DISTILBERT_MODEL_NAME, num_labels: int = 77, max_length: int = 64, device: str = "cuda"):
        """
        Инициализация BERT модели
        
        :param model_name: Название предобученной модели
        :type model_name: str
        :param run_labels: Количество классов для классификации
        :type run_labels: int
        :param max_length: Максимальная длина последовательности
        :type max_length: int
        :param device: Устройство для вычислений
        :type device: str
        """

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_class_classification"
        )

        self.max_length = max_length
        self.device = device
        self.model_name = model_name

    def tokenize(self, texts, padding=True, truncation=True, return_tensor="pt"):
        """
        Токенизация текстовых документов
        
        :param texts: список текстовых документов
        :param padding: если длина последовательности меньше max_length,
        :param truncation: Description
        :param return_tensor: Description
        """

        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensor=return_tensor
        )
    
    def predict(self, texts):
        """
        Предсказание классов для списка текстов
        
        :param texts: набор текстовых документов для тестирования
        """
        self.model.eval()
        inputs = self.tokenize(texts).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy(), torch.softmax(logits, dim=1).cpu().numpy()
