from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BankingTransformer:
    def __init__(self, model_name: str, num_labels: int = 77, max_length: int = 64, device: str = "cuda"):
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        self.max_length = max_length
        self.device = device
        self.model_name = model_name
    
    def predict(self, texts):
        """
        Предсказание классов для списка текстов
        
        :param texts: набор текстовых документов для тестирования
        """
        self.model.eval()
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                max_length=self.max_length, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()
