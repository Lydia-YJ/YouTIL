from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingManager:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print("✅ 모델 로딩 완료")

    def get_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            emb = output.last_hidden_state[:, 0, :]
            return emb.squeeze(0).float().cpu().numpy()
    