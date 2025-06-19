import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from preprocess.embeddings import EmbeddingManager
from preprocess.text_preprocess import TextPreprocessor
from preprocess.data_loader import get_loaders
#from models.MLP import MLP
from models.LSTM import LSTM

class Config:
    MODEL_TYPE = "lstm"

    INPUT_DIM = 1024
    DROPOUT = 0.3
    BATCH_SIZE = 64
    EPOCHS = 100
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

    # LSTM 전용 설정
    HIDDEN_DIM = 128
    NUM_LAYERS = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv("/home/yuri011228/moderation/train/train_data.csv")
test_df = pd.read_csv("/home/yuri011228/moderation/test/test_data.csv")

preprocessor = TextPreprocessor()
x_train = preprocessor.clean_all(train_df["text"].tolist())
y_train = train_df["label"].tolist()

x_test = preprocessor.clean_all(test_df["text"].tolist())
y_test = test_df["label"].tolist()

embedder = EmbeddingManager("intfloat/multilingual-e5-large-instruct") 

print("✅ 임베딩 추출 시작")
# for i, text in enumerate(x_train[:5]):
#     print(f" - {i+1}/{len(x_train)}: {text[:30]}...")
#     emb = embedder.get_embedding(text)
#     print(f"   → 벡터 shape: {emb.shape}")

x_train_vec = [embedder.get_embedding(text) for text in x_train]
x_test_vec = [embedder.get_embedding(text) for text in x_test]

train_loader, test_loader = get_loaders(x_train_vec, y_train, 
                                        x_test_vec, y_test, 
                                        batch_size=Config.BATCH_SIZE)

if Config.MODEL_TYPE == "mlp":
    model = MLP(
        input_dim=Config.INPUT_DIM, 
        dropout_rate=Config.DROPOUT
    ).to(device)
elif Config.MODEL_TYPE == "lstm":
    model = LSTM(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)
else:
    raise ValueError(f"Unsupported model type: {Config.MODEL_TYPE}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("학습 시작")
for epoch in range(100):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(batch_x.to(device))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f}")

print("평가 시작")
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x).cpu().numpy()
        preds = (preds > 0.5).astype(int)
        all_preds.extend(preds)

print("\n📊 분류 성능 보고서:")
print(classification_report(y_test, all_preds))
print(f"Accuracy: {accuracy_score(y_test, all_preds):.4f}")