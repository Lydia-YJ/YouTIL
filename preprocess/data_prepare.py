import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 원본 데이터 로드
df = pd.read_csv("/home/yuri011228/moderation/comment_data.csv") 
df = df[["Sentence", "Label"]].rename(columns={"Sentence": "text", "Label": "label"})

# train/test 분할
train_df, test_df = train_test_split(
    df, 
    test_size=0.3, 
    stratify=df["label"], #클래스 분포 비율을 맞추기 위해
    random_state=42)

# 폴더 생성
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# 저장
train_df.to_csv("train/train_data.csv", index=False)
test_df.to_csv("test/test_data.csv", index=False)

print("데이터 분할 및 저장 완료")