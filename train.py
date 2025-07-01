import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from configs.train_arguements import parse_args
from src.utils.compute_metric import custom_score
from src.models.get_model import get_model

args = parse_args()

# 데이터 불러오기
train = pd.read_csv(args.train_file)
test = pd.read_csv(args.test_file)

X = train[['title', 'full_text']]
y = train['generated']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42) 

# TF-IDF 벡터화
print("Vectorizing text data...")
get_title = FunctionTransformer(lambda x: x['title'], validate=False)
get_text = FunctionTransformer(lambda x: x['full_text'], validate=False)
print("Creating feature union...")
vectorizer = FeatureUnion([
    ('title', Pipeline([('selector', get_title),
                        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000))])),
    ('full_text', Pipeline([('selector', get_text), 
                            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000))])),
])
print("Feature union created.")
# 피처 변환
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print("Text data vectorized.")
# XGBoost 모델 설정 및 학습
model = get_model(args, X_train, y_train)
print(f"Model: {args.model_name}")
model.fit(X_train, y_train)

# 예측 수행
val_probs = model.predict_proba(X_val_vec)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {auc:.4f}")

# test용으로 'paragraph_text'를 'full_text'에 맞게 재명명
test = test.rename(columns={'paragraph_text': 'full_text'})
X_test = test[['title', 'full_text']]

X_test_vec = vectorizer.transform(X_test)

probs = model.predict_proba(X_test_vec)[:, 1]

# 훈련 세트에서 커스텀 메트릭 확인
train_pred = model.predict(X_train_vec)
final_score = custom_score(y_train, train_pred)
print(f"Training Custom Score: {final_score:.4f}")

# 제출 파일 생성
submit = pd.read_csv(args.submission_file)
submit['generated'] = probs
submit.head()

submit.to_csv(args.base_file, index=False)