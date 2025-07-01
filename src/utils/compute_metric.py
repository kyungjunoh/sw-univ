import numpy as np
from sklearn.metrics import roc_auc_score


def custom_score(y_true, y_pred):
    """
    Custom ROC-AUC score function for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities for the positive class
        
    Returns:
        float: ROC-AUC score
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            # If all labels are the same, return 0.5 (random performance)
            return 0.5
        
        # Calculate ROC-AUC score
        auc = roc_auc_score(y_true, y_pred)
        
        return auc
        
    except Exception as e:
        print(f"Error in custom_score: {e}")
        return 0.5


def xgb_auc_eval(y_pred, y_true):
    """
    XGBoost 호환 AUC 평가 함수
    XGBoost의 eval_metric으로 사용할 수 있는 형태
    
    Args:
        y_pred: 예측 확률값 (DMatrix의 get_label()로부터)
        y_true: 실제 레이블 (DMatrix 객체)
        
    Returns:
        tuple: (eval_name, eval_result)
    """
    try:
        # XGBoost DMatrix에서 실제 레이블 추출
        labels = y_true.get_label()
        
        # ROC-AUC 계산
        auc = roc_auc_score(labels, y_pred)
        
        # XGBoost는 (평가명, 점수) 튜플을 기대
        return 'auc', auc
        
    except Exception as e:
        print(f"Error in xgb_auc_eval: {e}")
        return 'auc', 0.5


def custom_score_wrapper(estimator, X, y):
    """
    Scikit-learn의 cross_val_score와 호환되는 스코어링 함수
    
    Args:
        estimator: 훈련된 모델
        X: 특성 데이터
        y: 실제 레이블
        
    Returns:
        float: ROC-AUC 점수
    """
    try:
        # 분류 모델의 경우 predict_proba 사용
        if hasattr(estimator, 'predict_proba'):
            y_pred_proba = estimator.predict_proba(X)[:, 1]  # 양성 클래스 확률
        else:
            y_pred_proba = estimator.predict(X)
        
        return custom_score(y, y_pred_proba)
        
    except Exception as e:
        print(f"Error in custom_score_wrapper: {e}")
        return 0.5
