import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from src.utils.compute_metric import custom_score, custom_score_wrapper

def xgb_model(args):
    # Use XGBClassifier for binary classification with ROC-AUC
    model = xgb.XGBClassifier(
            objective='binary:logistic',  # 이진 분류용 로지스틱 회귀
            eval_metric='auc',  # AUC 평가 메트릭 사용
            random_state=42,
            n_jobs=-1
            )
    return model

def xgb_automl_model(args, X_train, y_train, n_trials=500):
    """
    AutoML을 사용한 XGBoost 모델 최적화
    """
    def objective(trial):
        # 하이퍼파라미터 공간 정의
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # XGBoost 분류 모델 생성 (ROC-AUC용)
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # 이진 분류
            eval_metric='auc',  # AUC 평가 메트릭
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        # custom_score_wrapper를 scorer로 변환
        custom_scorer = make_scorer(custom_score_wrapper, greater_is_better=True)
        
        # Cross-validation으로 성능 평가
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, 
            scoring=custom_scorer,
            n_jobs=-1
        )
        
        # custom_score는 이미 최대화가 목표이므로 평균값을 그대로 반환
        return cv_scores.mean()
    
    # Optuna study 생성 및 최적화 실행
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial custom_score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # 최적 파라미터로 최종 모델 생성
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        **study.best_params
    )
    
    return best_model