import xgboost as xgb
import numpy as np
from src.models.model.xgb import xgb_model, xgb_automl_model

# `x3d_s` 모델 선택
def get_model(args, X_train=None, y_train=None):

    if args.model_name == "xgboost":
        model = xgb_model(args)
        return model
    elif args.model_name == "xgboost_automl":
        model = xgb_automl_model(args, X_train, y_train, n_trials=args.n_trials)
        return model