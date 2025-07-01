import numpy as np

def custom_objective(y_true, y_pred, alpha=0.5, eps=1e-8):
    mse = np.mean((y_true - y_pred) ** 2)
    gradient = -2 * (y_true - y_pred)
    hessian = np.ones_like(y_true) * 2
    
    return gradient, hessian

