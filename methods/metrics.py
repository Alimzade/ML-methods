import numpy as np
import scipy
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import silhouette_score
from scipy import stats


def evaluate_classification_model(model, X, y_true):
    # predict the class labels for input data
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return {'accuracy': acc, 'precision': prec, 'recall': recall, 'F1-score': F1, 'ROC AUC': roc_auc, 'confusion_matrix': cm}

def evaluate_regression_model(model, X, y_true):
    # predict the target variable for input data
    y_pred = model.predict(X)
    
    # compute mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # compute mean squared error
    mse = np.mean((y_true - y_pred)**2)
    
    # compute root mean squared error
    rmse = np.sqrt(mse)
    
    # compute R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    R2 = 1 - (ss_res / ss_tot)
    
    # compute Pearson correlation
    pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # compute Spearman correlation
    spearman_corr = scipy.stats.spearmanr(y_true, y_pred).correlation

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': R2, 'Pearson correlation': pearson_corr, 'Spearman correlation': spearman_corr}

def evaluate_clustering_model(model, X):
    # predict cluster labels for input data
    y_pred = model.predict(X)

    # compute silhouette score
    silhouette = silhouette_score(X, y_pred)

    return {'silhouette_score': silhouette}

