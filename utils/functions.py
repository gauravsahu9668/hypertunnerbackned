import io
import base64
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import learning_curve
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_plots(X, y, model, X_train, X_test, y_train, y_test,cv):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    residuals = y_test - y_test_pred

    base64_images = []

    # 1. Actual vs Predicted
    if X.shape[1] == 1:
        fig5, ax5 = plt.subplots()
        ax5.scatter(X, y, color='blue', label='Actual')
        ax5.plot(X, model.predict(X), color='red', label='Regression Line')
        ax5.set_xlabel("Feature")
        ax5.set_ylabel("Target")
        ax5.set_title("Regression Line Fit")
        ax5.legend()
        base64_images.append(fig_to_base64(fig5))

    # 2. Residuals vs Predicted
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test_pred, residuals, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predicted")
    base64_images.append(fig_to_base64(fig2))

    # 3. Residuals Histogram + KDE
    fig3, ax3 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax3, color='green')
    ax3.set_title("Residuals Distribution")
    ax3.set_xlabel("Residual")
    base64_images.append(fig_to_base64(fig3))

    # 4. Q-Q Plot
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot of Residuals")
    base64_images.append(fig_to_base64(fig4))

    # 5. Regression Line (if 1 feature)
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_test_pred, alpha=0.7)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    base64_images.append(fig_to_base64(fig1))

    # 6. Learning Curve
    fig6, ax6 = plt.subplots()
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error')
    ax6.plot(train_sizes, -train_scores.mean(axis=1), label="Training Error")
    ax6.plot(train_sizes, -test_scores.mean(axis=1), label="Validation Error")
    ax6.set_xlabel("Training Set Size")
    ax6.set_ylabel("Mean Squared Error")
    ax6.set_title("Learning Curve")
    ax6.legend()
    base64_images.append(fig_to_base64(fig6))

    return base64_images


import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.base import clone

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_all_logistic_graphs(X, y, model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    base64_images = []

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    base64_images.append(fig_to_base64(fig1))
    plt.close(fig1)

    # 2. Classification Report Heatmap
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    fig2, ax2 = plt.subplots()
    sns.heatmap(df_report, annot=True, cmap="YlGnBu", ax=ax2)
    ax2.set_title("Classification Report")
    base64_images.append(fig_to_base64(fig2))
    plt.close(fig2)

    # 3. ROC Curve
    if y_proba is not None and y_proba.shape[1] <= 10:
        fig3, ax3 = plt.subplots()
        for i in range(y_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
        ax3.plot([0, 1], [0, 1], 'k--')
        ax3.set_title("ROC Curve")
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.legend()
        base64_images.append(fig_to_base64(fig3))
        plt.close(fig3)

    # 4. Precision-Recall Curve
    if y_proba is not None and y_proba.shape[1] <= 10:
        fig4, ax4 = plt.subplots()
        for i in range(y_proba.shape[1]):
            precision, recall, _ = precision_recall_curve(y_test == i, y_proba[:, i])
            ax4.plot(recall, precision, label=f"Class {i}")
        ax4.set_title("Precision-Recall Curve")
        ax4.set_xlabel("Recall")
        ax4.set_ylabel("Precision")
        ax4.legend()
        base64_images.append(fig_to_base64(fig4))
        plt.close(fig4)

    # 5. Permutation Feature Importance
    fig5, ax5 = plt.subplots()
    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
        sorted_idx = result.importances_mean.argsort()
        ax5.boxplot(result.importances[sorted_idx].T, vert=False,
                    labels=np.array(X.columns)[sorted_idx])
        ax5.set_title("Permutation Feature Importance")
    except:
        ax5.text(0.5, 0.5, "Feature importance not available", ha='center')
    base64_images.append(fig_to_base64(fig5))
    plt.close(fig5)

    # 6. PCA 2D Projection
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig6, ax6 = plt.subplots()
        scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax6.set_title("PCA: 2D projection of data")
        base64_images.append(fig_to_base64(fig6))
        plt.close(fig6)

    # 7. Prediction Probability Histogram
    if y_proba is not None:
        fig7, ax7 = plt.subplots()
        for i in range(y_proba.shape[1]):
            sns.histplot(y_proba[:, i], kde=True, bins=25, label=f"Class {i}", ax=ax7)
        ax7.set_title("Prediction Probability Distribution")
        ax7.set_xlabel("Probability")
        ax7.legend()
        base64_images.append(fig_to_base64(fig7))
        plt.close(fig7)

    # 8. Decision Boundary using PCA
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        model_clone = clone(model)
        model_clone.fit(X_pca, y)

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model_clone.predict(grid).reshape(xx.shape)

        fig8, ax8 = plt.subplots()
        ax8.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')
        ax8.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1', edgecolor='k')
        ax8.set_title("Decision Boundary (PCA Projection)")
        base64_images.append(fig_to_base64(fig8))
        plt.close(fig8)
    except Exception as e:
        print("Error generating decision boundary:", e)

    return base64_images
