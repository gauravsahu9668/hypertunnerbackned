from flask import Blueprint
from flask import Flask, request, jsonify
from flask_cors import CORS
import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import io
import numpy as np
import base64
from sklearn.datasets import make_regression
from utils.functions import generate_plots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score,
    max_error,root_mean_squared_error
)
r_linearRegression= Blueprint('r_linearRegression',__name__)

@r_linearRegression.route('/train_r-linear',methods=['POST'])
def R_linearRegression():
    file=request.files.get('file')
    fileName=request.form.get('fileName')
    fit_intercept=request.form.get('fit_intercept')
    copy_X=request.form.get('copy_X')
    cv_value=request.form.get('cv_value')
    positive=request.form.get('positive')
    degree=int(request.form.get('degree'))

    X,y=make_regression(n_samples=400,n_features=1,n_informative=1,noise=10,random_state=42)
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    if fileName not in ("No File"):
        df=pd.read_csv(file)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        feature_names=X.columns.tolist()

    X_graph=X    
    if(degree>1):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(getattr(X, 'columns', None))
        X = pd.DataFrame(X_poly, columns=feature_names)

    
    if (fit_intercept=="true"):
        fit_intercept=True
    else:
        fit_intercept=False


    if (copy_X=="true"):
        copy_X=True
    else:
        copy_X=False


    if (positive=="true"):
        positive=True
    else:
        positive=False

    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
    model=LinearRegression(
        fit_intercept=fit_intercept,
        positive=positive,
        copy_X=copy_X
    )
    model.fit(X_train,y_train)
    y_test_pred=model.predict(x_test)
    y_train_pred=model.predict(X_train)

    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=int(cv_value)))
    base64_images = generate_plots(X ,y, model, X_train, x_test, y_train, y_test,cv=int(cv_value))
    response_graphs = []

    if X_graph.shape[1] == 1:
        fig5, ax5 = plt.subplots()
        ax5.scatter(X_graph, y, color='blue', label='Actual')  # Original 1D feature
        X_graph_transformed = poly.transform(X_graph) if degree > 1 else X_graph
        y_graph_pred = model.predict(X_graph_transformed)
        ax5.plot(X_graph, y_graph_pred, color='red', label='Regression Curve')
        ax5.set_xlabel("Feature")
        ax5.set_ylabel("Target")
        ax5.set_title("Polynomial Regression Curve")
        ax5.legend()
        base64_images.append(fig_to_base64(fig5))

    
    if X_graph.shape[1] == 1:
        response_graphs.extend([
        {'Regression Line': base64_images[0]},
        {'Residuals vs Predicted': base64_images[1]},
        {'Residuals Histogram + KDE': base64_images[2]},
        {'Q-Q Plot': base64_images[3]},
        {'Actual_predicted': base64_images[4]},
        {'Learning Curve': base64_images[5]},
    ])
    else:
        response_graphs.extend([
        {'Residuals vs Predicted': base64_images[0]},
        {'Residuals Histogram + KDE': base64_images[1]},
        {'Q-Q Plot': base64_images[2]},
        {'Actual_predicted': base64_images[3]},
        {'Learning Curve': base64_images[4]},
    ])
        
    return jsonify({
        'message': 'Model trained successfully',
         'graphs':response_graphs,
         'image':base64_images,
        'scores':  [
    {'name': 'R-squared Score', 'value': r2_score(y_test, y_test_pred)},
    {'name': 'Training r-squared Score', 'value': r2_score(y_train, y_train_pred)},
    {'name': 'Mean Absolute Error', 'value': mean_absolute_error(y_test, y_test_pred)},
    {'name': 'Mean Squared Error', 'value': mean_squared_error(y_test, y_test_pred)},
    {'name': 'Root Mean Squared Error', 'value': root_mean_squared_error(y_test, y_test_pred)},
    {'name': 'Mean Absolute Percentage Error', 'value': mean_absolute_percentage_error(y_test, y_test_pred)},
    {'name': 'Explained Variance Score', 'value': explained_variance_score(y_test, y_test_pred)},
    {'name': 'Maximum Error', 'value': max_error(y_test, y_test_pred)},
    {'name': 'Cross_Val_Score', 'value': cross_val_Score},
     {'name': 'Intercept', 'value': model.intercept_},
]
    })



def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# def generate_plots(X, y, model, X_train, X_test, y_train, y_test,cv):
#     y_test_pred = model.predict(X_test)
#     y_train_pred = model.predict(X_train)
#     residuals = y_test - y_test_pred

#     base64_images = []

#     # 1. Actual vs Predicted
#     if X.shape[1] == 1:
#         fig5, ax5 = plt.subplots()
#         ax5.scatter(X, y, color='blue', label='Actual')
#         ax5.plot(X, model.predict(X), color='red', label='Regression Line')
#         ax5.set_xlabel("Feature")
#         ax5.set_ylabel("Target")
#         ax5.set_title("Regression Line Fit")
#         ax5.legend()
#         st.pyplot(fig5)
#         base64_images.append(fig_to_base64(fig5))

#     # 2. Residuals vs Predicted
#     fig2, ax2 = plt.subplots()
#     ax2.scatter(y_test_pred, residuals, alpha=0.7)
#     ax2.axhline(0, color='red', linestyle='--')
#     ax2.set_xlabel("Predicted")
#     ax2.set_ylabel("Residuals")
#     ax2.set_title("Residuals vs Predicted")
#     st.pyplot(fig2)
#     base64_images.append(fig_to_base64(fig2))

#     # 3. Residuals Histogram + KDE
#     fig3, ax3 = plt.subplots()
#     sns.histplot(residuals, kde=True, ax=ax3, color='green')
#     ax3.set_title("Residuals Distribution")
#     ax3.set_xlabel("Residual")
#     st.pyplot(fig3)
#     base64_images.append(fig_to_base64(fig3))

#     # 4. Q-Q Plot
#     fig4 = plt.figure()
#     ax4 = fig4.add_subplot(111)
#     stats.probplot(residuals, dist="norm", plot=ax4)
#     ax4.set_title("Q-Q Plot of Residuals")
#     st.pyplot(fig4)
#     base64_images.append(fig_to_base64(fig4))

#     # 5. Regression Line (if 1 feature)
#     fig1, ax1 = plt.subplots()
#     ax1.scatter(y_test, y_test_pred, alpha=0.7)
#     ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     ax1.set_xlabel("Actual")
#     ax1.set_ylabel("Predicted")
#     ax1.set_title("Actual vs Predicted")
#     st.pyplot(fig1)
#     base64_images.append(fig_to_base64(fig1))

#     # 6. Learning Curve
#     fig6, ax6 = plt.subplots()
#     train_sizes, train_scores, test_scores = learning_curve(
#         model, X, y, cv=cv, scoring='neg_mean_squared_error')
#     ax6.plot(train_sizes, -train_scores.mean(axis=1), label="Training Error")
#     ax6.plot(train_sizes, -test_scores.mean(axis=1), label="Validation Error")
#     ax6.set_xlabel("Training Set Size")
#     ax6.set_ylabel("Mean Squared Error")
#     ax6.set_title("Learning Curve")
#     ax6.legend()
#     st.pyplot(fig6)
#     base64_images.append(fig_to_base64(fig6))

#     return base64_images

