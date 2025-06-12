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
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from utils.functions import fig_to_base64
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score,
    max_error,root_mean_squared_error
)
from utils.functions import generate_plots
r_decisionTree= Blueprint('r_dt',__name__)
@r_decisionTree.route('/train_r-dt',methods=['POST'])
def R_DecisionTree():
    file=request.files.get('file')
    fileName=request.form.get('fileName')
    criterion=request.form.get('criterion')
    splitter=request.form.get('splitter')
    max_depth=request.form.get('max_depth')
    min_samples_split=request.form.get('min_samples_split')
    min_samples_leaf=request.form.get('min_samples_leaf')
    max_features=request.form.get('max_features')
    max_leaf_nodes=request.form.get('max_leaf_nodes')
    min_impurity_decrease=request.form.get('min_impurity_decrease')
    min_weight_fraction_leaf=request.form.get('min_weight_fraction_leaf')
    cv_value=request.form.get('cv_value')
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


    model = DecisionTreeRegressor(
        criterion=criterion,splitter=splitter,
        max_depth= None if max_depth in ("0") else int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        min_weight_fraction_leaf=float(min_weight_fraction_leaf),
        max_features=None if max_features in ("0") else int(max_features),
        max_leaf_nodes=None if max_leaf_nodes in ("0") else int(max_leaf_nodes),
        min_impurity_decrease=float(min_impurity_decrease)
        
    )
    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
    model.fit(X_train, y_train)
    y_test_pred=model.predict(x_test)
    y_train_pred=model.predict(X_train)

    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=int(cv_value)))
    base64_images = generate_plots(X, y, model, X_train, x_test, y_train, y_test,cv=int(cv_value))

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
]
    })