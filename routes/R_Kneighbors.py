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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score,
    max_error,root_mean_squared_error
)
from utils.functions import generate_plots
r_kneighbors= Blueprint('r_kneighbors',__name__)
@r_kneighbors.route('/train_r-kneighbors',methods=['POST'])
def R_DecisionTree():

    file=request.files.get('file')
    fileName=request.form.get('fileName')
    n_neighbors=request.form.get('n_neighbors')
    weights=request.form.get('weights')
    cv_value=request.form.get('cv_value')
    algorithm=request.form.get('algorithm')
    leaf_size=request.form.get('leaf_size')
    p=request.form.get('p')

    X,y=make_regression(n_samples=400,n_features=1,n_informative=1,noise=10,random_state=42)
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    if fileName not in ("No File"):
        df=pd.read_csv(file)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        feature_names=X.columns.tolist()

    model=KNeighborsRegressor(
        weights=weights,
        n_neighbors=int(n_neighbors),
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        p=int(p)
    )
    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
    model.fit(X_train, y_train)
    y_test_pred=model.predict(x_test)
    y_train_pred=model.predict(X_train)

    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=int(cv_value)))
    base64_images = generate_plots(X, y, model, X_train, x_test, y_train, y_test,cv=int(cv_value))
    base64_objects = [{f'fig{i+1}': img} for i, img in enumerate(base64_images)]
    return jsonify({
        'message': 'Model trained successfully',
         'graphs':base64_objects,
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
     {'name': 'N features fit', 'value': model.n_features_in_},
]
    })