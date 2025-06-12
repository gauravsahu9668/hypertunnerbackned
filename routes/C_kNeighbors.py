from flask import Blueprint
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from utils.cfunction import generate_classification_plots
import io
import numpy as np
import base64
import matplotlib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils.functions import generate_plots
from sklearn.neighbors import KNeighborsClassifier
c_kneighbors= Blueprint('c_kneighbors',__name__)
@c_kneighbors.route('/train_c-kneighbors',methods=['POST'])
def R_DecisionTree():

    file=request.files.get('file')
    fileName=request.form.get('fileName')
    n_neighbors=request.form.get('n_neighbors')
    weights=request.form.get('weights')
    cv_value=int(request.form.get('cv_value'))
    algorithm=request.form.get('algorithm')
    leaf_size=request.form.get('leaf_size')
    p=request.form.get('p')

    X, y=make_classification(random_state=42,n_samples=400)
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    if fileName not in ("No File"):
        df=pd.read_csv(file)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        feature_names=X.columns.tolist()

    model=KNeighborsClassifier(
        weights=weights,
        n_neighbors=int(n_neighbors),
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        p=int(p)
    )
    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred=model.predict(x_test)
    score=accuracy_score(y_pred,y_test)
    p1=precision_score(y_test,y_pred,average='macro')
    p2=precision_score(y_test,y_pred,average='micro')
    p3=precision_score(y_test,y_pred,average='weighted')
    r1=recall_score(y_test,y_pred,average='micro')
    r2=recall_score(y_test,y_pred,average='macro')
    r3=recall_score(y_test,y_pred,average='weighted')
    Y_train_pred=model.predict(X_train)
    training_accuracy=accuracy_score(Y_train_pred,y_train)
    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=cv_value))
    base64_images =generate_classification_plots(X, y, model, X_train, x_test, y_train, y_test,cv=int(cv_value))
    base64_objects = [{f'fig{i+1}': img} for i, img in enumerate(base64_images)]
    return jsonify({
        'message': 'Model trained successfully',
        'graphs':base64_objects,
        'scores': [
            { 'name':'Testing accuracy_score',
             'value':score},
             { 'name':'Training accuracy_score',
             'value':training_accuracy},
             {
                 'name':'Cross_val_score',
                 'value':cross_val_Score
             },
            { 'name':'Precision_micro',
             'value':p1},
             { 'name':'Precision_macro',
             'value':p2},
             { 'name':'Precision_weighted',
             'value':p3},
             { 'name':'Recall_micro',
             'value':r1},
             { 'name':'Recall_macro',
             'value':r2},
             { 'name':'Recall_weighted',
             'value':r3}
        ]
    })