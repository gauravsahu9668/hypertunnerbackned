from flask import Blueprint
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import io
import numpy as np
import base64
import matplotlib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils.cfunction import generate_classification_plots


cBagging_blueprint = Blueprint('c-bagging',__name__)


@cBagging_blueprint.route('/train_c-bagging',methods=['POST'])
def c_Bagging():
    file=request.files.get('file')
    fileName=request.form.get('fileName')
    estimator=request.form.get('estimator')
    n_estimators=request.form.get('n_estimators')
    cv_value=request.form.get('cv_value')
    max_samples=request.form.get('max_samples')
    max_features=request.form.get('max_features')
    bootstrap=request.form.get('bootstrap')
    bootstrap_features=request.form.get('bootstrap_features')
    X, y=make_classification(random_state=42,n_samples=400)
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    if fileName not in ("No File"):
        df=pd.read_csv(file)
        X=df.iloc[:,:-1]
        y=df.iloc[:,-1]
        feature_names=X.columns.tolist()

    baseModel=DecisionTreeClassifier()
    if estimator=='KNeighborsClassifier':
        baseModel=KNeighborsClassifier()
    if estimator=='LogisticRegression':
        baseModel=LogisticRegression()
    if estimator=='SVC':
        baseModel=SVC(probability=True)
    if estimator=='GaussianNB':
        baseModel=GaussianNB()

    
    if (bootstrap=="true"):
        bootstrap=True
    else:
        bootstrap=False

    if (bootstrap_features=="true"):
        bootstrap_features=True
    else:
        bootstrap_features=False


    model=BaggingClassifier(
        estimator=baseModel,
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        max_features=float(max_features),
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
        oob_score=bootstrap
    )
    model1=BaggingClassifier(
        estimator=baseModel,
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        max_features=float(max_features),
        bootstrap=bootstrap,
        bootstrap_features=bootstrap_features,
        oob_score=bootstrap
    )
    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
    model.fit(X_train, y_train)
    y_pred=model.predict(x_test)
    Y_train_pred=model.predict(X_train)



    score=accuracy_score(y_pred,y_test)
    oob_score=0
    if bootstrap:
       oob_score=model.oob_score_

    training_accuracy=accuracy_score(Y_train_pred,y_train)
    p1=precision_score(y_test,y_pred,average='macro')
    p2=precision_score(y_test,y_pred,average='micro')
    p3=precision_score(y_test,y_pred,average='weighted')
    r1=recall_score(y_test,y_pred,average='micro')
    r2=recall_score(y_test,y_pred,average='macro')
    r3=recall_score(y_test,y_pred,average='weighted')
    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=int(cv_value)))

    # fig1
    base64_images=generate_classification_plots(X, y, model, X_train, x_test, y_train, y_test, cv=int(cv_value))
    base64_objects = [{f'fig{i+1}': img} for i, img in enumerate(base64_images)]
    return jsonify({
        'message': 'Model trained successfully',
         'graphs':base64_objects,
        'scores': [
            { 'name':'Testing accuracy score',
             'value':score},
             { 'name':'Training accuracy score',
             'value':training_accuracy},
             {
                'name':'Cross_val_score',
                'value':cross_val_Score
             },
             {
                'name':'OOB score',
                'value':oob_score
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