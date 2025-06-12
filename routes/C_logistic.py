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
from sklearn.preprocessing import PolynomialFeatures
from utils.functions import generate_all_logistic_graphs

c_logistic = Blueprint('c-logistic',__name__)


@c_logistic.route('/train_c-logistic',methods=['POST'])
def c_Logistic():
    file=request.files.get('file')
    fileName=request.form.get('fileName')
    penalty=request.form.get('penalty')
    cv_value=request.form.get('cv_value')
    max_iter=request.form.get('max_iter')
    c=request.form.get('c')
    l1_ratio=request.form.get('l1_ratio')
    fit_intercept=request.form.get('fit_intercept')
    multi_class=request.form.get('multi_class')
    degree=int(request.form.get('degree'))
    X, y=make_classification(random_state=42,n_samples=400)
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

    

    model=LogisticRegression(
        penalty=None if penalty=='None' else penalty,
        max_iter=int(max_iter),
        C=float(c),
        l1_ratio=float(l1_ratio) if penalty in ("elasticnet") else None,
        fit_intercept=fit_intercept,
        multi_class='multinomial',
        solver='saga'
    )
    

    X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
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
    cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=int(cv_value)))



    # fig1
    base64_images=generate_all_logistic_graphs(X, y, model, X_train, x_test, y_train, y_test)
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