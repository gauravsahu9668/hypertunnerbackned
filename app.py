from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import io
import numpy as np
import base64
import matplotlib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from routes.C_bagging import cBagging_blueprint
from routes.C_dt import c_dt_blueprint
from routes.R_linearRegression import r_linearRegression
from routes.R_DecisonTree import r_decisionTree
from routes.R_bagging import rBagging_blueprint
from routes.R_randomforest import r_randomforest 
from routes.C_logistic import c_logistic
from routes.R_Kneighbors import r_kneighbors
from routes.C_kNeighbors import c_kneighbors
from routes.C_random import c_randomforest
matplotlib.use('Agg') 
app = Flask(__name__)
CORS(app)


app.register_blueprint(cBagging_blueprint)
app.register_blueprint(rBagging_blueprint)
app.register_blueprint(c_dt_blueprint)
app.register_blueprint(r_linearRegression)
app.register_blueprint(r_decisionTree)
app.register_blueprint(r_randomforest)
app.register_blueprint(c_randomforest)
app.register_blueprint(c_logistic)
app.register_blueprint(r_kneighbors)
app.register_blueprint(c_kneighbors)

@app.route('/')
def home():
    return "Flask backend is running."

# @app.route('/train_c-dt', methods=['POST'])
# def train_model():
#     file=request.files.get('file')
#     fileName=request.form.get('fileName')
#     criterion=request.form.get('criterion')
#     splitter=request.form.get('splitter')
#     max_depth=request.form.get('max_depth')
#     min_samples_split=request.form.get('min_samples_split')
#     min_samples_leaf=request.form.get('min_samples_leaf')
#     max_features=request.form.get('max_features')
#     max_leaf_nodes=request.form.get('max_leaf_nodes')
#     min_impurity_decrease=request.form.get('min_impurity_decrease')
#     min_weight_fraction_leaf=request.form.get('min_weight_fraction_leaf')
#     cv_value=request.files.get('cv_value')
#     X, y=make_classification(random_state=42,n_samples=400)

#     feature_names = [f'Feature {i}' for i in range(X.shape[1])]
#     if fileName not in ("No File"):
#         df=pd.read_csv(file)
#         X=df.iloc[:,:-1]
#         y=df.iloc[:,-1]
#         feature_names=X.columns.tolist()


#     model = DecisionTreeClassifier(
#         criterion=criterion,splitter=splitter,
#         max_depth= None if max_depth in ("0") else int(max_depth),
#         min_samples_split=int(min_samples_split),
#         min_samples_leaf=int(min_samples_leaf),
#         min_weight_fraction_leaf=float(min_weight_fraction_leaf),
#         max_features=None if max_features in ("0") else int(max_features),
#         max_leaf_nodes=None if max_leaf_nodes in ("0") else max_leaf_nodes,
#         min_impurity_decrease=float(min_impurity_decrease)
#     )

#     X_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,train_size=0.8)
#     model.fit(X_train, y_train)
#     y_pred=model.predict(x_test)
#     score=accuracy_score(y_pred,y_test)

#     # fig1
#     fig , ax = plt.subplots(figsize=(25,30))
#     tree.plot_tree(model,filled=True,ax=ax,fontsize=15)
#     buf=io.BytesIO()
#     plt.savefig(buf,format='png')
#     buf.seek(0)
#     image_base64=base64.b64encode(buf.read()).decode('utf-8')
#     plt.close(fig)

#     # fig2
#     pca=PCA(n_components=2)
#     X_reduced = pca.fit_transform(X)

#     x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
#     y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(
#         np.linspace(x_min, x_max, 300),
#         np.linspace(y_min, y_max, 300)
#     )
#     grid = np.c_[xx1.ravel(), xx2.ravel()]
#     grid_original = pca.inverse_transform(grid)
#     preds = model.predict(grid_original).reshape(xx1.shape)
#     plt.figure(figsize=(10, 7))
#     plt.contourf(xx1, xx2, preds, alpha=0.3, cmap=plt.cm.tab10)
#     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolors='k')
#     plt.title("Decision Tree Boundary (PCA 2D Projection)")
#     plt.xlabel("PCA 1")
#     plt.ylabel("PCA 2")
#     buf2 = io.BytesIO()
#     plt.savefig(buf2, format='png')
#     buf2.seek(0)
#     plt.close()
#     image_base642 = base64.b64encode(buf2.getvalue()).decode('utf-8')

#     # fig3
#     importances = model.feature_importances_
#     plt.figure(figsize=(10, 6))
#     plt.barh(range(len(importances)), importances)
#     plt.yticks(range(len(importances)), feature_names)
#     plt.xlabel("Feature Importance")
#     plt.title("Decision Tree Feature Importance")
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', bbox_inches='tight')
#     buffer.seek(0)
#     plt.close()
#     image_base643 = base64.b64encode(buffer.getvalue()).decode('utf-8')

#     precison_macro=precision_score(y_test,y_pred,average='macro')
#     precison_micro=precision_score(y_test,y_pred,average='micro')
#     precison_weighted=precision_score(y_test,y_pred,average='weighted')

#     recall_score1=recall_score(y_test,y_pred,average='micro')
#     recall_score2=recall_score(y_test,y_pred,average='macro')
#     recall_score3=recall_score(y_test,y_pred,average='weighted')

#     cross_val_Score=np.mean(cross_val_score(model,X_train,y_train,cv=cv_value))
#     return jsonify({
#         'message': 'Model trained successfully',
#         'fig':image_base64,
#         'fig2':image_base642,
#         'fig3':image_base643,
#         'scores': {
#             'p1':precison_macro,
#             'p2':precison_micro,
#             'p3':precison_weighted,
#             'r1':recall_score1,
#             'r2':recall_score2,
#             'r3':recall_score3,
#             "cv_score":cross_val_Score,
#             'accuracy':score
#         }
#     })

if __name__ == '__main__':
    app.run(debug=True)
