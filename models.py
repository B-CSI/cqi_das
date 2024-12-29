import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, f1_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from lime.lime_tabular import LimeTabularExplainer
import shap
import pickle

from sklearn.feature_selection import RFE

def make_train_test_split(df_features):
    seed = 42
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(df_features.drop(['target'], axis=1), df_features['target'], 
                                                        test_size=test_size, stratify=df_features['target'], random_state=seed)
    return X_train, X_test, y_train, y_test

def run_DT_grid_search(X_train, y_train, verbose=10, scoring='precision'):
    dt = DecisionTreeClassifier(random_state=42)
    params = {
        'max_depth': [2,3,5,10,20],
        'min_samples_leaf': [5,10,20,50,100],
        'max_features': [10,20,50,'sqrt',None],
        'criterion': ['gini','entropy']
    }
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=dt, 
                            param_grid=params, 
                            cv=4, n_jobs=-1, verbose=verbose, scoring = scoring)

    grid_search.fit(X_train, y_train)
    score_df = pd.DataFrame(grid_search.cv_results_)
    return score_df 

def run_RF_grid_search(X_train, y_train, verbose=10, scoring='precision'):
    rf = RandomForestClassifier(random_state=42)
    params = {
        'max_depth': [2,3,5,10,20],
        'min_samples_leaf': [5,10,20,50,100],
        'max_features': [10,20,50,'sqrt',None],
        'n_estimators': [2,5,10,50,100,500]
    }
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, 
                            param_grid=params, 
                            cv=4, n_jobs=-1, verbose=verbose, scoring = scoring)

    grid_search.fit(X_train, y_train)
    score_rf = pd.DataFrame(grid_search.cv_results_)
    return score_rf 

def save_model(model, output_file="clf.pkl"):
    with open(output_file, 'wb') as file:  
        pickle.dump(model, file)

def load_model(input_file):
    return pickle.load(open(input_file, 'rb'))

def fit_DT_model(X_train, y_train, param_criterion='gini', param_max_depth=20, param_max_features=None, param_min_samples_leaf=50):
    #using best model
    dt = DecisionTreeClassifier(criterion=param_criterion, max_depth=param_max_depth, max_features=param_max_features, 
                                min_samples_leaf=param_min_samples_leaf, random_state=42)

    # fit the model
    dt.fit(X_train, y_train)
    return dt

def fit_RF_model(X_train, y_train, max_depth=20, max_features=None, min_samples_leaf=5, n_estimators=500, random_state=42):
    #using best model
    rf = RandomForestClassifier(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, 
                                n_estimators=n_estimators, random_state=random_state)

    # fit the model
    rf.fit(X_train, y_train)
    return rf

def fit_train_RFE(model, X_train, X_test, y_train, y_test, n_features=20, verbose=0):
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=verbose)  # Select top features

    # Fit the RFE model
    rfe.fit(X_train, y_train)

    # Get the selected features
    selected_features = X_train.columns[rfe.support_]

    # Print the ranking of the features
    print("Feature ranking:", rfe.ranking_)  # Features with ranking 1 are selected
    print("Selected features:", selected_features)

    # Optionally, train the model on the selected features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    # Train and evaluate the decision tree on the reduced set of features
    model.fit(X_train_rfe, y_train)

    show_confusion_matrix(model, X_test[selected_features], y_test, '')

    score = model.score(X_test_rfe, y_test)
    print(f"Model accuracy with selected features: {score}")

    return rfe, model, selected_features

def show_confusion_matrix(model, X_test, y_test, title = ""):
    y_pred = model.predict(X_test)

    cf_matrix = confusion_matrix(y_test, y_pred)
    _, ax2 = plt.subplots(1,1, figsize = (4,3))
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted labels');
    ax2.set_ylabel('True labels'); 
    ax2.set_title('Confusion Matrix' + title); 
    ax2.xaxis.set_ticklabels(['poor quality', 'high quality']); 
    ax2.yaxis.set_ticklabels(['poor quality', 'high quality']);
    plt.tight_layout()

def show_confusion_matrix_given_ytestypred(y_test, y_pred, title = ""):
    cf_matrix = confusion_matrix(y_test, y_pred)
    _, ax2 = plt.subplots(1,1, figsize = (4,3))
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted labels');
    ax2.set_ylabel('True labels'); 
    ax2.set_title('Confusion Matrix' + title); 
    ax2.xaxis.set_ticklabels(['poor quality', 'high quality']); 
    ax2.yaxis.set_ticklabels(['poor quality', 'high quality']);
    plt.tight_layout()

def view_feature_importances(model, X_train, figsize,title=''):
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    _, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(x=feature_scores, y=feature_scores.index)
    ax.set_title("Visualize feature scores of the features" + title)
    ax.set_yticklabels(feature_scores.index)
    ax.set_xlabel("Feature importance score")
    ax.set_ylabel("Features")
    plt.show()

def view_SHAP(model, X_test):    
    # load JS visualization code to notebook
    shap.initjs()

    # Create the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("Variable Importance Plot - Global Interpretation")
    figure = plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP feature importance plot, DT, training data', fontsize=18)
    plt.show()