import streamlit as st
import numpy as np
from sklearn import datasets as ds

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score as acc


st.title('Explore different classifier on different data sets.')


dataset = st.sidebar.selectbox('Choose dataset',('Iris', 'Breast Cancer', 'Wine'))
model = st.sidebar.selectbox('Choose Model', ('Logistic Regression', 'KNN', 'SVC', 'Decision Tree', 'Random Forest', 'Bagging', 'AdaBoost' , 'GradientBoost', 'Voting (Hard)', 'Voting (Soft)'))

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = ds.load_iris()
    elif name == 'Breast Cancer':
        data = ds.load_breast_cancer()
    else:
        data = ds.load_wine()

    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset)
st.write('Shape of the data in dataset:', X.shape)
st.write('No. of classes in the dataset:', len(np.unique(y)))

def get_parameters(clf):
    params = dict()
    if clf == 'KNN':
        params['n'] = st.sidebar.slider('n_neighbors',1,20)
    elif clf == 'Random Forest':
        params['n'] = st.sidebar.slider('n_estimators', 100,400)
        params['max_depth'] = st.sidebar.slider('max_depth',2,10)
    elif clf == 'Bagging':
        params['n'] = st.sidebar.slider('n_estimator', 100,400)
    elif clf == 'AdaBoost':
        params['n'] = st.sidebar.slider('n_estimator', 100,400)
        params['learning_rate'] = st.sidebar.slider('learning_rate',0.1,0.8)
    elif clf == 'GradientBoost':
        params['n'] = st.sidebar.slider('n_estimator', 100,400)
        params['max_depth'] = st.sidebar.slider('max_depth',2,10)
        params['learning_rate'] = st.sidebar.slider('learning_rate',0.1,0.8)
    
    return params

params = get_parameters(model)

def classifier_declaration(clf, params):
    classifier = None
    if clf == 'Logistic Regression':
        classifier = LogisticRegression(random_state = 0)
    elif clf == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors = params['n'])
    elif clf == 'SVC':
        classifier = SVC(probability = True, random_state = 0)
    elif clf == 'Decision Tree':
        classifier = DecisionTreeClassifier(random_state = 0)
    elif clf == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators = params['n'], max_depth = params['max_depth'])
    elif clf == 'Bagging':
        classifier = BaggingClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = params['n'], n_jobs = -1, random_state = 0)
    elif clf == 'AdaBoost':
        classifier = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), n_estimators = params['n'], learning_rate = params['learning_rate'], random_state=0)
    elif clf == 'GradientBoost':
        classifier = GradientBoostingClassifier(random_state=0, n_estimators = params['n'], learning_rate = params['learning_rate'], max_depth = params['max_depth'])
    elif clf == 'Voting (Hard)':
        log = LogisticRegression(random_state = 0)
        knn = KNeighborsClassifier()
        svc = SVC(probability = True, random_state = 0)
        classifier = VotingClassifier(estimators=[('l',log), ('k',knn), ('s',svc)], voting='hard')
    elif clf == 'Voting (Soft)':
        log = LogisticRegression(random_state = 0)
        knn = KNeighborsClassifier()
        svc = SVC(probability = True, random_state = 0)
        classifier = VotingClassifier(estimators=[('l',log), ('k',knn), ('s',svc)], voting='soft')
    
    return classifier

classifier = classifier_declaration(model, params)


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

st.write('Classifier Name:', model)
st.write('Model Accuracy:', acc(y_test, y_pred))


