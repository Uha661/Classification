import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


def Read_data(path):
    data = pd.read_csv(path,encoding='unicode_escape')
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data = data.fillna(' ')
    return data

def LabelEncoding(data):
    le=LabelEncoder()
    le.fit(data["productgroup"])
    x=le.transform(data["productgroup"])
    data["productgroup"]=x
    with open('Label.pkl', 'wb') as f:
        pickle.dump(le, f)
    return data

def SplitData(data):
    y=data["productgroup"]
    x = data['main_text'] + ',' + data['add_text'] + ',' + data['manufacturer']
    X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=.20)
    return (X_train, X_test, y_train, y_test)

def Vectorization(X_train):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    with open('vector.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    return X_train

def DecisionTree(X_train,y_train,X_test,y_test,vectorizer):
    dtree_model = DecisionTreeClassifier()
    dtree_model.fit(X_train, y_train)
    print("decisionTree Training confusion matrix:",confusion_matrix(y_train,dtree_model.predict(X_train)))
    print("decisionTree Testing confusion matrix:",confusion_matrix(y_test,dtree_model.predict(vectorizer.transform(X_test))))
    print("decisionTree Test accuracy:",accuracy_score(y_test,dtree_model.predict(vectorizer.transform(X_test))))
    with open('DecisionTree.pkl', 'wb') as f:
        pickle.dump(dtree_model, f)
    
def Svc(X_train,y_train,X_test,y_test,vectorizer):
    svm_model_linear = SVC()
    svm_model_linear.fit(X_train, y_train)
    print("SVC Training confusion matrix:",confusion_matrix(y_train,svm_model_linear.predict(X_train)))
    print("SVC Testing confusion matrix:",confusion_matrix(y_test,svm_model_linear.predict(vectorizer.transform(X_test))))
    print("SVC Test accuracy:",accuracy_score(y_test,svm_model_linear.predict(vectorizer.transform(X_test))))
    with open('SVC.pkl', 'wb') as f:
        pickle.dump(svm_model_linear, f)


def KNN(X_train,y_train,X_test,y_test,vectorizer):
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn.fit(X_train, y_train)
    print("KNN Training confusion matrix:",confusion_matrix(y_train,knn.predict(X_train)))
    print("KNN Testing confusion matrix:",confusion_matrix(y_test,knn.predict(vectorizer.transform(X_test))))
    print("KNN Test accuracy:",accuracy_score(y_test,knn.predict(vectorizer.transform(X_test))))
    with open('KNN.pkl', 'wb') as f:
        pickle.dump(knn, f)


def RandomForest(X_train,y_train,X_test,y_test,vectorizer):
    rf_model = RandomForestClassifier(random_state=121)
    rf_model.fit(X_train, y_train)
    print("RandomForest Training confusion matrix:",confusion_matrix(y_train,rf_model.predict(X_train)))
    print("RandomForest Testing confusion matrix:",confusion_matrix(y_test,rf_model.predict(vectorizer.transform(X_test))))
    print("RandomForest Test accuracy:",accuracy_score(y_test,rf_model.predict(vectorizer.transform(X_test))))
    with open('RandomForest.pkl', 'wb') as f:
        pickle.dump(rf_model, f)




def main():
    data = Read_data(r"testset_c.csv")
    data = LabelEncoding(data)
    X_train, X_test, y_train, y_test = SplitData(data)
    X_train = Vectorization(X_train)
    with open('vector.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    DecisionTree(X_train,y_train,X_test,y_test,vectorizer)
    Svc(X_train,y_train,X_test,y_test,vectorizer)
    KNN(X_train,y_train,X_test,y_test,vectorizer)
    RandomForest(X_train,y_train,X_test,y_test,vectorizer)


main()
