import pickle

def main(test_string,modelName):
    with open(modelName + '.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Label.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('vector.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    pred = model.predict(vectorizer.transform([test_string]))
    category = le.inverse_transform(pred)
    return category[0]

