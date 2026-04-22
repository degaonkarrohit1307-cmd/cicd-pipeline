from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.8, "Accuracy too low!"

def test_model_predicts():
    data = load_iris()
    X, y = data.data, data.target
    model = DecisionTreeClassifier()
    model.fit(X, y)
    prediction = model.predict([X[0]])
    assert prediction[0] in [0, 1, 2]

def test_data_loads():
    data = load_iris()
    assert len(data.data) == 150
