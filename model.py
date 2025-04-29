from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.array([[0.5], [14.0], [15.0], [28.0], [11.0], [8.0], [3.0], [-4.0], [6.0], [13.0], [21.0]])
y_train = np.array([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

def train_and_predict(value=10.0):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(np.array([[value]]))
    return preds, model

def get_accuracy(model):
    return model.score(X_train, y_train)