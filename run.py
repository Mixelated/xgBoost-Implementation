from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('diabetes.csv', delimiter=",")
# split data into X and y
# select from all the rows item 0 -> 8
X = dataset[:,0:8]

# select from all the rows the last item (either 1 or 0)
Y = dataset[:,8]

#split data in training and testing set
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

#made predictions for the test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))