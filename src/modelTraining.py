from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def KNN(x_train, x_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # Make predictions on the test data and evaluate the model
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def SVM(x_train, x_test, y_train, y_test):
    # Train an SVM model using the training data
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(x_train, y_train)

    # Predict labels for the test data
    accuracy = svm_model.score(x_test, y_test)

    return accuracy


def GBC(x_train, x_test, y_train, y_test):
    # Train an SVM model using the training data
    model = GradientBoostingClassifier(
        max_depth=5, n_estimators=100, learning_rate=0.1)
    model.fit(x_train, y_train)

    # Predict on testing data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def trainSVR(x_train, x_test, y_train, y_test):
    # Create SVR model
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

    # Train model on training data
    model.fit(x_train, y_train)

    # # Predict on testing data
    # y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = model.score(x_test, y_test)

    return accuracy


def bayes(x_train, x_test, y_train, y_test):
    # Create Naive Bayes model
    model = GaussianNB()

    # Train model on training data
    model.fit(x_train, y_train)

    # Predict on testing data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def randomForest(x_train, x_test, y_train, y_test):
    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42)

    # Train model on training data
    model.fit(x_train, y_train)

    # Predict on testing data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def decisionTree(x_train, x_test, y_train, y_test):
    # Create Decision Tree model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Train model on training data
    model.fit(x_train, y_train)

    # Predict on testing data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def logisticRegression(x_train, x_test, y_train, y_test):
    # Create Logistic Regression model
    model = LogisticRegression(random_state=42)

    # Train model on training data
    model.fit(x_train, y_train)

    # Predict on testing data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
