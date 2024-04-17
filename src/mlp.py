import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load training data
data_train = np.load("5.npz")["arr_0"]
X_train, y_train = data_train[:, :-1], data_train[:, -1:].squeeze()

# Load testing data
data_test = np.load("3.npz")["arr_0"]
X_test, y_test = data_test[:, :-1], data_test[:, -1:].squeeze()

# Initialize the MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # Example: one hidden layer with 100 neurons
    activation="relu",  # Activation function for the neurons
    solver="adam",  # The solver for weight optimization.
    alpha=0.0001,  # L2 penalty (regularization term) parameter.
    batch_size="auto",  # Size of minibatches for stochastic optimizers
    learning_rate="constant",  # Learning rate schedule for weight updates
    learning_rate_init=0.001,  # The initial learning rate
    max_iter=200,  # Maximum number of iterations
    shuffle=True,  # Whether to shuffle samples in each iteration
    random_state=42,  # Random state for reproducibility
    verbose=True,  # Whether to print progress messages to stdout
    early_stopping=False,  # Whether to use early stopping to terminate training when validation score is not improving
)

# Fit the model on training data
mlp.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = mlp.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
