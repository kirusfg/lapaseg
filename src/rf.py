from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

train_data = np.load("5.npz")["arr_0"]
X_train, y_train = train_data[:, :-1], train_data[:, -1:].squeeze()

test_data = np.load("3.npz")["arr_0"]
X_test, y_test = test_data[:, :-1], test_data[:, -1:].squeeze()

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=50,
    n_jobs=64,
    random_state=42,
    verbose=True,
)

rf = model.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
