import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
X_train, y_train = load_svmlight_file("a9a.txt")
X_test, y_test = load_svmlight_file("a9a.t")

# Convert to dense arrays if they are in sparse format
X_train = X_train.toarray()
X_test = X_test.toarray()

# Splitting the data to create a validation set
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Parameters and distributions to sample from
param_dist = {
    'n_estimators': [100, 300],
    'bootstrap': [True, False],
    'max_depth': [None, 20],
    'min_samples_leaf': [1, 4],
    'min_impurity_decrease': [0.0, 0.01]
}

# Randomized search on hyperparameters
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=32, cv=5, verbose=2, random_state=42, n_jobs=-1,
    return_train_score=True
)

# Find the best hyperparameters
rf_random.fit(X_train_part, y_train_part)
best_rf = rf_random.best_estimator_

# Evaluation on the test set
y_test_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Error Rate: {1 - test_accuracy:.2f}")
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Output the training and validation error for the best parameters
best_params = rf_random.best_params_
best_cv_accuracy = rf_random.best_score_
print(f"Best Parameters found: {best_params}")
print(f"Best CV Accuracy: {best_cv_accuracy:.2f}")
print(f"Best CV Error Rate: {1 - best_cv_accuracy:.2f}")

# Assuming rf_random is a fitted RandomizedSearchCV object
results = rf_random.cv_results_

# Iterate over the results to print the mean train and validation error
for i in range(len(results['params'])):
    # Extract mean training and validation scores, then calculate the errors
    mean_train_score = np.mean([results[f'split{j}_train_score'][i] for j in range(5)])
    mean_test_score = np.mean([results[f'split{j}_test_score'][i] for j in range(5)])
    mean_train_error = 1 - mean_train_score
    mean_val_error = 1 - mean_test_score

    # Print the errors
    print(f"Parameters: {results['params'][i]}")
    print(f"Mean Train Error: {mean_train_error:.2f}")
    print(f"Mean Validation Error: {mean_val_error:.2f}")