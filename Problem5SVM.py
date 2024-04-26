import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
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

# Parameters and distributions to sample from for SVM
param_dist_svm = {
    'C': [1, 10],
    'gamma': [0.1, 0.01],
    'kernel': ['rbf']
}

# Randomized search on hyperparameters for SVM
svm_random = RandomizedSearchCV(
    SVC(random_state=42),
    param_distributions=param_dist_svm,
    n_iter=4, cv=5, verbose=2, random_state=42, n_jobs=-1,
    return_train_score=True
)

# Find the best hyperparameters for SVM
svm_random.fit(X_train_part, y_train_part)
best_svm = svm_random.best_estimator_

# Evaluation on the test set for SVM
y_test_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"SVM Test Accuracy: {test_accuracy:.2f}")
print(f"SVM Test Error Rate: {1 - test_accuracy:.2f}")
print("SVM Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Output the training and validation error for the best SVM parameters
best_params_svm = svm_random.best_params_
best_cv_accuracy_svm = svm_random.best_score_
print(f"SVM Best Parameters found: {best_params_svm}")
print(f"SVM Best CV Accuracy: {best_cv_accuracy_svm:.2f}")
print(f"SVM Best CV Error Rate: {1 - best_cv_accuracy_svm:.2f}")

# Assuming svm_random is a fitted RandomizedSearchCV object
results_svm = svm_random.cv_results_

# Iterate over the results to print the mean train and validation error for SVM
for i in range(len(results_svm['params'])):
    # Extract mean training and validation scores, then calculate the errors for SVM
    mean_train_score_svm = np.mean([results_svm[f'split{j}_train_score'][i] for j in range(5)])
    mean_test_score_svm = np.mean([results_svm[f'split{j}_test_score'][i] for j in range(5)])
    mean_train_error_svm = 1 - mean_train_score_svm
    mean_val_error_svm = 1 - mean_test_score_svm
        
    # Print the errors for SVM
    print(f"SVM Parameters: {results_svm['params'][i]}")
    print(f"SVM Mean Train Error: {mean_train_error_svm:.2f}")
    print(f"SVM Mean Validation Error: {mean_val_error_svm:.2f}")