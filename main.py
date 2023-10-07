from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Load the dataset
data_dir = 'Agricultural-crops'
data = load_files(data_dir)
images = [imread(file) for file in data.filenames]
target = data.target

# Resize images and convert to grayscale (if they are in RGB)
X = [resize(image, (64, 64)).mean(axis=2).ravel() for image in images]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create the SVM model
svm = SVC()
svm.fit(X_train, y_train)

parameters = {'C': [0.1, 1, 10, 100]}
clf = GridSearchCV(svm, parameters, cv=5)
clf.fit(X_train, y_train)

print(f"Best parameters: {clf.best_params_}")
accuracy = svm.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

