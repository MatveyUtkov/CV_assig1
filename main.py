import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

# Load data
DATA_DIR = "Agricultural-crops"
CATEGORIES = os.listdir(DATA_DIR)
IMG_SIZE = 64
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)

    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        img = imread(img_path)
        img_resized = resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img_resized.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# First split the data into 90% training+validation and 10% testing
X_temp, X_test, y_temp, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_temp = scaler.fit_transform(X_temp)
X_test = scaler.transform(X_test)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# SVM
svm_accuracies = []
for train_index, val_index in kf.split(X_temp):
    X_train, X_val = X_temp[train_index], X_temp[val_index]
    y_train, y_val = y_temp[train_index], y_temp[val_index]

    svm_clf = svm.SVC(kernel='linear', C=1)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    svm_accuracies.append(accuracy)
avg_svm_accuracy = np.mean(svm_accuracies)

print(f"Average SVM Accuracy: {avg_svm_accuracy * 100:.2f}%")
