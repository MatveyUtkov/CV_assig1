import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold

# Constants
IMG_SIZE = 32

# Load the dataset
data_dir = 'Agricultural-crops'
categories = os.listdir(data_dir)


def load_dataset(data_dir):
    categories = os.listdir(data_dir)
    X, y = [], []

    for label, category in enumerate(categories):
        for img_file in os.listdir(os.path.join(data_dir, category)):
            img_path = os.path.join(data_dir, category, img_file)
            image = cv2.imread(img_path)
            image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image_resized.flatten())
            y.append(label)

    X = np.array(X) / 255.0  # Normalize
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias dimension
    y = np.array(y)

    return X, y


X, y = load_dataset('Agricultural-crops')
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


def svm_loss(W, X, y, reg):
    scores = X.dot(W)
    correct_scores = scores[np.arange(X.shape[0]), y]
    margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1)
    margins[np.arange(X.shape[0]), y] = 0
    loss = margins.sum() / X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)

    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(X.shape[0]), y] = -row_sum.T
    dW = X.T.dot(binary) / X.shape[0]

    return loss, dW


def softmax_loss(W, X, y, reg):
    scores = X.dot(W)
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    softmax_output = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1, keepdims=True)
    loss = -np.sum(np.log(softmax_output[np.arange(X.shape[0]), y])) / X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)

    softmax_output[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(softmax_output) / X.shape[0]

    return loss, dW


# Training with SGD
def train(X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, loss_function='svm'):
    num_classes = np.max(y) + 1
    num_features = X.shape[1]
    W = 0.001 * np.random.randn(num_features, num_classes)

    for it in range(num_iters):
        indices = np.random.choice(X.shape[0], batch_size, replace=True)
        X_batch = X[indices]
        y_batch = y[indices]

        if loss_function == 'svm':
            loss, dW = svm_loss(W, X_batch, y_batch, reg)
        elif loss_function == 'softmax':
            loss, dW = softmax_loss(W, X_batch, y_batch, reg)
        else:
            raise ValueError('Invalid loss_function "%s"' % loss_function)

        W -= learning_rate * dW

    return W


def train_and_evaluate(X_train_val, y_train_val, num_iters=100):
    kf = KFold(n_splits=5)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        print(f"Fold {fold + 1} ")

        # SVM Training and Evaluation
        W_svm = train(X_train, y_train, loss_function='svm', num_iters=num_iters)
        train_accuracy_svm = np.mean(y_train == np.argmax(X_train.dot(W_svm), axis=1))
        val_accuracy_svm = np.mean(y_val == np.argmax(X_val.dot(W_svm), axis=1))
        print(f"SVM - Training Accuracy: {train_accuracy_svm:.2f}")
        print(f"SVM - Validation Accuracy: {val_accuracy_svm:.2f}")

        # Softmax Training and Evaluation
        W_softmax = train(X_train, y_train, loss_function='softmax', num_iters=num_iters)
        train_accuracy_softmax = np.mean(y_train == np.argmax(X_train.dot(W_softmax), axis=1))
        val_accuracy_softmax = np.mean(y_val == np.argmax(X_val.dot(W_softmax), axis=1))
        print(f"Softmax - Training Accuracy: {train_accuracy_softmax:.2f}")
        print(f"Softmax - Validation Accuracy: {val_accuracy_softmax:.2f}")

# Train and evaluate
train_and_evaluate(X_train_val, y_train_val)