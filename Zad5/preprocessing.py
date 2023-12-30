from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_dataset():
    digits = load_digits()
    images = digits.images
    labels = digits.target
    flattened_images = images.reshape((images.shape[0], -1))
    normalized_images = flattened_images / 16.0
    X_train, X_test, y_train, y_test = train_test_split(normalized_images, labels, test_size=0.2, random_state=42)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    return X_train, X_test, y_train_one_hot, y_test_one_hot

