import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, window_size=100):
        self.scaler = StandardScaler()
        self.window_size = window_size

    def fit_transform(self, train_data):
        scaled_train = self.scaler.fit_transform(train_data)
        return self.to_windows(scaled_train)

    def transform(self, test_data):
        scaled_test = self.scaler.transform(test_data)
        return self.to_windows(scaled_test)

    def to_windows(self, data):
        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i : i + self.window_size])
        return np.array(windows)
