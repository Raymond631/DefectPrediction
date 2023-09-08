import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]
        self.priors = np.zeros(self.num_classes)
        self.means = np.zeros((self.num_classes, self.num_features))
        self.variances = np.zeros((self.num_classes, self.num_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / X.shape[0]
            self.means[i] = X_c.mean(axis=0)
            self.variances[i] = X_c.var(axis=0)

    def predict(self, X):
        print('数据')
        posteriors = np.zeros((X.shape[0], self.num_classes))

        for i in range(self.num_classes):
            prior = np.log(self.priors[i])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.variances[i]))
            exponent = -0.5 * np.sum(((X - self.means[i]) ** 2) / self.variances[i], axis=1)
            posteriors[:, i] = prior + likelihood + exponent

        return self.classes[np.argmax(posteriors, axis=1)]

