import numpy as np
from collections import Counter

# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=7, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        # If stopping condition met, return leaf node
        if depth >= self.max_depth or len(unique_labels) == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        best_feature, best_threshold, best_gain = self._best_split(X, y, n_features)
        if best_feature is None or best_gain < 0.01:
            return Node(value=self._most_common_label(y))

        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        feature_subset = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

        for feature in feature_subset:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold, best_gain

    def _information_gain(self, y, X_column, threshold):
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])

        n = len(y)
        left_weight = len(left_idx) / n
        right_weight = len(right_idx) / n

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _split(self, X_column, threshold):
        left_idx = np.where(X_column <= threshold)[0]
        right_idx = np.where(X_column > threshold)[0]
        return left_idx, right_idx

    def _entropy(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        return self._traverse_tree(x, node.left if x[node.feature] <= node.threshold else node.right)

# Random Forest Classifier
# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf  # Add this
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = int(0.8 * X.shape[0])
        idxs = np.random.choice(X.shape[0], n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
