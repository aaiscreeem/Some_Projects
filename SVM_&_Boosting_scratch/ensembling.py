from base_ensemble import *
from utils import *
import random
import numpy as np
import multiprocessing
from functools import partial

def most_common_value(y):
    count_dict = {}
    for val in y:
        if val in count_dict:
            count_dict[val] += 1
        else:
            count_dict[val] = 1

    most_common = None
    max_count = -1
    
    for key, count in count_dict.items():
        if count > max_count:
            most_common = key
            max_count = count
    
    return most_common
    
    
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, n_samples=None):
        self.feature = feature
        self.threshold = threshold  
        self.left = left
        self.right = right
        self.value = value  
        self.n_samples = n_samples  
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, min_gain=0, criterion="gini", _class_weight=1.0, k_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.criterion = criterion  # "gini" or "entropy"
        self.root = None
        self._class_weight = _class_weight  # Weight of y=1 relative to y=0
        self.k_features = k_features
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            value = most_common_value(y)
            return Node(value=value, n_samples=n_samples)
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # Stop if no split is good enough
        if best_gain < self.min_gain:
            value = most_common_value(y)
            return Node(value=value, n_samples=n_samples)
        
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        
        # Recurse
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right, n_samples=n_samples)
        
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None

        n_feats = X.shape[1]  # total number of features

        k = self.k_features 
        feat_indices = random.sample(range(n_feats), k)

        for feat_idx in feat_indices:
            X_column = X[:, feat_idx]
            thresholds = None

            sorted_idx = np.argsort(X_column)
            X_column_sorted, y_sorted = X_column[sorted_idx], y[sorted_idx]

            thresholds = np.unique((X_column_sorted[:-1] + X_column_sorted[1:]) / 2)

            # Vectorized gain calculation
            gains = np.array([self._information_gain(y, X_column, thr) for thr in thresholds])

            # Find the threshold with the maximum gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            # Update the best split if a better gain is found
            if max_gain > best_gain:
                best_gain = max_gain
                split_idx = feat_idx
                split_threshold = thresholds[max_gain_idx]

        return split_idx, split_threshold, best_gain

    
    def _information_gain(self, y, X_column, threshold):
        if self.criterion == "gini":
            parent_impurity = self._gini(y)
        else:
            parent_impurity = self._entropy(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs) / n, len(right_idxs) / n
        
        # Weighted average of the child impurities
        if self.criterion == "gini":
            return parent_impurity - (n_l * self._gini(y[left_idxs]) + n_r * self._gini(y[right_idxs]))
        else:
            return parent_impurity - (n_l * self._entropy(y[left_idxs]) + n_r * self._entropy(y[right_idxs]))
        
    def _split(self, X_column, split_thresh):
        left_idx = np.argwhere(X_column <= split_thresh).flatten()
        right_idx = np.argwhere(X_column > split_thresh).flatten()

        return left_idx, right_idx

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)

        # Ensure we have both class labels (y=0 and y=1) in ps
        if len(ps) > 1:
            # Apply class weight to y=1 if it exists
            ps_weighted = np.copy(ps)
            ps_weighted[1] *= self._class_weight
        else:
            # If there's only one class, no need to weight
            ps_weighted = ps

        return -np.sum([p * np.log2(p) for p in ps_weighted if p > 0])

    
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)

        if len(ps) > 1:
            # Apply class weight to y=1 if it exists
            ps_weighted = np.copy(ps)
            ps_weighted[1] *= self._class_weight
        else:
            # If there's only one class, no need to weight
            ps_weighted = ps

        return 1 - np.sum([p ** 2 for p in ps_weighted])

        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, X, node):
        if node.is_leaf_node():
            return node.value
        
        feature_value = X[node.feature]
        

        if feature_value <= node.threshold:
            return self._traverse_tree(X, node.left)
        else:
            return self._traverse_tree(X, node.right)


class RandomForestClassifier(BaseEnsembler):
    def __init__(self, num_trees=20, bootstrap_fraction=0.3, min_samples_split=7, max_depth=20, min_gain=0.2, criterion="gini", class_weight=1.0, k_features=None, weighted_forest=True):
        super().__init__(num_trees)
        # for forest
        self.num_trees = num_trees
        self.bootstrap_fraction = bootstrap_fraction
        self.k_features = k_features
        self.trees = []
        self.weighted_forest = weighted_forest
        self.tree_errors= []
        self.tree_weights = []
        # for trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.criterion = criterion
        self.class_weight = class_weight

    def _train_single_tree(self, X, y,_):
        # Bootstrap sampling (sample with replacement)
        n_samples = X.shape[0]
        sample_indices = np.random.choice(n_samples, int(n_samples * self.bootstrap_fraction), replace=True)
        X_sample, y_sample = X[sample_indices], y[sample_indices]

        # Create and fit a new tree
        tree = DecisionTree(
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            min_gain=self.min_gain,
            criterion=self.criterion,
            _class_weight=self.class_weight,
            k_features=self.k_features
        )
        tree.fit(X_sample, y_sample)
        
        error=0
        if self.weighted_forest:
            # Predict on the full dataset
            y_pred = tree.predict(X)
            error = np.mean(y_pred != y)  # Calculate the error rate

        return tree, error

    def fit(self, X, y):
        n_features = X.shape[1]
        y=np.where( y==-1,0,y)
        if self.k_features is None:
            self.k_features = int(np.sqrt(n_features))
        else:
            self.k_features = int(min(self.k_features, np.sqrt(n_features)))

        # Use multiprocessing to fit trees in parallel
        num_processes = min(20, self.num_trees)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(partial(self._train_single_tree, X, y), range(self.num_trees))

        # Unzip the results into trees and their corresponding errors
        self.trees, self.tree_errors = zip(*results)  # This maintains the order of trees and errors

        if self.weighted_forest:
            # Calculate the weights based on the errors
            adjusted_errors = [max(1e-9,error) for error in self.tree_errors]
            self.tree_weights = [1 / error for error in adjusted_errors]
            self.tree_weights /= np.sum(self.tree_weights)  # Normalize weights
        else:
            self.tree_weights = np.ones(self.num_trees) / self.num_trees  # Uniform weights if not weighted

    
    def predict(self, X):
        # Get predictions from each tree, shape will be (num_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Reshape tree_weights to (num_trees, 1) to align with tree_preds for broadcasting
        weighted_preds = (self.tree_weights[:, np.newaxis] * tree_preds).sum(axis=0)

        # Use threshold of 0.5 to determine final predictions
        final_preds = np.where(weighted_preds >= 0.5, 1, -1)
        return final_preds

    
    


class DecisionStump:
    def __init__(self, w):  # w is the weight vector    
        self.w = w
        self.best_feat = None
        self.best_threshold = None
        self.best_cost = 1.0   
        self.reverse = False  # False if xj >= threshold predicts 1, True means it predicts -1

    def fit(self, X, y):
        n_features, n_instances = X.shape[1], X.shape[0]

        for feat_idx in range(n_features):
            sorted_idx = np.argsort(X[:, feat_idx])[::-1]
            X_sorted, y_sorted, w_sorted = X[sorted_idx, feat_idx], y[sorted_idx], self.w[sorted_idx]

            is_misclassified = np.zeros(n_instances, dtype=bool)

        # Mark misclassified samples
            for i in range(n_instances):
                if (X_sorted[i] == X_sorted[0] and y_sorted[i] == -1):
                    is_misclassified[i] = True
                elif y_sorted[i] * (X_sorted[i] - X_sorted[0]) < 0:
                    is_misclassified[i] = True

        # Calculate cost as the sum of weights for misclassified samples
            cost = np.sum(w_sorted[is_misclassified])    
            
            # Update best values if the initial cost is lower
            if cost < self.best_cost:
                self.best_feat = feat_idx
                self.best_threshold = X_sorted[0]
                self.best_cost = cost
                self.reverse = False  # Maintain reverse attribute as False
            
            # Check if 1 - cost is better than best cost
            if 1 - cost < self.best_cost:
                self.best_feat = feat_idx
                self.best_threshold = X_sorted[0]
                self.best_cost = 1 - cost
                self.reverse = True  # Update reverse attribute

            # Incrementally update the cost as we move through X_sorted
            for k in range(1, n_instances):
                if is_misclassified[k - 1]:
                    if ((X_sorted[k - 1] == X_sorted[k] and y_sorted[k - 1] == 1) or 
                        (y_sorted[k - 1] * (X_sorted[k - 1] - X_sorted[k])) > 0):
                        cost -= w_sorted[k - 1]
                        is_misclassified[k - 1] = False
                else:
                    if ((X_sorted[k - 1] == X_sorted[k] and y_sorted[k - 1] == -1) or 
                        (y_sorted[k - 1] * (X_sorted[k - 1] - X_sorted[k])) < 0):
                        cost += w_sorted[k - 1]
                        is_misclassified[k - 1] = True

                if not is_misclassified[k] and y_sorted[k] == -1:
                    cost += w_sorted[k]
                    is_misclassified[k] = True
                elif is_misclassified[k] and y_sorted[k] == 1:
                    cost -= w_sorted[k]
                    is_misclassified[k] = False
                    
                # Update best values if the current cost is better
                if cost < self.best_cost:
                    self.best_feat = feat_idx
                    self.best_threshold = X_sorted[k]
                    self.best_cost = cost
                    self.reverse = False  

                # Check for the reverse condition again
                if 1 - cost < self.best_cost:
                    self.best_feat = feat_idx
                    self.best_threshold = X_sorted[k]
                    self.best_cost = 1 - cost
                    self.reverse = True  


        return self


    def predict(self, X):
        """
        Predicts 1 if the value of the best feature for each sample in X is >= best_threshold,
        and -1 otherwise.
        """
        if self.reverse:
            return np.where(X[:, self.best_feat] < self.best_threshold, 1, -1)
        return np.where(X[:, self.best_feat] >= self.best_threshold, 1, -1)
    
    def cost(self):
        return self.best_cost


class AdaBoostClassifier(BaseEnsembler):

    def __init__(self, num_trees = 10):
        super().__init__(num_trees)
        self.ensemble=[]
        self.weight=[]
        

    def fit(self, X, y):
        '''
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes
        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
            y : output data. Shape : (no, of examples, )

        Ouput:

            None
        '''
        m= X.shape[0]
        # initialise weight vector of length m to 1/m
        w = np.ones(m)/m
        for _ in range(self.num_trees):
            tree = DecisionStump(w)
            tree.fit(X, y)
            y_hat = tree.predict(X)
            e = tree.cost()
            if e <= 0:
                self.ensemble.append(tree)
                self.weight.append(1e9)
                break
            
            self.ensemble.append(tree)
            alpha = np.log((1 - e) / e) / 2
            self.weight.append(alpha)
            w *= np.exp(alpha * (y != y_hat) - alpha * (y == y_hat))
            
            # clip weights to avoid extreme values before normalization
            w = np.clip(w, 1e-10, None)
            w_sum = np.sum(w)
            
            if w_sum == 0:
                w = np.ones(m) / m  # Reset if normalization fails
            else:
                w /= w_sum  # Normalize weights to sum to 1
            
            
                
            
        
        
        

    def predict(self, X):
        '''
        TODO
        Function signature SHOULD NOT BE CHANGED. Please maintain the output format and output sizes

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        Ouput:
            predictions : Shape : (no. of examples, )
        '''
        predictions = sum(tree.predict(X) * alpha for tree, alpha in zip(self.ensemble, self.weight))
        return np.where(predictions >= 0, 1, -1)

    
 

    





