from utils import *
import numpy as np
from cvxopt import matrix, solvers

class SoftMarginSVMQP:

    def __init__(self, C, kernel='linear', gamma=1.0, zero_threshold=1e-5):
        '''
        Additional hyperparameters allowed
        '''
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.zero_threshold = zero_threshold
        self.alphas = None  # Non zero lagrange multipliers
        self.sv_X = None    # Support vectors
        self.sv_y = None    # Labels of support vectors
        self.W = None       # Weight vector for linear kernel only
        self.b = 0          # Bias term


    def fit(self, X, y):
        
        m = X.shape[0]  # Number of examples and features

        K=self.get_K(X)
        P = matrix(np.outer(y, y) * K + 1e-6 * np.eye(m), tc='d')  # Adds regularization        
        q = matrix(-np.ones((m, 1)))
        G = matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = matrix(np.vstack((np.zeros((m, 1)), np.ones((m, 1)) * self.C)))
        A = matrix(y.reshape(1, -1).astype('double'))
        b = matrix(0.0)
        # Use CVXOPT to solve the quadratic programming problem
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract Lagrange multipliers
        alphas = np.ravel(solution['x'])

        # Support vectors have non-zero Lagrange multipliers
        sv = alphas > self.zero_threshold   
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        
        if self.kernel_type == 'linear':
            # Compute W as a weighted sum of support vectors
            self.W = np.sum((self.alphas * self.sv_y)[:, None] * self.sv_X, axis=0)
        else:
            self.W = None  # Not used in non-linear kernels

        # Compute the bias term b using support vectors
        self.b = np.mean([
            y_i - np.sum(self.alphas * self.sv_y * (self.sv_X @ X_i))
            for y_i, X_i in zip(self.sv_y, self.sv_X)
        ])

    def predict(self, X):
        '''
        Predict function for SVM.

        Args:
            X : input data. Shape : (no. of examples , flattened dimension)
        
        Output:
            predictions : Shape : (no. of examples, )
        
        '''
        if self.kernel_type == 'linear':
            decision_values = X @ self.W + self.b # w transpose x+b
        else:
            # For RBF, evaluate kernel for all support vectors and compute predictions
            kernel_values = np.exp(-self.gamma * np.linalg.norm(X[:, None] - self.sv_X, axis=2) ** 2)
            decision_values = np.dot(kernel_values, self.alphas * self.sv_y) + self.b
        return np.where(decision_values >= 0, 1, -1)
    

    def kernel_eval(self, x1, x2):
        '''
        Evaluates the kernel function between two input vectors.

        Args:
            x1, x2 : input vectors
            
        Output:
            kernel evaluation (float)
        '''
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)


    def linear(self, X):
        return np.dot(X , X.T)
    
    def rbf(self , X):
        sq_dists = np.sum(X*2, axis=1).reshape(-1, 1) + np.sum(X*2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-self.gamma * sq_dists)
    
    def get_K(self, X):
        if self.kernel_type == 'linear':
            return self.linear(X)
        else:
            return self.rbf(X)