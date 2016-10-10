from scipy.linalg import svd
import theano
import theano.tensor as T
import numpy as np

'''
https://github.com/sdanaipat/Theano-ZCA
'''

class ZCA(object):
    def __init__(self):
        X_in = T.matrix('X_in')
        u = T.matrix('u')
        s = T.vector('s')
        eps = T.scalar('eps')

        X_ = X_in - T.mean(X_in, 0)
        sigma = T.dot(X_.T, X_) / X_.shape[0]
        self.sigma = theano.function([X_in], sigma, allow_input_downcast=True)

        Z = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps))), u.T)
        X_zca = T.dot(X_, Z.T)
        self.compute_zca = theano.function([X_in, u, s, eps], X_zca, allow_input_downcast=True)

        self._u = None
        self._s = None

    def fit(self, X):
        cov = self.sigma(X)
        u, s, _ = svd(cov)
        self._u = u.astype(np.float32)
        self._s = s.astype(np.float32)
        del cov

    def transform(self, X, eps):
        return self.compute_zca(X, self._u, self._s, eps)

    def fit_transform(self, X, eps):
        self.fit(X)
        return self.transform(X, eps)
        
X_train = np.load('X_train.npy')
X_train_shape = X_train.shape
X_train_flattened = X_train.reshape(X_train_shape[0],np.prod(X_train_shape[1:]))

X_test = np.load('X_test.npy')
X_test_shape = X_test.shape
X_test_flattened = X_test.reshape(X_test_shape[0],np.prod(X_test_shape[1:]))

X = np.concatenate((X_train_flattened,X_test_flattened))

zca = ZCA()
output = zca.fit_transform(X,10**-5)
X_train_output = output[:X_train_shape[0]]
X_test_output = output[X_train_shape[0]:]

X_train_output = X_train_output.reshape((X_train_shape[0],X_train_shape[1],X_train_shape[2],X_train_shape[3]))
X_test_output = X_test_output.reshape((X_test_shape[0],X_test_shape[1],X_test_shape[2],X_test_shape[3]))

print "X_train shape"
print X_train_output.shape
print "X_test shape:"
print X_test_output.shape

np.save('X_train_zca', X_train_output)
np.save('X_test_zca', X_test_output)