import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

input = np.load('X_train.npy')   
labels = np.genfromtxt('../data/y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

convolutional_layers = 2
feature_maps = [3,5,5]
filter_shapes = [(4,4),(4,4)]
feedforward_layers = 1
feedforward_nodes = [1000]
classes = 10
input_shapes = []

flattened = list(input.shape[2:])
input_shapes.append((flattened[0],flattened[1]))
for i in range(convolutional_layers):
    flattened[0] = flattened[0] - filter_shapes[i][0] + 1
    flattened[1] = flattened[1] - filter_shapes[i][1] + 1
    input_shapes.append((flattened[0],flattened[1]))
flattened = np.prod(flattened)
flattened *= feature_maps[-1]

class convolutional_layer(object):
    def __init__(self, input, output_maps, input_maps, filter_height, filter_width, input_height, input_width):
        self.input = input
        self.bound = np.sqrt(6./(input_maps*filter_height*filter_width + output_maps*filter_height*filter_width))
        self.w = theano.shared(np.asarray(np.random.uniform(low=-self.bound,high=self.bound,size=(output_maps, input_maps, filter_height, filter_width)),dtype=input.dtype))
        self.b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(output_maps)),dtype=input.dtype))
        self.conv_out = conv2d(input=self.input, filters=self.w)
        self.lin_output = self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.gamma = theano.shared(value = np.ones((output_maps*(input_height-filter_height+1)*(input_width-filter_width+1)), dtype=theano.config.floatX))
        self.beta = theano.shared(value = np.zeros((output_maps*(input_height-filter_height+1)*(input_width-filter_width+1)), dtype=theano.config.floatX))
        self.lin_output = self.lin_output.dimshuffle(1,2,3,0).reshape((output_maps*(input_height-filter_height+1)*(input_width-filter_width+1),self.input.shape[0])).transpose()
        self.batch_norm = T.nnet.bn.batch_normalization(self.lin_output,gamma=self.gamma,beta=self.beta,mean=T.mean(self.lin_output, axis=0),std=T.sqrt(T.var(self.lin_output, axis=0)+0.00001))
        self.batch_norm = self.batch_norm.transpose().reshape((output_maps,(input_height-filter_height+1),(input_width-filter_width+1),self.input.shape[0])).dimshuffle(3,0,1,2)
        self.output = T.nnet.sigmoid(self.batch_norm)
    def get_params(self):
        return self.w,self.b,self.gamma,self.beta

class feedforward_layer(object):
    def __init__(self,input,features,nodes):
        self.input = input
        self.bound = np.sqrt(1.5/(features+nodes))
        self.w = theano.shared(np.asarray(np.random.uniform(low=-self.bound,high=self.bound,size=(features,nodes)),dtype=theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(nodes)),dtype=theano.config.floatX))
        self.gamma = theano.shared(value = np.ones((nodes,), dtype=theano.config.floatX))
        self.beta = theano.shared(value = np.zeros((nodes,), dtype=theano.config.floatX))
        self.lin_output = -T.dot(self.input,self.w)-self.b
        self.batch_norm = T.nnet.bn.batch_normalization(self.lin_output,gamma=self.gamma,beta=self.beta,mean=T.mean(self.lin_output, axis=0),std=T.sqrt(T.var(self.lin_output, axis=0)+0.00001))
        self.output = T.nnet.sigmoid(self.batch_norm)
    def get_params(self):
        return self.w,self.b,self.gamma,self.beta

class neural_network(object):
    def __init__(self,convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes):
        self.input = T.tensor4()       
        self.convolutional_layers = []
        self.convolutional_layers.append(convolutional_layer(self.input,feature_maps[1],feature_maps[0],filter_shapes[0][0],filter_shapes[0][1],input_shapes[0][0],input_shapes[0][1]))
        for i in range(1,convolutional_layers):
            self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1],input_shapes[i][0],input_shapes[i][1]))
        self.feedforward_layers = []
        self.feedforward_layers.append(feedforward_layer(self.convolutional_layers[-1].output.flatten(2),flattened,feedforward_nodes[0]))
        for i in range(1,feedforward_layers):
            self.feedforward_layers.append(feedforward_layer(self.feedforward_layers[i-1].output,feedforward_nodes[i-1],feedforward_nodes[i]))
        self.output_layer = feedforward_layer(self.feedforward_layers[-1].output,feedforward_nodes[-1],classes)
        self.params = []
        for l in self.convolutional_layers + self.feedforward_layers:
            self.params.extend(l.get_params())
        self.params.extend(self.output_layer.get_params())
        self.target = T.matrix()
        self.output = self.output_layer.output
        self.cost = -self.target*T.log(self.output)-(1-self.target)*T.log(1-self.output)
        self.cost = self.cost.mean()
        self.updates = self.adam(self.cost,self.params)
        self.propogate = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.classify = theano.function([self.input],self.output,allow_input_downcast=True)
    
    def adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.01, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        self.i = theano.shared(np.float32(0.))
        i_t = self.i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            self.m = theano.shared(p.get_value() * 0.)
            self.v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * self.m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * self.v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((self.m, m_t))
            updates.append((self.v, v_t))
            updates.append((p, p_t))
        updates.append((self.i, i_t))
        return updates
    
    def train(self,X,y,batch_size=None):
        if batch_size:
            indices = np.random.permutation(X.shape[0])[:batch_size]
            X = X[indices,:,:,:]
            y = y[indices]
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        return self.propogate(X,target)
    
    def predict(self,X):
        prediction = self.classify(X)
        return np.argmax(prediction,axis=1)

print "building neural network"
nn = neural_network(convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes)

test_error = []
batch_size = 2000

for i in range(3000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i training error: %f \r" % (i+1, cost))
    sys.stdout.flush()
    pred = nn.predict(X_test)
    test_error.append(1-float(np.sum(pred==y_test))/len(pred))

plt.scatter(range(len(test_error)),test_error,alpha=0.5)
plt.title("Test Set Accuracy")
plt.xlabel('Iteration')
plt.ylabel('Test Set Error')
plt.show()