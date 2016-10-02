import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random

input = np.load('X_train.npy')   
labels = np.genfromtxt('../data/y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

class neural_network(object):
    def __init__(self):
        self.input = T.tensor4()
        
        #conv 1
        self.conv1_w = theano.shared(self.conv_weight_init(3, 90, 3, 3))
        self.conv1_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(90)),dtype=theano.config.floatX))
        self.conv1_conv = conv2d(input=self.input, filters=self.conv1_w, border_mode='half')
        self.conv1_out = self.maxout(self.conv1_conv + self.conv1_b.dimshuffle('x', 0, 'x', 'x'),90,10)
        
        #conv 2
        self.conv2_w = theano.shared(self.conv_weight_init(10, 90, 3, 3))
        self.conv2_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(90)),dtype=theano.config.floatX))
        self.conv2_conv = conv2d(input=self.conv1_out, filters=self.conv2_w, border_mode='half')
        self.conv2_out = self.maxout(self.conv2_conv + self.conv2_b.dimshuffle('x', 0, 'x', 'x'),90,10)
        
        #conv 3
        self.conv3_w = theano.shared(self.conv_weight_init(10, 180, 3, 3))
        self.conv3_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(180)),dtype=theano.config.floatX))
        self.conv3_conv = conv2d(input=self.conv2_out, filters=self.conv3_w, border_mode='half')
        self.conv3_mp = downsample.max_pool_2d(self.conv3_conv, ds=(2,2), ignore_border=True)
        self.conv3_out = self.maxout(self.conv3_mp + self.conv3_b.dimshuffle('x', 0, 'x', 'x'),180,20)
        
        #conv 4
        self.conv4_w = theano.shared(self.conv_weight_init(20, 180, 3, 3))
        self.conv4_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(180)),dtype=theano.config.floatX))
        self.conv4_conv = conv2d(input=self.conv3_out, filters=self.conv4_w, border_mode='half')
        self.conv4_out = self.maxout(self.conv4_conv + self.conv4_b.dimshuffle('x', 0, 'x', 'x'),180,20)
        
        #conv 5
        self.conv5_w = theano.shared(self.conv_weight_init(20, 360, 3, 3))
        self.conv5_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(360)),dtype=theano.config.floatX))
        self.conv5_conv = conv2d(input=self.conv4_out, filters=self.conv5_w, border_mode='half')
        self.conv5_mp = downsample.max_pool_2d(self.conv5_conv, ds=(2,2), ignore_border=True)
        self.conv5_out = self.maxout(self.conv5_mp + self.conv5_b.dimshuffle('x', 0, 'x', 'x'),360,40)
        
        #conv 6
        self.conv6_w = theano.shared(self.conv_weight_init(40, 360, 3, 3))
        self.conv6_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(360)),dtype=theano.config.floatX))
        self.conv6_conv = conv2d(input=self.conv5_out, filters=self.conv6_w, border_mode='half')
        self.conv6_out = self.maxout(self.conv6_conv + self.conv6_b.dimshuffle('x', 0, 'x', 'x'),360,40)
        
        #ff2
        self.ff2_w = theano.shared(self.ff_weight_init(2560,10),borrow=True)
        self.ff2_b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(10)),dtype=theano.config.floatX))
        self.ff2_out = T.nnet.sigmoid(-T.dot(self.conv6_out.flatten(2),self.ff2_w)-self.ff2_b)

        self.params = [self.conv1_w,self.conv1_b,self.conv2_w,self.conv2_b,self.conv3_w,self.conv3_b,self.conv4_w,self.conv4_b,\
            self.conv5_w,self.conv5_b,self.conv6_w,self.conv6_b,self.ff2_w,self.ff2_b]
        self.target = T.matrix()
        self.cost = -self.target*T.log(self.ff2_out)-(1-self.target)*T.log(1-self.ff2_out)
        self.cost = self.cost.mean()
        self.updates = self.adam(self.cost, self.params)
        self.propogate = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.classify = theano.function([self.input],self.ff2_out,allow_input_downcast=True)
    
    def conv_weight_init(self, input_maps, output_maps, filter_height, filter_width):
        bound = np.sqrt(6./(input_maps*filter_height*filter_width + output_maps*filter_height*filter_width))
        w = np.asarray(np.random.uniform(low=-bound,high=bound,size=(output_maps, input_maps, filter_height, filter_width)),dtype=theano.config.floatX)
        return w
    
    def ff_weight_init(self,fan_in,fan_out):
        bound = np.sqrt(1.5/(fan_in+fan_out))
        w = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(w)
        if fan_in > fan_out:
            w = u[:fan_in,:fan_out]
        else:
            w = v[:fan_in,:fan_out]
        w = w.astype(theano.config.floatX)
        return w
        
    def maxout(self,conv_out,channels_in,channels_out):
        maxouts = []
        step = channels_in//channels_out
        for i in range(channels_out):
            maxouts.append(T.max(conv_out[:,i*step:(i+1)*step,:,:],axis=1,keepdims=True))
        maxout = T.concatenate(maxouts,axis=1)
        return maxout
    
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
        y = np.concatenate((y,np.arange(10))) #make sure y includes all possible labels
        if random.random() < .5:
            X = np.fliplr(X)
            y = np.flipud(y)
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        target = target[:-10,:] #drop extra labels inserted at end
        return self.propogate(X,target)
    
    def predict(self,X):
        prediction = self.classify(X)
        return np.argmax(prediction,axis=1)

print "building neural network"
nn = neural_network()

batch_size = 100

for i in range(50000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i training error: %f \r" % (i+1, cost))
    sys.stdout.flush()
    if (i+1)%100 == 0:
        preds = []
        for j in range(0,X_test.shape[0],batch_size):
             preds.append(nn.predict(X_test[j:j+batch_size,:]))
        pred = np.concatenate(preds)
        error = 1-float(np.sum(pred==y_test))/len(pred)
        print "error at iteration %i: %.4f" % (i+1,error) 
