import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation,Dense,LSTM
from tensorflow.keras.models import Sequential
import Code.weights as mn

""" In this file there is the code of the layers and the implementation of the Neural Networks used."""


"""
This layer (a subclasses of the layers.Layer main class of Tensorflow.keras) has the easy task to take 
input values (x,y) at time t_i and return a vector which contains all the component of the Kronecker power with same term
reduction, 

(x,y) ---------- > (x, y, x^2, xy, y^2, x^3, x^2 * y, x*y^2, y^3).

The other layer (the one named custom) takes as input the output of the 'filter' layer and returns 
two values : (x,y) at time t_i+1.
 
"""
class filter(Layer) :
    def __init__(self,num_units,activation="relu"):
        super(filter,self).__init__()
        self.num_units=num_units
        self.activation=Activation(activation)

    def build(self,input_shape):
        self.weight=self.add_weight(shape=[input_shape[-1],self.num_units],trainable=False)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
             initial_value=b_init(shape=(self.num_units,), dtype='float32'),
            trainable=False)

    def call(self,input):
        t0 = np.array([[[1.,0.,1.,1.,0.,1.,1.,1.,0.],[0.,1.,0.,0.,1.,0.,0.,0.,1.]]],dtype='float32')
        t1 = np.array([[[0.,0.,1.,0.,0.,1.,1.,0.,0.],[0.,0.,0.,1.,1.,0.,0.,1.,1.]]],dtype='float32')
        t2 = np.array([[[0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,1.,1.]]],dtype='float32')
        t0 = tf.constant(t0)
        t1 = tf.constant(t1)
        t2 = tf.constant(t2)
        tmp0 = tf.matmul(input,t0)
        tmp1 = tf.matmul(input,t1) + tf.constant([[1.,1.,0.,0.,0.,0.,0.,0.,0.]])
        tmp2 = tf.matmul(input,t2) + tf.constant([[1.,1.,1.,1.,1.,0.,0.,0.,0.]])

        y=tmp0 * tmp1 * tmp2
        return y
################################################################################


#################################################################################
class custom(Layer) :
    def __init__(self,num_units):
        super(custom,self).__init__()
        self.num_units=num_units

    def build(self,input_shape):
        self.weight = self.add_weight(shape=[input_shape[-1], self.num_units], trainable=True)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
            initial_value=b_init(shape=(self.num_units,), dtype='float32'),
            trainable=True)


    def call(self,input):
        y=tf.matmul(input,self.weight) + self.bias
        return y
##############################################################################

##### Using this function is possible to create the PNN
def createPNN(coeff=0,dt=0,n_dim=0,initialization=False):
    if(initialization == True):
      w0,w1,w2,w3=mn.weight(coeff,dt,n_dim)       ###  evaluation of the weight
      W_init = np.array([[w1[0][0], w1[1][0]],
                       [w1[0][1], w1[1][1]],
                       [w2[0][0], w2[1][0]],
                       [w2[0][1], w2[1][1]],
                       [w2[0][2], w2[1][2]],
                       [w3[0][0], w3[1][0]],
                       [w3[0][1], w3[1][1]],
                       [w3[0][2], w3[1][2]],
                       [w3[0][3], w3[1][3]]
                       ], dtype='float32'), \
              np.array([0,0], dtype='float32')
    else :
       W_init = np.array([[1, 0],
                         [0, 1],
                         [0, 0],
                         [0, 0],
                         [0, 0],
                         [0, 0],
                         [0, 0],
                         [0, 0],
                         [0, 0]
                       ], dtype='float32'), \
              np.array([0,0], dtype='float32')


    l1=filter(9)
    l2=custom(2)
    model=Sequential()
    model.add(l1)
    model.add(l2)
    model.predict([[0., 0.]])
    opt = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.99,
                                  beta_2=0.99999, epsilon=1e-1, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.layers[1].set_weights(W_init)      #### setting the right weight, the ones found using the TM method
    return model
#################################################################################

def pred(nstep,model,x0):
    xx=model
    x=np.zeros(nstep+1)
    y=np.zeros(nstep+1)
    x[0]=x0[0][0]
    y[0]=x0[0][1]
    tmp=x0
    for i in range(nstep):
        vet=xx.predict(tmp)
        x[i+1]=vet[0][0][0]
        y[i+1]=vet[0][0][1]
        tmp=vet

    return x,y

#############################################################################
def createMLP(inputDim, outputDim):
    model = Sequential()
    model.add(Dense(4, input_dim=inputDim, activation='sigmoid'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(outputDim, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def createLSTM(inputDim, outputDim):
    model = Sequential()
    model.add(LSTM(10, input_dim=inputDim, input_length=1))
    model.add(Dense(outputDim, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, 2))
    X = X0.reshape(-1,2)
    for i in range(N):
        if reshape:
            X = model.predict(X.reshape(1,1,2))
        else:
            X = model.predict(X)
        ans[i] = X
    return np.vstack((X0, ans[:-1]))

