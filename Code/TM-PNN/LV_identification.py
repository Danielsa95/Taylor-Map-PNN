import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import NN_structure as nn

def f(t,z):
    x,y=z
    return [y+x*y , -2*x  -x*y]

# Time Parameters
Tf=4.65
dt=0.01
N=int(Tf/dt)
time=np.arange(0,Tf+dt,dt)

## Now i will consider the solution corresponting to initial condition (0.5,0.5) as the train set,
# the other three are outer,inner and fixed test, corresponding to (0.8,0.8),(0.1,0.1),(0,0).

## TRAIN SET
sol_train= solve_ivp(f, [0, Tf], [0.5,0.5] , t_eval=time)
X_train=sol_train.y
X_train=X_train.transpose()
# OUTER TEST SET
sol_outer= solve_ivp(f, [0, Tf], [0.8,0.8] , t_eval=time)
X_test_outer=sol_outer.y
X_test_outer=X_test_outer.transpose()
# INNER TEST SET
sol_inner= solve_ivp(f, [0, Tf], [0.1,0.1] , t_eval=time)
X_test_inner=sol_inner.y
X_test_inner=X_test_inner.transpose()
# FIXED POINT TEST SET
sol_fixed= solve_ivp(f, [0, Tf], [0.,0.] , t_eval=time)
X_test_fixed=sol_fixed.y
X_test_fixed=X_test_fixed.transpose()

#### plot
figure,ax=plt.subplots(1,2,figsize=(15,5))
ax[0].plot(X_train[:, 0], X_train[:, 1], 'b-', label='train data')
ax[0].plot(X_test_fixed[:, 0], X_test_fixed[:, 1], 'y*', label='test: fixed point')
ax[0].plot(X_test_outer[:, 0], X_test_outer[:, 1], 'g--', label='test: outer track')
ax[0].plot(X_test_inner[:, 0], X_test_inner[:, 1], 'r--', label='test: inner track')


ax[1].plot(time, X_train[:, 0], 'b-', label='train data')
ax[1].plot(time, X_test_fixed[:, 0], 'y--', label='test: fixed point')
ax[1].plot(time, X_test_outer[:, 0], 'g--', label='test: outer track')
ax[1].plot(time, X_test_inner[:, 0], 'r--', label='test: inner track')

for k in [0, 1]:
    handles, labels = ax[k].get_legend_handles_labels()
    ax[k].legend(handles, labels)

ax[0].set_title('Dynamics in phase space')
ax[1].set_title('Dynamics in time space')

plt.show()

#### create the three types of neural network

pnn = nn.createPNN()
num_epoch = 2000
pnn.fit(X_train[:-1], X_train[1:],epochs=num_epoch, batch_size=50, verbose=0)
print('TM-PNN is built')


mlp = nn.createMLP(2,2)
num_epoch = 2000
mlp.fit(X_train[:-1], X_train[1:],epochs=num_epoch, batch_size=50, verbose=0)
print('MLP is built')

lstm = nn.createLSTM(2,2)
num_epoch = 2000
lstm.fit(X_train[:-1].reshape((-1, 1, X_train.shape[1])), X_train[1:], epochs=num_epoch, batch_size=50, verbose=0)
print('LSTM is built')

f, ax = plt.subplots(2, 3, figsize=(15, 10))
for i in range(3):
    ax[0, i].plot(X_train[:, 0], X_train[:, 1], 'b-', label='train data')
    ax[0, i].plot(X_test_fixed[:, 0], X_test_fixed[:, 1], 'y*', label='test: fixed point')
    ax[0, i].plot(X_test_outer[:, 0], X_test_outer[:, 1], 'g-', label='test: outer track')
    ax[0, i].plot(X_test_inner[:, 0], X_test_inner[:, 1], 'r-', label='test: inner track')

    ax[1, i].plot(time, X_train[:, 0], 'b-', label='train data')
    ax[1, i].plot(time, X_test_fixed[:, 0], 'y-', label='test: fixed point')
    ax[1, i].plot(time, X_test_outer[:, 0], 'g-', label='test: outer track')
    ax[1, i].plot(time, X_test_inner[:, 0], 'r-', label='test: inner track')

for k in range(3):
    handles, labels = ax[1, k].get_legend_handles_labels()
    ax[1, k].legend(handles, labels)

for i in range(2):
    ax[i, 0].set_title('PNN')
    ax[i, 1].set_title('MLP')
    ax[i, 2].set_title('LSTM')

# then predict via different neural networks
reshapes = [False, False, True]
for i, model in enumerate([pnn, mlp, lstm]):
    X_train_predict = nn.iterative_predict(model, X_train[0], N, reshape=reshapes[i])
    X_test_outer_predict = nn.iterative_predict(model, X_test_outer[0], N, reshape=reshapes[i])
    X_test_inner_predict = nn.iterative_predict(model, X_test_inner[0], N, reshape=reshapes[i])
    X_test_fixed_predict = nn.iterative_predict(model, X_test_fixed[0], N, reshape=reshapes[i])

    # for X_predict in [X_train_predict, X_test_inner_predict]:
    for X_predict in [X_train_predict, X_test_outer_predict, X_test_inner_predict, X_test_fixed_predict]:
        ax[0, i].plot(X_predict[::15, 0], X_predict[::15, 1], 'r*')
        ax[0, i].plot(X_predict[:, 0], X_predict[:, 1], 'r-', alpha=0.4)

        ax[1, i].plot(time[1::15], X_predict[::15, 0], 'r*')
        ax[1, i].plot(time[1:], X_predict[:, 0], 'r-', alpha=0.4)

plt.show()


######  it is a study that consists in evaluating the mse of the pnn for different training epoch
norme=list()
for i in [1000,2000,3000,4000,5000]:
   rmse=np.zeros(3)
   pnn = nn.createPNN()
   model=pnn
   num_epoch = i
   pnn.fit(X_train[:-1], X_train[1:],epochs=num_epoch, batch_size=50, verbose=0)
   print('TM-PNN is built')
   X_test_outer_predict = nn.iterative_predict(model, X_test_outer[0], N, reshape=False)
   X_test_inner_predict = nn.iterative_predict(model, X_test_inner[0], N, reshape=False)
   X_test_fixed_predict = nn.iterative_predict(model, X_test_fixed[0], N, reshape=False)
   norm1=scipy.linalg.norm(X_test_outer_predict[:,0]-X_test_outer[1:,0])
   norm2=scipy.linalg.norm(X_test_inner_predict[:,0]-X_test_inner[1:,0])
   norm3=scipy.linalg.norm(X_test_fixed[1:,0]-X_test_fixed_predict[:,0])
   rmse[0]=norm1
   rmse[1]=norm2
   rmse[2]=norm3
   norme.append(rmse)

print(norme)
