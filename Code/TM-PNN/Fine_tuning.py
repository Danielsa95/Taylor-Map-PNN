import NN_structure as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
"""
In this code the fine-tuning of the pendulum is performed.
First initializes the tm-pnn by knowing the equations of the system (wide length L=0.30); then 
consider the solution evaluated using L=25 as training set to improve the prevision capacity of the NN
"""

g=9.81
L_t=0.3
L_r=0.25
def theory(t,z):
    x,y=z
    return [y, -(g/L_t)*x  + g/(6*L_t) * x**3]

def real(t,z):
    x,y=z
    return [y, -(g/L_r)*x  + g/(6*L_r) * x**3]

# Time Parameters
Tf=5.00
dt=0.01
N=int(Tf/dt)
time=np.arange(0,Tf+dt,dt)
X0=np.array([0.09,0.])

## Now i will consider two solutions corresponting to initial condition (0.09,0) : the first is theoric solution,
## related to the value L=0.3; the second one is related to L=0.28 and represent the real solutions.

## TRAIN SET
sol_train = solve_ivp(real, [0, Tf], [0.09,0] , t_eval=time)
X_train=sol_train.y
X_train=X_train.transpose()

sol_theory = solve_ivp(theory, [0, Tf], [0.09,0] , t_eval=time)
X_theory=sol_theory.y
X_theory=X_theory.transpose()

test_1_sol = solve_ivp(real, [0, Tf], [0.27,0] , t_eval=time)
test_1 = test_1_sol.y
test_1=test_1.transpose()

test_2_sol= solve_ivp(real, [0, Tf], [-0.17,0] , t_eval=time)
test_2 = test_2_sol.y
test_2 = test_2.transpose()

# Builting the PNN
P0=np.array([[0],[0]])
P1=np.array([0,1,-g/L_t,0]).reshape(2,2)
P2=np.array([0,0,0,0,0,0]).reshape(2,3)
P3=np.array([0,0,0,0,(g/6*L_t),0,0,0]).reshape(2,4)
coef=(P0,P1,P2,P3)

X1=np.array([-0.17,0])
X2=np.array([0.27,0])
pnn=nn.createPNN(coef,dt=0.01,n_dim=2,initialization=True)
num_epoch = 2000
pnn.fit(X_train[:-1], X_train[1:],epochs=num_epoch, batch_size=50, verbose=0)
print('TM-PNN is built')
X_predict = nn.iterative_predict(pnn, X0, N, reshape=False)
X_predict_1 = nn.iterative_predict(pnn, X2, N, reshape=False)
X_predict_2 = nn.iterative_predict(pnn, X1, N, reshape=False)


#Plot
plt.plot(time[::5],X_train[::5,0],'c-')
plt.plot(time,X_theory[:,0],color='orange')
plt.plot(time[1::5],X_predict[::5,0],'bo')
plt.legend(["Train", "Theory","prevision"], loc ="lower right")
plt.title("Train set and theoric solution")
plt.show()

plt.plot(time[::5],X_train[::5,0],'g-')
plt.plot(time[1::5],X_predict[::5,0],'g--')

plt.plot(time[::5],test_1[::5,0],'c-')     #0.27
plt.plot(time[1::5],X_predict_1[::5,0],'c--')

plt.plot(time[::5],test_2[::5,0],'m-')
plt.plot(time[1::5],X_predict_2[::5,0],'m--')

plt.title("Testing the PNN with other initial conditions")
plt.show()
