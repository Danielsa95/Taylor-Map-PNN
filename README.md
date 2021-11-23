# Taylor-Map-PNN

"Code" folder contains two _.py files: "weights.py" and "helper_fun.py". In the first we can find the implementation of the method used to evaluate the weights matrices;
in the second there are many helper functions, used in "weights.py" and in the some other files. 
"Code" contains also other two folders: 
1) TM_Examples, which contains all the examples considered for testing the algorithm in "weights.py" and also some consideration are done. In particular
   a comparison between the solution found using the weights of the code and the one obtained using Runge-Kutta45 (in the folder there is a file which contains the output of this algorithm) is performed.
2) TM-PNN, which contains two test files that are the solutions of Lotka-Volterra using RK45 and four other _.py files: the NN_structure.py is the most important and contain
   all the function used to implement the PNN, the LSTM and the MLP; Lotka-Votlerra.py contains the reworking of the method used in "helper_fun.py" to find the solution; 
   Fine-tuning.py is the code for the fine tuning of the pendulum; Lv_identification contains the code for the identification problem of Lotka-Volterra system, included the          comparison with the two other NN models .
