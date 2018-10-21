import numpy as np
    
class LogisticLoss:
    def apply_activation(x):
        return 1/(1+np.exp(np.multiply(x,-1)))
        
    def get_loss(Y, A_L):
        m = Y.shape[1]

        cost = (-1/m)*np.sum(np.dot(Y,np.log(A_L).T)+np.dot((1-Y),np.log(1-A_L).T))

        cost = np.squeeze(cost)   
        assert(cost.shape == ())
    
        return cost
        

    def grad(Y, A_L):
        return A_L-Y

    