import numpy as np
    
class LogisticLoss:
    def apply(x):
        return 1/(1+np.exp(np.multiply(x,-1)))
        
    def cost(Y, Z_L):
        A_L = LogisticLoss.apply(Z_L) 
        m = Y.shape[1]

        cost = (-1/m)*(np.dot(Y,np.log(A_L).T) + ((1-Y) * np.log(1-A_L)) )

        cost = np.squeeze(cost)   
        assert(cost.shape == ())
    
        return cost
        

    def grad(Y, A_L):
        return A_L-Y

    