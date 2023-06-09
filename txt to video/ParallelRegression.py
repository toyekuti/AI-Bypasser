# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext


def readData(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
                         (x,y)
         where x is a numpy array and y is a real value. The input file should be a 
         'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
        
         (array(1.0,2.1,3.1),4.5)
    """ 
    return spark_context.textFile(input_file)\
        	.map(lambda line: line.split(','))\
        	.map(lambda words: (words[:-1],words[-1]))\
        	.map(lambda inp: (np.array([ float(x) for x in inp[0]]),float(inp[1])))

def readBeta(input):
    """ Read a vector Î² from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read().strip().split(',')
        return np.array( [float(val) for val in str_list] )           

def writeBeta(output,beta):
    """ Write a vector Î² to a CSV file ouptut
    """
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')
    
def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad


    
def predict(x,beta):
    """ Given vector x containing features and parameter vector β, 
        return the predicted value: 

                        y = <x,β>   
    """
    return x.dot(beta)

def f(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector Î², return the square error:

                 f(Î²;x,y) =  (y - <x,Î²>)^2	
    """
    pass

def localGradient(x,y,beta):
    """ Given vector x containing features, true label y, 
        and parameter vector Î², return the gradient âˆ‡f of f:

                âˆ‡f(Î²;x,y) =  -2 * (y - <x,Î²>) * x	

        with respect to parameter vector Î².

        The return value is  âˆ‡f.
    """
    pass

def F(data,beta,lam = 0):
    """  Compute the regularized mean square error:

             F(Î²) = 1/n Î£_{(x,y) in data}    f(Î²;x,y)  + Î» ||Î² ||_2^2   
                  = 1/n Î£_{(x,y) in data} (y- <x,Î²>)^2 + Î» ||Î² ||_2^2 

         where n is the number of (x,y) pairs in RDD data. 

         Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector Î²
            - lam:  the regularization parameter Î»

         The return value is F(Î²).
    """
    pass
     
def gradient(data,beta,lam = 0):
    """ Compute the gradient  âˆ‡F of the regularized mean square error 
                F(Î²) = 1/n Î£_{(x,y) in data} f(Î²;x,y) + Î» ||Î² ||_2^2   
                     = 1/n Î£_{(x,y) in data} (y- <x,Î²>)^2 + Î» ||Î² ||_2^2   
                 
        where n is the number of (x,y) pairs in data. 

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector Î²
             - lam:  the regularization parameter Î»

        The return value is an array containing âˆ‡F.
    """
    pass

def hcoeff(data,beta1, beta2, lam = 0):
    """ Compute the coefficients a,b,c of quadratic function h, defined as :           
                       h(Î³) = F(Î²_1 + Î³Î²_2) = aÎ³^2 + bÎ³ + c
        where F is the reqularized mean square error function.

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta1: vector Î²_1
            - beta2: vector Î²_2
            - lam: the regularization parameter Î»

        The return value is a tuple containing (a,b,c).    
    """
    pass


def exactLineSearch(data,beta,g,lam = 0):
    """ Given  data, a vector x, and a direction g, return
                   Î³ = argmin_{Î³} F(data, Î²-Î³g)

        The solution is found by first computing the coefficients of the quadratic
        polynomial 
                   h(Î³) = F(data, Î²-Î³g) = aÎ³^2 + bÎ³ + c
        The return value is Î³* = -b/(2*a)

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector Î²
            - g: direction vector g
            - lam: the regularization parameter Î»

        The return value is Î³*     

    """
    pass

def test(data,beta):
    """ Compute the mean square error  

        	 MSE(Î²) =  1/n Î£_{(x,y) in data} (y- <x,Î²>)^2

        of parameter vector Î² over the dataset contained in RDD data, where n is the size of RDD data.
        
        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector Î²

        The return value is MSE(Î²).  
    """
    pass

def train_GD(data,beta_0, lam,max_iter,eps):
    """ Perform gradient descent to  minimize F given by
  
             F(Î²) = 1/n Î£_{(x,y) in data} f(Î²;x,y) + Î» ||Î² ||_2^2   

        where
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector Î²
             - lam:  is the regularization parameter Î»
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient

        The function performs gradient descent with a gain found through 
        exact line search. That is, it computes
                   
                   Î²_k+1 = Î²_k - Î³_k âˆ‡F(Î²_k) 
        	
        where the gain Î³_k is given by
        
                   Î³_k = argmin_{Î³} F(Î²_k - Î³ âˆ‡F(Î²_k))

        and terminates after max_iter iterations or when ||âˆ‡F(Î²_k)||_2<Îµ.   

        The function returns:
             -beta: the trained Î², 
             -gradNorm: the norm of the gradient at the trained Î², and
             -k: the number of iterations performed
    """
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        grad = gradient(data,beta,lam)  
        obj = F(data,beta,lam)   
        gamma = exactLineSearch(data,beta,grad,lam)
        gradNorm = np.linalg.norm(grad)
        print('k =',k,'\tt =',time()-start,'\tF(Î²_k) =',obj,'\t||âˆ‡F(Î²_k)||_2=',gradNorm,'\tÎ³_k =',gamma)
        beta = beta - gamma * grad
        k = k + 1

    return beta,gradNorm,k


def solveLin(z,K):
    """ Solve problem
           Minimize:  z^T Î²
           subject to:  ||Î²||_1 <=Îš
        
        The return value is the optimal Î²*.
    """
    pass

def exactLineSearchFW(data,beta,s):
    """ Given  data, a vector x, and a direction g, return
                   Î³' = argmin_{Î³ in [0,1]} F(data, (1-Î³)Î²+Î³s)

        The solution is found by first computing the coefficients of the quadratic
        polynomial 
                   h(Î³) = F(data, (1-Î³)Î² + Î³ s) = aÎ³^2 + bÎ³ + c

        Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: first interpolation vector Î²
            - s: second interpolation vector s

        The return value is Î³'     

    """
    pass


def train_FW(data,beta_0, K,max_iter,eps):
    """ Use the Frank-Wolfe algorithm   minimize F_0 given by
  
             F_0(Î²) = 1/n Î£_{(x,y) in data} f(Î²;x,y)    
        
        Subject to:
             ||Î²||_1 <= K

        Inputs are:
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector Î²
             - K:  the bound K
             - max_iter: maximum number of iterations
             - eps: upper bound on the convergence criterion

        The function runs the Frank-Wolfe algorithm with a step-size found through 
        exact line search. That is, it computes
                   
                   s_k =  argmin_{s:||s||_1<=K} s^T âˆ‡F_0(Î²_k) 
                   Î²_k+1 = (1-Î³_Îº)Î²_k + Î³_k s_k 
        	
        where the gain Î³_k is given by
        
                   Î³_k = argmin_{Î³ in [0,1]} F_0((1-Î³_Îº)Î²_k + Î³_Îº s_k))

        and terminates after max_iter iterations or when (Î²_k-s_k)^Tâˆ‡F(Î²_k)<Îµ.   

        The function returns:
             -beta: the trained Î², 
             -criterion: the condition (Î²_k-s_k)^T âˆ‡F(Î²_k)
             -k: the number of iterations performed
    """
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Regression.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter Î»')
    parser.add_argument('--K', type=float,default=100.00, help='L1 norm threshold')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='Îµ-tolerance. If the l2_norm gradient is smaller than Îµ, gradient descent terminates.')
    parser.add_argument('--N',type=int,default=25,help='Level of parallelism')
    parser.add_argument('--solver',default='GD',choices=['GD', 'FW'],help='GD learns Î²  via gradient descent, FW learns Î² using the Frank Wolfe algorithm')
                
    args = parser.parse_args()
    
    sc = SparkContext(appName='Parallel Regression')
    sc.setLogLevel('warn')
    
    beta = None
                            
    if args.traindata is not None:
        # Train a linear model Î² from data, and store it in beta
        print('Reading training data from',args.traindata)
        data = readData(args.traindata,sc)
        data = data.repartition(args.N).cache()
        
        x,y = data.take(1)[0]
        dim = len(x)
        
        if args.solver == 'GD':
            start = time()
            print('Gradient descent training on data from',args.traindata,'with Î» =',args.lam,', Îµ =',args.eps,', max iter = ',args.max_iter)
            beta0 = np.zeros(dim)
            beta, gradNorm, k = train_GD(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 
            print('Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps, 'Training time:', time()-start)
            print('Saving trained Î² in',args.beta)
            writeBeta(args.beta,beta)
        
        else:
            start = time()
            print('Frank-Wolfe training on data from',args.traindata,'with K =',args.K,', Îµ =',args.eps,', max iter = ',args.max_iter)
            beta0 = np.zeros(dim)
            beta, criterion, k = train_FW(data,beta_0=beta0,K=args.K,max_iter=args.max_iter,eps=args.eps) 
            print('Algorithm ran for',k,'iterations. Converged:',criterion<args.eps, 'Training time:', time()-start)
            print('Saving trained Î² in',args.beta)
            writeBeta(args.beta,beta)
     
    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print('Reading test data from',args.testdata)
        data = readData(args.testdata,sc)
        data = data.repartition(args.N).cache()
        
        print('Reading Î² from',args.beta)
        beta = readBeta(args.beta)

        print('Computing MSE on data',args.testdata)
        MSE = test(data,beta)
        print('MSE is:', MSE)

import numpy as np
x = np.array([np.cos(t) for t in range(-5,5)])
beta = np.array([np.sin(t) for t in range(-5,5)])

print(predict(x,beta))