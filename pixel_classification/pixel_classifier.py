'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import math

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    
    self.theta = np.array([0.36599892,0.3245804, 0.30942068])
    self.mu = np.array([[0.75250609,0.34808562,0.34891229],[0.3506091, 0.73551489, 0.32949353],[0.34735903, 0.33111351, 0.73526495]])
    self.sigma = np.array([[0.03705927 ,0.06196869, 0.06202255],[0.05573463, 0.03478593, 0.05602188],[0.05453762, 0.05683331, 0.03574061]])

  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''

    # YOUR CODE HERE
    theta = self.theta
    mu = self.mu
    sigma = self.sigma
    y = np.zeros((X.shape[0],1))
    p = np.ones((X.shape[0],3))
    
    for i in range(3):#number of class
        for j in range(3):#r,g,b 
            for k in range(X.shape[0]):
                dom = 1/math.sqrt(2*sigma[i,j]*sigma[i,j]*math.pi)
                exp = (-1)*(X[k,j]-mu[i,j])**2/(2*sigma[i,j]*sigma[i,j])
                p[k,i] = p[k,i]*dom*math.exp(exp)
        p[:,i] = p[:,i]*theta[i]
    
    y = np.argmax(p,axis = 1)
    y = y+1
    return y