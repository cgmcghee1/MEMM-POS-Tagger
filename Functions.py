import numpy as np
import math
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def softmax(z):
    """
    Softmax function which takes an input z and produces a probability
    distribution for each of the elements. The maximum of z is removed from
    each element to help with numerical stability as the exponent can make z
    large quickly.
    """
    z -= np.max(z)
    return np.exp(z)/np.sum(np.exp(z), axis = 0)

def loss(y, yhat):
    """
    Calculates the loss for a prediction versus the actual value.
    """
    return -(1/y.shape[1])*np.sum(y*np.log(yhat))

def grad(x, y, yhat, lambd, w):
    """
    Calculates the gradient for gradient descent based on the feature input X,
    the prediction y and the actual value y. The extra term lambd * w at the end
    is to penalize larger weights of the gradient.
    """
    cost = np.subtract(y,yhat)
    grad = -(1/y.shape[1])*((np.dot(x, cost.T)) + lambd * w)
    return(grad)



def onehot(y):
    """
    Takes an input y and produces a one hot encoding for that input.
    """
    onehotframe = pd.get_dummies(pd.DataFrame(y))
    tagset = [tag[2:] for tag in list(onehotframe.columns)]
    y = onehotframe.to_numpy(dtype = 'float64')
    return y, tagset


def dictfeatures(sentence, word, index):
    """
    For any token input in a given sequence, this function returns a dictionary
    of the features for that token.
    """
    if index == 0:
        prevtag = 'START'
    else:
        prevtag = sentence[index - 1][1]

    if index == 0:
        prevword = 'START'
    else:
        prevword = sentence[index - 1][0]

    if index == len(sentence[:]) - 1:
        nextword = 'END'
    else:
        nextword = sentence[index + 1][0]

    featuredict = {'Word' : word,
                   'Prev_Tag' : prevtag,
                   'Prev_Word' : prevword,
                   'Next_Word' : nextword,
                   'Prefix 1' : word[0],
                   'Prefix 2' : word[:2],
                   'Prefix 3' : word[:3],
                   'Suffix 1' : word[-1],
                   'Suffix 2' : word[-2:],
                   'Suffix 3' : word[-3:],
                   'Intercept' : 1
                    }
    return featuredict

