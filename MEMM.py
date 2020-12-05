import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import math
from nltk.corpus import treebank
from sklearn.feature_extraction import DictVectorizer
from Functions import *
import matplotlib.pyplot as plt

class Model:
    """
    This is a POS tagger based on a Maximum Entropy Markov Model. The default
    training corpus is the treebank corpus and the hyperparameters for regression
    are tuned to this corpus.
    """


    def __init__(self, corpus = treebank, ttsplit = 0.9):
        """
        Initialises the train and test data for the model, utilising the DicVectorizer
        tool from SciKit Learn and a onhot encoding function which uses Pandas.
        """

        self.X_dict = []
        self.y = []

        for sent in corpus.tagged_sents():
            i = 0
            for (word,tag) in sent:
                    self.X_dict.append(dictfeatures(sent,word,i))
                    self.y.append(tag)
                    i += 1

        self.vectorizer = DictVectorizer(sparse = False).fit(self.X_dict)

        self.X = self.vectorizer.transform(self.X_dict).T
        self.y, self.tagset = onehot(self.y)
        self.y = self.y.T


        self.X_train = self.X[:, :int(ttsplit*self.X.shape[1])]
        self.X_test = self.X[:, int(ttsplit*self.X.shape[1]):]
        self.y_train = self.y[:, :int(ttsplit*self.y.shape[1])]
        self.y_test = self.y[:, int(ttsplit*self.y.shape[1]):]

        self.numclasses = self.y_train.shape[0]
        self.numfeatures = self.X_train.shape[0]
        self.w = np.zeros(shape = (self.numfeatures, self.numclasses))


    def evaluate(self):
        """
        Predicts the tag for words in the test data and returns a percentage accuracy
        """
        correct = 0
        for i in range(self.X_test.shape[1]):
            yhat = softmax(np.dot(self.w.T ,self.X_test[:, i]))
            if np.argmax(yhat) == np.argmax(self.y_test[:, i]):
                correct += 1
        test_accuracy = correct/self.X_test.shape[1]
        print('Test Accuracy = ' + str(test_accuracy*100) +'%')

        return None

    def train(self, batch_size = 32 , lrate = 0.6, lambd = 0.05):
        """
        Trains the weights via multinomial logistic regression and stores them
        in a class variable self.w
        """
        num_cycles = int(self.X_train.shape[1]/batch_size)
        for i in range(num_cycles):
            X = self.X_train[:, (i)*batch_size:(i+1)*batch_size]
            y = self.y_train[:, (i)*batch_size:(i+1)*batch_size]
            yhat = softmax(np.dot(self.w.T, X))
            if i % batch_size == 0:
                print('Loss for example ' + str(i) +' = ' + str(loss(y,yhat)))
            wgrad = grad(X, y, yhat, lambd, self.w )
            self.w = self.w - lrate*wgrad
        return self


    def tagger(self, sentence):
        """
        Takes an input sentence and tokenises it, extracts features then vectorises them.
        A prediction is made on each token, which is then stored in a backpointer, once the
        end of the sentence is reached this backpointer is returned as the tag sequence for
        the sentence
        """
        backpointer = []
        splitsent = [[word,index] for (index, word) in enumerate(word_tokenize(sentence))]
        for word, index in splitsent:
            X_dict = dictfeatures(splitsent, word, index)
            X = self.vectorizer.transform(X_dict).T
            yhat = softmax(np.dot(self.w.T, X))
            pred = self.tagset[np.argmax(yhat)]
            splitsent[index][1] = pred
            backpointer.append(pred)
        return backpointer

