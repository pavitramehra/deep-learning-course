import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  L = [0]*num_train
  #print(num_train)
  #print(num_classes)
  dscores = np.empty(shape = (num_classes,1))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_index = y[i]
    exp_scores = np.exp(scores)
    score_sum = np.sum(exp_scores)
    log_score = np.log(score_sum)
    L[i] = -scores[correct_class_index] + log_score
    dscores = exp_scores/(score_sum)
    dscores[correct_class_index] -= 1
    sX = X[i].reshape(1,-1).T
    dscores = dscores.reshape(1,-1)
    dW += (sX.dot(dscores))/num_train
  reg_loss = reg*np.sum(W**2)
  total_loss = np.sum(L)
  loss = total_loss/num_train + reg_loss
  dW += 2*reg*(W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  dscores = np.empty(shape = (num_train,num_classes))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  score_sum = np.sum(exp_scores,axis=1)
  log_score = np.log(score_sum)
  correct_scores = scores[range(num_train),y]
  sloss = -correct_scores+log_score
  total_loss = np.sum(sloss) 
  reg_loss =  np.sum(W**2) 
  loss = total_loss/num_train + reg_loss*reg
  dscores = exp_scores/(score_sum.reshape(num_train, 1))
  dscores[range(num_train),y] -= 1
  dW = (X.T).dot(dscores)/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

