from matplotlib.figure import DrawEvent
from matplotlib.pyplot import ylim
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on mini batches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a mini batch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print('W',W.shape)
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #print("X",X.shape)
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores.shape)
    correct_class_score = scores[y[i]]
    count = 0.0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]         #for incorrect term
        #print(dW[:,j].shape)
        dW[:,y[i]] -=X[i]       #for correct term
        #dW[:,[y[i]]] -=X[i]
        #print("Xi",X[i].shape)
        #print(dW[:,j].shape)
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train  
  dW/=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += 2*reg*W    #adding the regualarization term to gradient
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  total_score = X.dot(W)     #total score
  num_rows, num_columns = total_score.shape
  score_correct_labels = total_score[range(num_rows),y]
  score_correct_labels = score_correct_labels.reshape(num_rows,-1)
  #print('loss_term_correct', score_correct_labels.shape)
  margin = np.maximum(0,total_score-score_correct_labels+1)
  margin[range(num_rows),y] = 0;
  loss = np.sum(margin)
  loss /= num_rows
  loss += reg*(np.sum(W*W))
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin_count = np.zeros(margin.shape)    
  margin_count[margin>0] = 1
  margin_count[range(num_rows),y] -=np.sum(margin_count, axis = 1)
  dW = X.T.dot(margin_count)/num_rows + 2*reg*W
  #print(margin_count.shape)
  #print('margin',margin_count)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
