import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if predictions.ndim == 1:
      p = np.copy(predictions) - np.max(predictions)
      e = np.exp(p)
      return e / e.sum()
    else:
      p = np.copy(predictions) - np.max(predictions, axis=1).reshape(-1, 1)    
      e = np.exp(p)
      return e / e.sum(axis=1).reshape(-1, 1)

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if np.ndim(probs) == 1:
      return -np.log(probs[target_index])

    plog = -np.log(probs)
    plog_by_idx = plog[range(probs.shape[0]), target_index.flatten()]
    loss = plog_by_idx.mean()
    
    return loss

    #raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    grad = np.zeros_like(predictions)

    if np.ndim(predictions) == 1:
      grad[target_index] = 1
      grad = (dprediction - grad)
    else:
      i = np.arange(grad.shape[0])
      grad[i, target_index.flatten()] = 1
      grad = (dprediction - grad) / dprediction.shape[0]
    
    return loss, grad


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    loss = reg_strength * (W * W).sum()
    grad = reg_strength * 2 * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    loss, dP = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T @ dP
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")
            for i in batches_indices:
              batchX = X[i]
              batchY = y[i]
              
              loss, grad = linear_softmax(batchX, self.W, batchY)
              loss1, grad1 = l2_regularization(self.W, reg)

              loss += loss1
              grad += grad1

              self.W -= grad * learning_rate
              loss_history.append(loss)

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        r = X @ self.W
        pred = softmax(r)
        y_pred = np.argmax(pred, axis=1)
        return y_pred



                
                                                          

            

                
