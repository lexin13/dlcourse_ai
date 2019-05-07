import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                #if i_test == 0 and i_train == 0:
                    #print(len(self.train_X[i_train]))
                    #print(self.train_X[i_train])
                    #print(len(X[i_test]))
                    #print(X[i_test])
                    #print(np.sum(np.abs(self.train_X[i_train] - X[i_test])))
                dists[i_test][i_train] = np.sum(np.abs(self.train_X[i_train] - X[i_test]))
                pass
        #print('dists shape', np.shape(dists))
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            dists[i_test] = np.sum(np.abs(self.train_X - X[i_test]), 1)
            #if i_test == 0:
            #    print(dists[i_test])
            pass
        #print('dists shape', np.shape(dists))
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        #print(np.shape(self.train_X))
        #print(np.shape(X))
        # Using float32 to to save memory - the default is float64
        # dists = np.zeros((num_test, num_train), np.float32)
        dists = np.array([np.sum(np.abs(self.train_X - x), 1) for x in X])
        #print('dists shape', np.shape(dists))
        return dists
        # TODO: Implement computing all distances with no loops!
        pass

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        #print('dists:', np.shape(dists))
        #print(dists)
        #print('k =', self.k)
        
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            near_args = np.argsort(dists[i])[:self.k]
            # print(near_args)
            near_train_y = self.train_y[near_args]
            # print(near_train_y)
            res = self.max_count_value(near_train_y)
            # print(self.train_y[np.argmin(dists[i])], np.argmin(dists[i]))
            
            pred[i] = res
            # pred[i] = self.train_y[np.argmin(dists[i])]
            
            
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred

    def max_count_value(self, arr):
        u, c = np.unique(arr, return_counts=True)
        # print(u)
        # print(c)
        return u[c==c.max()][0]

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred
