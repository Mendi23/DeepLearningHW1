import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236605.dataloader_utils as dataloader_utils
from . import dataloaders



class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        
        #x_train is a tenzor
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
    
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)
        
        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]

        
        neg_ind = dist_matrix.numpy().argpartition(self.k,axis=1)[:,:self.k]
       
        neg = self.y_train[neg_ind]
        
#         print(dist_matrix.numpy().partition(3,axis=1)[:,:3])
        

        
        
        y_pred = np.zeros(n_test)
        for i in range(n_test):
            
            
            neg_ind = dist_matrix[:,i].numpy().argpartition(self.k)[:self.k]
            test_neg = self.y_train[neg_ind]
            
            
            
#             test_neg = neg[i]
            (values,counts) = np.unique(test_neg,return_counts=True)
            ind = np.argmax(counts)
            
            y_pred[i] = values[ind]
            
            
#             if i < 15:
#                 print (values , ' count: ' , counts)
# #                 print (test_neg, ' dists: ', dist_matrix[i])
#                 print(y_pred[i])
            

        return torch.tensor(y_pred, dtype=torch.int64)

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        # ====== YOUR CODE: ======
        a = self.x_train.transpose(0,1)
        b = x_test.transpose(0,1)
        
        N = a.size()[1]
        M = b.size()[1]
        
        A = a.transpose(0,1).mm(a).diag().view(N,1)
        B = b.transpose(0,1).mm(b).diag().view(M,1)
        
        o_N = torch.ones(N).view(N,1)
        o_M = torch.ones(M).view(M,1)
        
        
        C1 = A.mm(o_M.transpose(0,1))
        C2 = o_N.mm(B.transpose(0,1))
        C3 = 2*a.transpose(0,1).mm(b)
        
        dists = C1 + C2 - C3
        
        
        
        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    # ====== YOUR CODE: ======
    num_correct = (y == y_pred).sum()
    
#     print(y[:30])
#     print(y_pred[:30])
    
#     print ('Correct - ', (y == y_pred).sum(), ' total - ', len(y))
    
    accuracy = num_correct.item() / len(y)
    # ========================

    return accuracy


# def Kfolds(k, data):
#     fold_size = int(len(data)/k)
    
#     perm = torch.randperm(len(dataset)).tolist()
    
#     train_perm = perm[:-k]
#     validation_perm = perm[-k:]
    
#     train_sampler = SubsetRandomSampler(train_perm)
#     validation_sampler = SubsetRandomSampler(validation_perm)
    
#     train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                            num_workers=num_workers, sampler=train_sampler)
#     valid_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                            num_workers=num_workers, sampler=validation_sampler)
    
#     # ========================

#     return train_dl, valid_dl


def implementKNN(classifier: KNNClassifier, dl_train, dl_test):
    # Get all test data to predict in one go
    x_test, y_test = dataloader_utils.flatten(dl_test)

    # Test kNN Classifier
#     knn_classifier = KNNClassifier(k=100)
    classifier.train(dl_train)
    y_pred = classifier.predict(x_test)
    
    # Calculate accuracy
    acc = accuracy(y_test, y_pred)
    
    
    return acc


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []#np.zeros(k_choices)

    for i, k in enumerate(k_choices):
        print('i - ', i)
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        fold_acc = []
        for j in range(num_folds):
            train_dl, valid_dl = dataloaders.create_train_validation_loaders(ds_train, validation_ratio=1/num_folds)
            
            
            acc = implementKNN(model, train_dl, valid_dl)
            fold_acc.append(acc)
            
        
        accuracies.append(fold_acc)
        # ========================
    
    
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]
    
    
    
    return best_k, accuracies
