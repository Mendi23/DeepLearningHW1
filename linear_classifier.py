import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number of features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.
        
        
        # ====== YOUR CODE: ======
        mean = torch.zeros(n_features, n_classes)
        std = torch.ones(n_features, n_classes) * weight_std
        
        # Weights are from size (D,C)
        self.weights = torch.normal(mean=mean, std=std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x.mm(self.weights)
        
        y_pred = class_scores.argmax(dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        assert y.shape == y_pred.shape
        assert y.dim() == 1

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        num_correct = (y == y_pred).sum()
    
        acc = num_correct.item() / len(y)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])
        
        # Weights are from size (D,C)
        W = self.weights
        step_size = learn_rate
        
        
        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            
            # self.weights is the initial weights of the class
            
            
            acc = 0
            loss = 0
            count = 0
            
#             step_size = learn_rate/(epoch_idx + 1)
            
            #evaluate the model on the trainning set and update the weights.
            for X,Y in dl_train:
                
                y_pred, class_scores = self.predict(X)
                
                
                # calculate the accuracy and the loss.
                acc += self.evaluate_accuracy(Y, y_pred)
                loss += loss_fn.loss(X, Y, class_scores, y_pred) + 0.5*weight_decay * W.norm() * W.norm()
                
                count += 1
                

                # now update the weights.
                W -= step_size*(loss_fn.grad() + weight_decay * W)
#                 W /= W.norm()
                
            
            train_res.accuracy.append(acc / count)
            train_res.loss.append(loss / count)
            
#             print('Norm - ', W.norm())
            
            acc = 0
            loss = 0
            count = 0
            
            # now lets evaluate the model on the validation set
            for X,Y in dl_valid:
                
                y_pred, class_scores = self.predict(X)
                
                acc += self.evaluate_accuracy(Y, y_pred)
                loss += loss_fn.loss(X, Y, class_scores, y_pred) + weight_decay * W.norm()
                
                count += 1

            valid_res.accuracy.append(acc / count)
            valid_res.loss.append(loss / count)
            
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        # Weights are from size (D,C)
        
        print('W size - ', self.weights.size())
        w_images = self.weights[:-1,:].transpose(0,1).view(self.n_classes, img_shape[0], img_shape[1], img_shape[2])
        
        # ========================

        return w_images
