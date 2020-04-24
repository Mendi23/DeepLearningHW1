import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        C = x_scores.shape[1]
        l = list(range(len(y)))
        true_scores = x_scores[l,y].view(len(l),1)
        
        
        o = torch.ones(C).view(1,C)

        
        true_Scores_mat = true_scores.mm(o)
        
        #Delta matrix
        D = torch.ones(x_scores.size()) * self.delta
        
        
        # matrix pf size N,C where every item (i,j) is the dist from  sample i to line j + d - 
        # the dist from sample i to the line yi
        M = x_scores + D - true_Scores_mat
        M = M.float()
        
        
        
        #aapply max element wise
        O = torch.zeros(M.size()).float()
        L = torch.max(M,O)
        
        
        
        # by changing a bit the formula for L_i we can pull out delta
        # L_i(W) =  ( Sigma_{j \in C} max(0,m_{i,j}) ) - delta
        loss = L.sum() / len(y) - self.delta
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        # M is a matrix containig for every index (i,j) the score m(i,j)
        self.M = M
#         M[l,y] -= self.delta we need to redo this line later
        self.l = l
        self.y = y
        
        self.X = x
        
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        G1 = (self.M > 0).long()
        #we want only the indexes in wich j != y_i
        
        
        
        
#         G1[self.l,self.y] = 0
        
        # Given A - 2 dim tensor, A.sum(dim=1) will return an array with the sum of esch line
        G2_elements = G1.sum(dim=1) 
        

        
        indeces = torch.zeros(G1.size()).long()
        indeces[self.l,self.y] = 1
        
        # we will multiply a matrix containing 1 in every index (i,y_i) and multiply it by a 
        # diagonal matrix so that every index like that will contain the wanted value
        diag = torch.diag(G2_elements)
        
        
        
        G2 = diag.mm(indeces)
        
        M = G1 - G2
        
        
        N = self.X.size()[0]
        
        grad = self.X.transpose(0,1).mm(M.float()) / N
        
        
        # ========================

        return grad
