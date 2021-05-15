import torch 
import torch.nn as nn 
import torch.functional as F 
from torchmetrics.functional import iou 
import pytorch_lightning as pl 

from tqdm import tqdm 

class KNNShapley: 

    def __init__(self, K=10):
        self.name = "KNN Shapley"
        self.K = K 

    def __str__(self):
        return self.name 

    def _get_shapley_value(self, X_train, y_train, X_test, y_test):
        N = len(X_train)
        M = len(X_test)

        dist = torch.cdist(X_train.view(len(X_train), -1), X_test.view(len(X_test), -1))
        _, indices = torch.sort(dist, axis=0)
        y_sorted = y_train[indices]

        score = torch.zeros_like(dist)

        score[indices[N-1], range(M)] = (y_sorted[N-1] == y_test).float() / N
        for i in range(N-2, -1, -1):
            score[indices[i], range(M)] = score[indices[i+1], range(M)] + \
                                        1/self.K * ((y_sorted[i] == y_test).float() - (y_sorted[i+1] == y_test).float()) * min(self.K, i+1) / (i+1)
        return score.mean(axis=1)

    def score(self, dm, model):
        
        # x_shape = list(model(dm.train_set[0][0].unsqueeze(0)).shape)
        # x_shape[0] = 0

        # y_shape = list(dm.train_set[0][1].unsqueeze(0).shape)
        # y_shape[0] = 0

        # X_train = torch.zeros(x_shape)
        # X_test = torch.zeros(x_shape)
        # y_train = torch.zeros(y_shape).to(torch.long)
        # y_test = torch.zeros(y_shape).to(torch.long)

        N = len(dm.train_set)
        M = len(dm.test_set)
        s = torch.zeros((N, M))

        for i, (x, y) in enumerate(tqdm(dm.test_set)):
            X = model(x.unsqueeze(0))
            for j, (X_b, y_b) in enumerate(tqdm(dm.train_dataloader(), desc="Train Inference")): 
                X_hat = model(X_b) 

                diff = (X_hat - X).reshape((N, -1))
                dist_b = torch.einsum('ij, ij->i', diff, diff)
                break
            break
                # X_train = torch.cat([X_train, x.detach().to(device="cpu")])
                # y_train = torch.cat([y_train, y.to(device="cpu")])

            # for X, y in tqdm(dm.test_dataloader(), desc="Test Inference"):
            #     x = model(X)
            #     X_test = torch.cat([X_test, x.detach().to(device="cpu")])
            #     y_test = torch.cat([y_test, y.to(device="cpu")]) 
        print(dist_b.shape)

        return dist_b
        # return self._get_shapley_value(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)    