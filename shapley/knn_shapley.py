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

    def score(self, dm, model):
        N = len(dm.train_set)
        M = len(dm.test_set)
        s = torch.zeros((N, M))
        dm.shuffle = False
        
        if type(dm.train_set[0][1]) == int:
            y_train = torch.zeros(0)
        else:
            y_shape = list(dm.train_set[0][1].unsqueeze(0).shape)
            y_shape[0] = 0
            y_train = torch.zeros(y_shape)

        stored = False
        for i, (x, y) in enumerate(tqdm(dm.test_set, desc="Computing Shapley")):
            dist = torch.zeros(0)
            X = model(x.unsqueeze(0))

            for j, (X_b, y_b) in enumerate(dm.train_dataloader()):
                X_hat = model(X_b) 
                diff = (X_hat - X).reshape((X_b.shape[0], -1))
                dist_b = torch.einsum('ij, ij->i', diff, diff)
                dist = torch.cat([dist, dist_b.detach()])

                if not stored:
                    y_train = torch.cat([y_train, y_b])

            idx = torch.argsort(dist)
            ans = y_train[idx]
            s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
            cur = N - 2

            for j in range(N - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                cur -= 1

            stored = True 

        return torch.mean(s, -1) 