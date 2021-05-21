from models.resnet import ResNet 
from shapley.dshap import DShap 
from utils.dataset import FashionMNISTDataModule
from shapley.knn_shapley import KNNShapley


def main():
    dm = FashionMNISTDataModule() 
    dm.setup() 

    model = ResNet()

    measure = KNNShapley()

    dshap = DShap(dm, model, measure=measure)
    scores = dshap.run() 
    return scores

if __name__ == "__main__":
    scores = main()
    print(scores)
