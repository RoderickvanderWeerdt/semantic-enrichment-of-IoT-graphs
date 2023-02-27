from torch import nn
import pandas as pd

from epochs_hotcold_dataset import Emb_Hotcold_Dataset, ToTensor
from torch.utils.data import DataLoader
import torch
torch.manual_seed(0)

def shuffle_dataset(dataset_fn):
    df = pd.read_csv(dataset_fn, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("shuffled_dataset.csv", index=False)
    # df.to_csv(dataset_fn, index=False)

def check_if_emb_exists(dataset_fn, i):
    df = pd.read_csv(dataset_fn, sep=",")
    if i == 0:
        i = ''
    else:
        i = str(i)
    if "emb"+i in df.keys():
        return True
    else:
        # print("couldn't find emb"+i)
        return False

def epochs_perform_predictions(epochs, dataset_fn, show_all):
    shuffle_dataset(dataset_fn)
    for i in range(0, epochs):
        if check_if_emb_exists(dataset_fn, i):
            results = epoch_perform_prediction(i, dataset_fn, show_all)
            res = str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')
            print("after", i+1, "epoch(s):", "\t"+str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ','))

def epoch_perform_prediction(epochs, dataset_fn, show_all):
    # shuffle_dataset(dataset_fn)
    training_data = Emb_Hotcold_Dataset(csv_file="shuffled_dataset.csv", train=True, transform=ToTensor(), emb_id = epochs)
    test_data = Emb_Hotcold_Dataset(csv_file="shuffled_dataset.csv", train=False, transform=ToTensor(), emb_id = epochs)
    
    batch_size = 4
    
    

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    train_tss = []
    test_tss = []

    for sample in test_dataloader:
        X = sample['embedding']
        y = sample['target_class']
        if show_all: print("Shape of X [N, C, H, W]: ", X.shape)
        if show_all: print("Shape of y: ", y.shape, y.dtype)
        break

    # exit()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if show_all: print("Using {} device".format(device))

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(100, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    model = model.float()
    if show_all: print(model)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        total = 0
        correct = 0
        for batch, sample in enumerate(dataloader):
            X = sample['embedding']
            y = sample['target_class']
            X, y = X.to(device), y.to(device)
            
            train_tss.append(y)

            # Compute prediction error
            pred = model(X.float())
            loss = loss_fn(pred.float(), y)
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # print(batch)
            # print(predicted, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                if show_all: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print("x", X[0], "y", y[0], "pred", pred[0])
                # print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))
        
        correct /= total
        if show_all: print(f"Train Accuracy: {(100*correct):>0.1f}%")
        return correct


    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for sample in dataloader:
                X = sample['embedding']
                y = sample['target_class']
                X, y = X.to(device), y.to(device)
                test_tss.append(y)
                pred = model(X.float())
                _, predicted = torch.max(pred.data, 1)
#                 print(predicted[0],y[0])
                total += y.size(0)
                correct += (predicted == y).sum().item()
        correct /= total
        if show_all: print(f"Test Accuracy: {(100*correct):>0.1f}%")
        return correct

    epochs = 20
    model = model.float()
    test_res = []
    train_res = []
    for t in range(epochs):
        if show_all: print(f"Epoch {t+1}\n-------------------------------")
        train_res.append(train(train_dataloader, model, loss_fn, optimizer))
        test_res.append(test(test_dataloader, model, loss_fn))
    if show_all: print("Done!")
    return {"train_result":train_res[-1],"test_result":test_res[-1], "targets_x": train_tss, "targets_y": test_tss}