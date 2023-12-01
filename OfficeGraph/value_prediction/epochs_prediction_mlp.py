from torch import nn
import pandas as pd

from epochs_prediction_dataset import Emb_Target_Dataset, ToTensor
from torch.utils.data import DataLoader
import torch
import numpy as np
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
    # for i in range(0, epochs):
    for i in [0]:
        if check_if_emb_exists(dataset_fn, i):
            results = epoch_perform_prediction(i, dataset_fn, show_all)
            res = str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')
            print("after", i+1, "epoch(s):", "\t"+str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',') +"\t"+ str(results["mape_result"]).replace('.', ',')+"\t"+str(results["base_mae"]).replace('.', ',') +"\t"+ str(results["base_mape"]).replace('.', ','))

def epoch_perform_prediction(epochs, dataset_fn, show_all):
    # shuffle_dataset(dataset_fn)
    training_data = Emb_Target_Dataset(csv_file="shuffled_dataset.csv", train=True, transform=ToTensor(), emb_id = epochs)
    test_data = Emb_Target_Dataset(csv_file="shuffled_dataset.csv", train=False, transform=ToTensor(), emb_id = epochs)
    
    batch_size = 4
    
    

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # train_tss = []
    # test_tss = []

    for sample in test_dataloader:
        X = sample['embedding']
        y = sample['target']
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
                nn.Linear(512, 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    model = model.float()
    if show_all: print(model)


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        mae_list = []
        for batch, sample in enumerate(dataloader):
            X = sample['embedding']
            y = sample['target']
            X, y = X.to(device), y.to(device)
            
            # train_tss.append(y)

            # Compute prediction error
            pred = model(X.float())
            # mae_list.append(calc_MAE(pred.float(), y.float()))
            mae_list.append(calc_MAE(dataloader.dataset.denormalize_single(pred.float()), dataloader.dataset.denormalize_single(y.float())))

            loss = loss_fn(pred.float(), y.float())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                # if show_all: print(f"loss: {loss:>7f}  [running MAE: {np.mean(np.array(mae_list)):>5f}]")
        mae = np.mean(np.array(mae_list))
        # mae_denormalized = dataloader.dataset.denormalize_single(mae)
        mae_denormalized = mae
        if show_all: print(f"Train MAE: {mae_denormalized:>5f}")
        return mae_denormalized


    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        mae_list = []
        mape_list = []
        y_s = torch.empty((0), dtype=torch.float32)
        p_s = torch.empty((0), dtype=torch.float32)
        with torch.no_grad():
            for sample in dataloader:
                X = sample['embedding']
                y = sample['target']
                X, y = X.to(device), y.to(device)
                # test_tss.append(y)
                pred = model(X.float())
                # for i, j in zip(dataloader.dataset.denormalize_single(pred.float()), dataloader.dataset.denormalize_single(y.float())):
                #     print(i, j)
                # mape_list.append(calc_MAPE(pred.float(), y.float()))
                y = dataloader.dataset.denormalize_single(y.float())
                pred = dataloader.dataset.denormalize_single(pred.float())
                mae_list.append(calc_MAE(pred, y))
                # mape_list.append(calc_MAPE(pred, y))
                p_s = torch.cat((p_s, torch.flatten(pred)), 0)
                y_s = torch.cat((y_s, y), 0)

        mae = np.mean(np.array(mae_list))
        # mape = np.mean(np.array(mape_list))
        mape = calc_MAPE_norm(dataloader, p_s, y_s, True)
        # mape = calc_MAPE(p_s, y_s, True)
        mae_denormalized = mae
        # mae_denormalized = dataloader.dataset.denormalize_single(mae)
        baseline_pred = torch.mean(y_s)
        base_mae = calc_MAE(baseline_pred, y_s, True)
        # base_mape = calc_MAPE(baseline_pred, y_s, True)
        base_mape = calc_MAPE_norm(dataloader, baseline_pred, y_s, True)
        if show_all: print(f"Test MAE: {mae_denormalized:>5f}, MAPE: {mape:>5f}")
        # return mape
        return mae_denormalized, mape, base_mae, base_mape
        ####                                        #####

    mlp_epochs = 40 #### HARD CODED
    model = model.float()
    train_mae = []
    test_mae = []
    test_mape = []
    for t in range(mlp_epochs):
        if show_all: print(f"MLP Epoch {t+1}\n-------------------------------")
        train_mae.append(train(train_dataloader, model, loss_fn, optimizer))
        mae, mape, base_mae, base_mape = test(test_dataloader, model, loss_fn)
        test_mae.append(mae)
        test_mape.append(mape)
    if show_all: print("Done!")
    return {"train_result":train_mae[-1],"test_result":test_mae[-1], "mape_result":test_mape[-1], "base_mae": base_mae, "base_mape": base_mape}#, "targets_x": train_tss, "targets_y": test_tss}

def calc_MAE(pred, y, pre_flattened=False):
    if not pre_flattened:
        pred = torch.flatten(pred)
    # pred = torch.flatten(pred)
    # mae = ((abs(pred[mask] - data.y[mask])).sum()) / (len(pred[mask])*1.0)
    residuals = abs(y - pred)
    # print(residuals)
    # print(torch.flatten(pred), y)
    residuals_sum = residuals.sum()
    n = len(y)
    mae = residuals_sum / n
    # print(mae.item())
    return mae.item()

def calc_MAPE_norm(dataloader, pred, y, pre_flattened=False):
    pred = dataloader.dataset.normalize_single(pred)
    y = dataloader.dataset.normalize_single(y)
    return calc_MAPE(pred, y, pre_flattened=False)

def calc_MAPE(pred, y, pre_flattened=False):
    if pre_flattened:
        p = pred
    else:
        p = torch.flatten(pred)
    # residuals = (y - torch.flatten(pred))
    # residuals = abs(/y)
    # residuals_sum = residuals.sum()
    # n = len(y)
    # mape = residuals_sum / n
    # print(mape)
    # p = torch.flatten(pred)
    residual = y - p
    # print(p, y)
    # print(residual)
    residual_percentage = residual / y
    # print(y)
    # exit()
    # print(residual, y)
    # print(residual_percentage)
    residual_abs = abs(residual_percentage)
    # print(residual_abs)
    residual_sum = residual_abs.sum()
    # print(residual_sum)
    mape = residual_sum / len(y)
    # print(mape)
    # for i in range(len(y)):
    #     if residual[i] == 0:
    #         print("found infinity cause")
    #     if y[i] == 0:
    #         print("FOUND ZERO")


    # if mape.item() > 1:
    #     print(y, torch.flatten(pred))
    mape = mape.item() * 100 #as percentage
    # print(mape)
    return mape