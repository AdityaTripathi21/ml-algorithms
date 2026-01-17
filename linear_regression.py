import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

df = pd.read_csv('housing.csv') # turn csv into dataframe

x = df[['median_income']].values    # double brackets for (N,1) df shape
y = df[['median_house_value']].values # or use .to_numpy() {newer way and better}

print(x.dtype)

# convert to tensors
x = torch.tensor(x, dtype=torch.float32) # (N,1 shape)
y = torch.tensor(y, dtype=torch.float32) # (N,1 shape)

torch.manual_seed(42) # fix seed for fixed results
perm = torch.randperm(x.size(0)) # random perm needed to reduce sampling bias because 
                                 # data is ordered

x = x[perm] 
y = y[perm]

split = int(0.8 * x.size(0)) # 80% split

x_train, x_test = x[:split], x[split:] # (0.8N, 1), (0.2N, 1)
y_train, y_test = y[:split], y[split:] 

model = nn.Linear(1,1) # Input is median income, output is median house value

loss_function = nn.MSELoss() # Mean squared error is the loss function

optimizer = optim.SGD(model.parameters(), lr = 0.01) # stochastic gradient descent
                                                     # learning rate of 0.1
                                                     # even though this is SGD,
                                                     # we are computing the full batch

epochs = 1000

for epoch in range(epochs):
    y_hat = model(x_train) # forwards pass

    loss = loss_function(y_hat, y_train) # compute loss

    optimizer.zero_grad() # zero out gradients to not accumulate gradients

    loss.backward() # backprop, compute loss with respect to parameters to get gradients

    optimizer.step() # update the parameters

    if epoch % 100 == 0: # just keep track of loss
        print(f"Epoch {epoch}, MSE Loss: {loss.item():.2f}")



with torch.no_grad(): # faster execution, and use with statement for duration of the context
    train_preds = model(x_train)
    test_preds = model(x_test)

    train_rmse = torch.sqrt(loss_function(train_preds, y_train)) # RMSE, average residual magnitude
    test_rmse = torch.sqrt(loss_function(test_preds, y_test))    # RMSE

    print("Train RMSE:", train_rmse.item())
    print("Test RMSE:", test_rmse.item())






    






















