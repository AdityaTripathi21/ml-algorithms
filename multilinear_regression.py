import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

df = pd.read_csv('housing.csv') # turn csv into dataframe

x = df[['longitude', 'latitude', 'housing_median_age', 'median_income']]    # double brackets for (N,2) df shape
y = df[['median_house_value']] 

x = x.to_numpy()
y = y.to_numpy()

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# custom dataset class, must implement len and getitem, init is optional, this is protocol, not inheritance
class CustomDataset(Dataset):
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# shuffling to reduce sampling bias
perm = torch.randperm(x.size(0)) # remember that x.size(0) is the first dimension of x, which is N
x = x[perm]
y = y[perm]

split = int(0.8 * x.size(0)) # 80% split

x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

    
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True    # shuffling within the training dataset
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False    # don't shuffle test data
)

model = nn.Linear(4,1) # 4 features, 1 output
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_dataloader:
        y_hat = model(x_batch)
        loss = loss_function(y_hat, y_batch)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

    avg_loss = total_loss/len(train_dataloader)
    print(f"Epoch {epoch+1}, Train Loss: ", avg_loss)










