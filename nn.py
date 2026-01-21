import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv("housing.csv")

x = df.drop(columns=["median_house_value", "ocean_proximity"])  # drop y and categorical column
y = df[['median_house_value']]

# replace missing values
x = x.fillna(x.mean())

x = x.to_numpy()
y = y.to_numpy()

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# shuffling to reduce sampling bias
perm = torch.randperm(x.size(0)) # remember that x.size(0) is the first dimension of x, which is N
x = x[perm]
y = y[perm]

split = int(0.8 * x.size(0)) # 80% split

x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_mean = x_train.mean(dim=0) # dim=0 is column wise, so x_mean and x_std have shape (d,)
x_std = x_train.std(dim=0)

# standardization because some features are too large, and this causes the loss to be nan
x_train = (x_train - x_mean) / (x_std + 1e-8)
x_test  = (x_test  - x_mean) / (x_std + 1e-8)

y_scale = 100000.0

y_train = y_train / y_scale
y_test  = y_test  / y_scale

# custom dataset class, must implement len and getitem, init is optional, this is protocol, not inheritance
class CustomDataset(Dataset):
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

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

input_dim = x_train.shape[1]

model = nn.Sequential(          # nn.sequential stacks layers together and allows nice composition
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)     # Adam gives each parameter own lr + momentum (mean of gradients)

epochs = 100

# training loop
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
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss}")



model.eval() # model.eval for testing
test_loss = 0.0

with torch.no_grad():
    for x_batch, y_batch in test_dataloader:
        y_hat = model(x_batch)

        loss = loss_function(y_hat, y_batch)
        test_loss += loss.item()

    test_loss /= len(test_dataloader) # average batch loss over the entire epoch
    print("Test Loss:", test_loss)
    
    print("RMSE: ", torch.sqrt(torch.tensor(test_loss))) # multiply RMSE by 100k because of scaling/standardization
    # RMSE ends up being about 0.51, so predictions are on average 51k off, better than any type of linear regression
