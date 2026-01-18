import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv("housing.csv")

x = df.drop(columns=["median_house_value", "ocean_proximity"])  # drop y and numeric column
y = df[['median_house_value']]

# replace missing values
x = x.fillna(x.mean())

x = x.to_numpy()
y = y.to_numpy()

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


