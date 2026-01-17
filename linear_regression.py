import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

df = pd.read_csv('housing.csv') # turn csv into dataframe

x = df[['median_income']].values    # double brackets for (N,1) df shape
y = df[['median_house_value']].values







