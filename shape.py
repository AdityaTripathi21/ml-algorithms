import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv') # turn csv into dataframe
print(df.shape) # (rows, cols)
print(df.describe()) # stats for each timeseries (column)

df['median_income'].hist(bins=50)
plt.xlabel("Median Income")
plt.ylabel("Count")
plt.show()

df['median_house_value'].hist(bins=50)
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.show()


# as median income increases, median house value also increaes, 
# however, median house values above 500k are underrepresented 
# as there seems to be a cut off in the data, with a horizontal line in the plot

plt.scatter(
    df['median_income'],
    df['median_house_value'],
    alpha=0.1
)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()