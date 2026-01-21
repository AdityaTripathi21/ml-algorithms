# using sklearn instead of PyTorch because PyTorch doesn't have a decision tree class

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


df = pd.read_csv("housing.csv")

x = df.drop(columns=["median_house_value", "ocean_proximity"])  # drop y and categorical column
y = df['median_house_value']    # no double brackets because sklearn expects (N,) instead of (N, 1)

x = x.fillna(x.mean())  # fill in missing values

x = x.to_numpy()
y = y.to_numpy()


# train_test_split does shuffling for you and decides the test and training split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# instantiating the model
tree = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=50,
    random_state=42
)

# training
tree.fit(x_train, y_train)

# predicting on test set
y_hat = tree.predict(x_test)

# test rmse
test_rmse = root_mean_squared_error(y_test, y_hat)
# print("RMSE:", test_rmse)

# predicting on training set
y_hat_train = tree.predict(x_train)

# train rmse
train_rmse = root_mean_squared_error(y_train, y_hat_train)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# finetuning depth
for depth in [2, 4, 6, 8, 10]:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=50,
        random_state=42
    )
    tree.fit(x_train, y_train)

    train_rmse = root_mean_squared_error(y_train, tree.predict(x_train))
    test_rmse  = root_mean_squared_error(y_test, tree.predict(x_test))

    # print(f"depth={depth} | train={train_rmse:.0f} | test={test_rmse:.0f}")

# finetuning min_samples_leaf
for leaf in [5, 20, 50, 100, 300]:
    tree = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=leaf,
        random_state=42
    )
    tree.fit(x_train, y_train)

    train_rmse = root_mean_squared_error(y_train, tree.predict(x_train))
    test_rmse  = root_mean_squared_error(y_test, tree.predict(x_test))

    # print(f"leaf={leaf} | train={train_rmse:.0f} | test={test_rmse:.0f}")


# finetuned tree
tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

tree.fit(x_train, y_train)

train_rmse = root_mean_squared_error(y_train, tree.predict(x_train))
test_rmse  = root_mean_squared_error(y_test, tree.predict(x_test))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# in the end, test rmse reduced by 10k and training rmse reduced by almost 14k

# visualizing the tree
plt.figure(figsize=(32, 14))
plot_tree(
    tree,
    feature_names=df.drop(columns=["median_house_value", "ocean_proximity"]).columns,
    filled=True,
    fontsize=10
)
plt.show()

# the tree is greedy, and it picks one feature out of all the features at each node,
# and this feature gives the best immediate error reduction.
# From the tree, it seeems that median income is a very important feature early on

# for regression trees like this, the leaf values will be the median house value.
# the leaf samples are chosen from training data, and for regression trees like this,
# the prediction will be the mean of the samples

# gini impurity or entropy would be used for classification trees