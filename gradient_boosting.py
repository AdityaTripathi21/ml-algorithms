import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


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

model = XGBRegressor(
    n_estimators=300,   # total number of trees 
    max_depth=4,        # max depth of each tree
    learning_rate=0.05, # learning rate for each new tree, basically how strongly a tree contributes to the correction
    subsample=0.8,      # trees only see 80% of training data, similar to bootstrapping in rf
    colsample_bytree=0.8,   # trees only see 80% of the features, similar to bootstrapping in rf
    objective="reg:squarederror",   # loss function - regression: mean squared error
    random_state=42
)

model.fit(x_train, y_train)

y_hat = model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_hat)
print("XGBoost RMSE: ", rmse)   # around 50k, and the code ran extremely fast, which is why gradient boosting is used
                                # also more accurate than rf



