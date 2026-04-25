assignment_info = {
    "Name": "Pulkit Srivastava",
    "Roll Number": "102303803",
    "Group": "3C55"
}

import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def custom_simulator(n, r, d, l, t):
    th = r * (1 - l) * (n / 100)
    la = d * (1 + l)
    return th, la

data = []
for _ in range(1000):
    n = random.randint(5, 100)
    r = random.uniform(1, 100)
    d = random.uniform(1, 50)
    l = random.uniform(0, 0.3)
    t = random.uniform(5, 50)
    
    th, la = custom_simulator(n, r, d, l, t)
    data.append([n, r, d, l, t, th, la])

df = pd.DataFrame(data, columns=["n", "r", "d", "l", "t", "th", "la"])

X = df[["n", "r", "d", "l", "t"]]
y = df["th"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "XGBoost": XGBRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results.append([name, mse, mae, r2])

results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2_Score"])
results_df = results_df.sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

print(pd.DataFrame([assignment_info]).to_string(index=False))
print("\n")
print(results_df.to_string(index=False))
