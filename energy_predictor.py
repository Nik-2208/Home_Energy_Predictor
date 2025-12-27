import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
import joblib

df = pd.read_csv(r"C:\Users\nikhi\OneDrive\Desktop\AI and ML\ML Projects\Energy_Consumption_Predictor\smart_home_energy_consumption_large.csv")

# print(df.head(5))
# print(df.columns)


df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Month'] = df['DateTime'].dt.month
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Hour'] = df['DateTime'].dt.hour

X = df[['Appliance Type', 'Outdoor Temperature (°C)', 'Season', 'Household Size', 'Month', 'DayOfWeek', 'Hour']]
y = df['Energy Consumption (kWh)']

X = pd.get_dummies(X, columns=['Appliance Type', 'Season'], drop_first=True)

X_train,X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

rf = RandomForestRegressor(n_estimators=700,max_depth=20,min_samples_split=5,random_state=42)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print("MSE: ",mean_squared_error(y_test,y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score: ",r2_score(y_test,y_pred))
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

joblib.dump(rf, "rfr_model.pkl",compress=9)



'''-----------------------------------------------------------------------------------------------'''


# X = np.array([2, 4, 6, 8, 10])
# y = np.array([5, 9, 13, 17, 21])

#MANUAL MACHINE LEARNING:-
'''
imp = 0
b = 0

y_pred = imp*X+b

error = y-y_pred
mse = np.mean(error ** 2)
print(mse)

learn_rate = 0.01

dw = -2 * np.mean(X * error)
db = -2 * np.mean(error)

imp = imp - learn_rate * dw
b = b - learn_rate * db

for i in range(1000):
    y_pred = imp * X + b
    error = y - y_pred
    dw = -2 * np.mean(X * error)
    db = -2 * np.mean(error)
    imp -= learn_rate * dw
    b -= learn_rate * db

#print(mse)
'''

#DATA VISUALIZATION 
'''
import matplotlib.pyplot as plt

plt.scatter(X, y, label="Actual Energy")
plt.plot(X, imp*X + b, color="red", label="Predicted Line")
plt.xlabel("Appliance Hours")
plt.ylabel("Energy Used")
plt.legend()
plt.show()


residuals = y - (imp*X + b)

plt.scatter(X, residuals)
plt.axhline(0, color='red')
plt.xlabel("Appliance Hours")
plt.ylabel("Error")
plt.show()
'''


#LINEAR REGRESSION:-
'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
'''

#SIGMOID FUNCTION - Gives Confidence Score between 0 and 1
'''def sigmoid(z):
    return 1 / (1 + np.exp(-z))

imp = 0.5
b = -4

z = imp * X + b
y_prob = sigmoid(z)
print(y_prob)

y_pred = (y_prob >= 0.5).astype(int)
print(y_pred)
'''

#CONFUSION MATRIX
'''from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))
'''

#GRIDSEARCHCV - Used to find best parameters in dataset
'''
from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(
    SVC(),
    params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
'''

