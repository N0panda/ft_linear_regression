import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

data = pd.read_csv("data.csv")

def normalisation(X):
    return ((X - np.mean(X)) / np.std(X))

def reverse_norm(X, base):
    return ((X * np.std(base)) + np.mean(base))

def ft_predict(X, theta0, theta1):
    return theta0 + (theta1 * X)

def ft_cost(X, y, theta0, theta1):
    y_pred = ft_predict(X, theta0, theta1)
    m = X.shape[0]
    cost = (1/(2 * m)) * np.sum((y_pred - y)**2)
    return float(cost)

def ft_grad_descent(X, y, theta, alpha, n):
    i = 0
    m = len(y)
    theta0 = theta[0, 0]
    theta1 = theta[1, 0]
    while i < n:
        tmp0 = theta0 - ((alpha / m) * np.sum(ft_predict(X, theta0, theta1) - y))
        tmp1 = theta1 - ((alpha / m) * np.sum((ft_predict(X, theta0, theta1) - y) * X))
        theta0 = tmp0
        theta1 = tmp1
        i += 1
    return np.array([[theta0], [theta1]])

file = pd.DataFrame({"theta0": [0], "theta1": [0]})
file.to_csv("theta.csv", index=False)
theta = pd.read_csv("theta.csv")
theta = np.array(theta).T

X = data.values[:, 0]
X = X.reshape(X.shape[0], 1)
X = normalisation(X)

y = data.values[:, 1]
y = y.reshape(y.shape[0], 1)
y = normalisation(y)

dx = data.values[:, 0]
dx = dx.reshape(dx.shape[0], 1)

plt.plot(data.values[:, 0], data.values[:, 1], 'r.')
plt.plot(data.values[:, 0], ft_predict(dx, theta[0, 0], theta[1, 0]))
plt.xlabel("Km")
plt.ylabel("Price")
plt.title("Before regression")
plt.show()

cost_before = ft_cost(X, y, theta[0, 0], theta[1, 0])
theta = ft_grad_descent(X, y, theta, 0.01, 1000)
cost_after = ft_cost(X, y, theta[0, 0], theta[1, 0])

theta_file = pd.DataFrame({"theta0": theta[0], "theta1": theta[1]})
theta_file.to_csv("theta.csv", index=False)

y_pred = ft_predict(X, theta[0, 0], theta[1, 0])
y_pred = reverse_norm(y_pred, data.values[:, 1])

plt.plot(data.values[:, 0], data.values[:, 1], 'r.')
plt.plot(data.values[:, 0], y_pred)
plt.xlabel("Km")
plt.ylabel("Price")
plt.title("After regression")
plt.show()
print("Cost before = ", cost_before," , Cost after = ", cost_after)
