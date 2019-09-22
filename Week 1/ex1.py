import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'Data/'

data = pd.read_csv(path + 'ex1data1.txt', sep=',', names=['X1', 'X2'])

m = len(data)

X = np.stack([np.ones(m), data['X1']], axis = 1)
y = data['X2'].copy()

def cost(X, y, theta):
    m = len(X)
    if type(theta) == list:
        theta = np.array(theta)
    return np.sum((X @ theta.T - y) ** 2)/(2 * m)

def gradient_descent(X, y, theta, alpha = 0.01, iter = 3000):
    m = len(X)
    theta = theta.copy()
    J_hist = []

    for i in range(iter):
        temp0 = theta[0] - alpha * np.sum(X @ theta.T - y) / m
        temp1 = theta[1] - alpha * np.sum((X @ theta.T - y) * X[:,1]) / m

        theta[0] = temp0
        theta[1] = temp1

        J_hist.append(cost(X, y, theta))
    
    return theta, J_hist

theta = np.zeros(2)

theta, J_hist = gradient_descent(X, y, theta)

plt.plot(np.linspace(1, len(J_hist), num=3000), J_hist)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost $J(\\theta_0, \\theta_1)$')
plt.show()

plt.scatter(X[:,1], y)
plt.plot(X[:,1], np.dot(X, theta), '-', color='r')
plt.legend(['Training data', 'Linear regression'])
plt.show()