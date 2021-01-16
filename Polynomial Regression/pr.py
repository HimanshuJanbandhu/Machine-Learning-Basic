import pandas as pd
import numpy as np

df = pd.read_csv('position_salaries.csv')
df.head()
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
df.head()

df = df.drop(columns='Position')

y = df['Salary']
X = df.drop(columns='Salary')
X.head()

X['Level1'] = X['Level']**2
X['Level2'] = X['Level']**3
X.head()

m = len(X)
X = X/X.max()

def hypothesis(X, theta):
    pred = theta*X
    return np.sum(pred, axis=1)

def cost(X, y, theta):
    y1 = hypothesis(X, theta)
    return np.sum(np.sqrt((y1-y)**2))/2*m

def gradient_descent(X, y, theta, epoch, alpha):
    J=[]
    k=0
    while k<epoch:
        y1 = hypothesis(X, theta)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*sum((y1-y)*X.iloc[:, c])/m
        j = cost(X, y, theta)
        J.append(j)
        k += 1
    return J, theta

theta = np.array([0.0]*len(X.columns))
J, theta = gradient_descent(X, y, theta, 700, 0.05)

y_hat = hypothesis(X, theta)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=X['Level'],y= y)
plt.scatter(x=X['Level'], y=y_hat)
plt.show()

plt.figure()
plt.scatter(x=list(range(0, 700)), y=J)
plt.show()
