import pandas as pd;
import numpy as np;

df = pd.read_csv('Heart.csv')


def hypothesis(theta, X):
    z = np.dot(theta,X.T)
    return 1/(1+np.exp(-(z))) - 0.0000001

def cost(X, y, theta):
    y1 = hypothesis(theta,X)
    return -(1/len(X))*np.sum(y*np.log(y1) + (1-y)*np.log(1-y1))

def gradientDescent(X, y, theta, alpha, epoch):
    m = len(X)
    J = [cost(X, y, theta)]
    for i in range(0,epoch):
        h = hypothesis(theta, X)
        for j in range(0, len(X.columns)):
            theta[j] -= (alpha/m) * np.sum((h-y)*X.iloc[:,j])
        J.append(cost(X, y, theta))
    return J, theta


def predict(X, y, theta, alpha, epoch):
    J, theta = gradientDescent(X, y, theta, alpha, epoch)
    h = hypothesis(theta, X)
    for i in range(0,len(h)):
        if h[i]>=0.5:
            h[i] = 1
        else:
            h[i] = 0
    y = list(y)
    acc=0
    for i in range(0,len(y)):
        if h[i]==y[i]:
            acc+=1
    return J, acc


df["ChestPain"]= df.ChestPain.replace({"typical": 1, "asymptomatic": 2, "nonanginal": 3, "nontypical": 4})
df["Thal"] = df.Thal.replace({"fixed": 1, "normal":2, "reversable":3})
df["AHD"] = df.AHD.replace({"Yes": 1, "No":0})


df = pd.concat([pd.Series(1, index = df.index, name = '00'), df], axis=1)

X = df.drop(columns=["Unnamed: 0"])
y= df["AHD"]


theta = [0.5]*len(X.columns)
J, acc = predict(X, y, theta, 0.0001, 20000)


print(str(acc) + " are correctly predicted from " + str(len(y)) + " examples.")


import matplotlib.pyplot as plt
plt.figure(figsize = (12, 8))
plt.scatter(range(0, len(J)), J)
plt.show()
