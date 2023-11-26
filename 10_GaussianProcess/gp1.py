import numpy as np
from matplotlib import pyplot as plt

# specify training data
xTrain = np.array([1,2,3,4])
# yTrain = np.array([.25, .95, 2.3, 3.9])
yTrain = np.sin(xTrain)
# specify test data
xTest = np.arange(0, 7, .2)



# defin hyperparameters
c2=0.25 # constant coefficient of quadratic term in prior mean function
ell=1 # horizontal length scale parameter in the squared exponential function
sigmaF2=2 #sigmaF2 is the variance of the multivariate gaussian distribution
sigmaN2=0.005 #sigmaN2 is the variance of the regression noise-term
# mean for prior and cov functions
def priormean(xin):
    print(type(xin))
    return c2*xin**2

def corrFunc(xa,xb):
    return sigmaF2*np.exp(-((xa-xb)**2)/(2.0*ell**2))

mxTrain = priormean(xTrain)
mxTest = priormean(xTest)

# Calculate the covariance matrix by evaluating the covariance function at the training data x-values.
KB=np.zeros((len(xTrain),len(xTrain)))
for i in range(len(xTrain)):
    for j in range(i,len(xTrain)):
        noise=(sigmaN2 if i==j else 0)
        k=corrFunc(xTrain[i],xTrain[j])+noise
        KB[i][j]=k
        KB[j][i]=k
KBInv = np.linalg.inv(KB)
# Calculate the covariance matrix K_Star  between training x-values and prediction x-values
Ks=np.zeros((len(xTest),len(xTrain)))
for i in range(len(xTest)):
    for j in range(len(xTrain)):
        k=corrFunc(xTest[i],xTrain[j])
        Ks[i][j]=k
# Calculate the covariance matrix K_Star_Star  between prediction x-values
Kss=np.zeros((len(xTest),len(xTest)))
for i in range(len(xTest)):
    for j in range(i,len(xTest)):
        noise=(sigmaN2 if i==j else 0)
        k=corrFunc(xTest[i],xTest[j])+noise
        Kss[i][j]=k
        Kss[j][i]=k
        
# Calculate the prediction
mus = priormean(xTest)
yTest = mus+np.dot(np.dot(Ks, KBInv),(yTrain-mxTrain))

# Calculate the covariance of the predictions
yvar=np.diag(Kss-np.dot(Ks,np.dot(KBInv,np.transpose(Ks))))
stds=np.sqrt(yvar)

# plotting
plt.plot(xTest,mxTest, c='r', label='mean')
plt.plot(xTrain, yTrain, c='b', label='trained')
plt.plot(xTest, yTest, c='g', label='test prediction')
fillx = np.hstack((xTest, xTest[::-1]))
filly = np.hstack((yTest+2*stds, yTest[::-1]-2*stds[::-1]))
plt.fill(fillx, filly, facecolor='gray', edgecolor='white', alpha=0.3)
plt.scatter(xTrain, yTrain, zorder = 20)
plt.title("gaussian process without noise consideration")
plt.legend()
plt.show()