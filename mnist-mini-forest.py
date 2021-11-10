import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sympy import plot
mat = scipy.io.loadmat('MNISTmini.mat')

data_fea = mat['train_fea1']
data_gnd = mat['train_gnd1']

# class 0 = 5s, class 1 = 9s
label = [5,9]

# Fetch all indices of both classes
c1_idx = np.where(data_gnd[:,0] == label[0])
c2_idx = np.where(data_gnd[:,0] == label[1])

# Get 1500 from each class
c1_idx = np.array(c1_idx)
c1_idx = c1_idx[0,0:1500]
c2_idx = np.array(c2_idx)
c2_idx = c2_idx[0,0:1500]

# Concatenate arrays and perform random permutation
# all_idx = np.array([])
# for i in range(500, 2000, 500):
#     temp1 = c1_idx[i-500:i] #i = 500,  0-499, i=1000, 500-999, 
#     temp2 = c2_idx[i-500:i]
#     all_idx = np.concatenate([temp1, temp2])
# all_idx = np.random.permutation(all_idx)

# train/val/test split
train_idx = np.concatenate([c1_idx[:500], c2_idx[:500]]) #all_idx[0:1000]
validation_idx = np.concatenate([c1_idx[500:1000], c2_idx[500:1000]])  #all_idx[1001:2000]
test_idx = np.concatenate([c1_idx[1000:1500], c2_idx[1000:1500]]) #all_idx[2001:3000]

# x_train: digits, y_train: labels
x_train = data_fea[train_idx,:]/255
y_train = data_gnd[train_idx,:]
plotx = []
ploty1 = []
ploty = []
for i in range(1, 51):
    # logregr model initialization and training
    clf=RandomForestClassifier(n_estimators=i, bootstrap=False, max_leaf_nodes=100)
    clf.fit(x_train,y_train.ravel())

    # Make predictions
    y_pred=clf.predict(data_fea[validation_idx,:])

    # Assess performance
    val_score = clf.score(data_fea[validation_idx,:], data_gnd[validation_idx,:])
    train_score = clf.score(data_fea[train_idx,:], data_gnd[train_idx,:])
    ploty.append(val_score)
    ploty1.append(train_score)
    plotx.append(i)

df = pd.DataFrame()
df['x'] = plotx
df['val_score'] = ploty
df['train_score'] = ploty1
maxVal = df['val_score']
maxValidx = maxVal.idxmax()
maxTrain = df['train_score']
maxTrainidx = maxTrain.idxmax()
x = plotx[maxValidx]
maxVal = ploty[maxValidx]

ax = plt.gca()
df.plot(kind='line', x='x', y='val_score', color='blue', ax=ax)
df.plot(kind='line', x='x', y='train_score', color='red', ax=ax)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))
ax1.plot(x, maxVal, 'go', label='marker only')

plt.show()




