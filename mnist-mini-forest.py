import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
all_idx = np.concatenate([c1_idx, c2_idx])
all_idx = np.random.permutation(all_idx)

# train/val/test split
train_idx = all_idx[0:1000]
validation_idx = all_idx[1001:2000]
test_idx = all_idx[2001:3000]

# x_train: digits, y_train: labels
x_train = data_fea[train_idx,:]/255
y_train = data_gnd[train_idx,:]

# logregr model initialization and training
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train.ravel())


# Make predictions
y_pred=clf.predict(data_fea[test_idx,:])

# Assess performance
score = clf.score(data_fea[test_idx,:], data_gnd[test_idx,:])
print(score)




