import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

mat = scipy.io.loadmat("MNISTmini.mat")

data_fea = mat["train_fea1"].astype(np.float32)/255
data_gnd = mat["train_gnd1"]

# class 0 = 5s, class 1 = 9s
label = [5, 9]

# Fetch all indices of both classes
c1_idx = np.where(data_gnd[:, 0] == label[0])
c2_idx = np.where(data_gnd[:, 0] == label[1])

# Get 1500 from each class
c1_idx = np.array(c1_idx)
c1_idx = c1_idx[0, 0:1500]
c2_idx = np.array(c2_idx)
c2_idx = c2_idx[0, 0:1500]


# train/val/test split
train_idx = np.concatenate([c1_idx[:500], c2_idx[:500]])  # all_idx[0:1000]
validation_idx = np.concatenate(
    [c1_idx[500:1000], c2_idx[500:1000]]
)  # all_idx[1001:2000]
test_idx = np.concatenate(
    [c1_idx[1000:1500], c2_idx[1000:1500]])  # all_idx[2001:3000]

# x_train: digits, y_train: labels
x_train = data_fea[train_idx, :]
y_train = data_gnd[train_idx, :]

plotI = []
plotMaxDepthVal = []
plotMaxDepthTrain = []
for i in range(1, 1000):
    # random forest model initialization and training
    maxDepth = RandomForestClassifier(
        n_estimators=1, bootstrap=False, random_state=0, max_depth = i)
    maxDepth.fit(x_train, y_train.ravel())
    
    # Make predictions
    maxDepthPred = maxDepth.predict(data_fea[validation_idx, :])

    # Assess performance
    maxDepth_val_score = maxDepth.score(
        data_fea[validation_idx, :], data_gnd[validation_idx, :])
    maxDepth_train_score = maxDepth.score(data_fea[train_idx, :], data_gnd[train_idx, :])


    # Push the performace and value of C into list
    plotMaxDepthVal.append(maxDepth_val_score)
    plotMaxDepthTrain.append(maxDepth_train_score)
    plotI.append(i)


# Find (x,y) of best validation score
maxDepthVal = max(plotMaxDepthVal)
maxDepthValidx = plotMaxDepthVal.index(maxDepthVal)
maxplotIidx = plotI[maxDepthValidx]
print(maxplotIidx)
print(maxDepthVal)

# Plot the Score of both validation and training
plt.figure(1)
plt.semilogx(plotI, plotMaxDepthVal, color="blue", label="Val_Score")
plt.semilogx(plotI, plotMaxDepthTrain, color="red", label="Train_Score")
plt.legend(loc="lower right")
plt.text(maxplotIidx, maxDepthVal, ' {} , {}'.format(maxplotIidx, maxDepthVal))
plt.xlabel("Max Depth Value Increasing Continously")
plt.ylabel("Score Values")

# Show Graph
plt.show()

plotI2 = []
plotMaxLeafVal = []
plotMaxLeafTrain = []
for i in range(2,1000):
    # random forest model initialization and training
    maxLeaf = RandomForestClassifier(
            n_estimators=1, bootstrap=False, random_state=0, max_leaf_nodes = i, max_depth = maxDepthValidx)
    maxLeaf.fit(x_train, y_train.ravel())

    # Make predictions
    maxLeafPred = maxLeaf.predict(data_fea[validation_idx, :])

    # Assess performance
    maxLeaf_val_score = maxLeaf.score(
        data_fea[validation_idx, :], data_gnd[validation_idx, :])
    maxLeaf_train_score = maxLeaf.score(data_fea[train_idx, :], data_gnd[train_idx, :])

    # Push the performace and value of C into list
    plotMaxLeafVal.append(maxLeaf_val_score)
    plotMaxLeafTrain.append(maxLeaf_train_score)
    plotI2.append(i)


# Find (x,y) of best validation score
maxLeafVal = max(plotMaxLeafVal)
maxLeafValidx = plotMaxLeafVal.index(maxLeafVal)
maxplotI2idx = plotI2[maxLeafValidx]
print(maxplotI2idx)
print(maxLeafVal)

# Plot the Score of both validation and training
plt.figure(2)
plt.semilogx(plotI2, plotMaxLeafVal, color="blue", label="Val_Score")
plt.semilogx(plotI2, plotMaxLeafTrain, color="red", label="Train_Score")
plt.legend(loc="lower right")
plt.text(maxplotI2idx, maxLeafVal, ' {} , {}'.format(maxplotI2idx, maxLeafVal))
plt.xlabel("Max Leaf Node Value Increasing Continously")
plt.ylabel("Score Values")

# Show Graph
plt.show()


plotI3 = []
plotMaxEstimatorsVal = []
plotMaxEstimatorsTrain = []
for i in range(1,500):
    # random forest model initialization and training
    maxEstimators = RandomForestClassifier(
            n_estimators=i, bootstrap=False, random_state=0, max_leaf_nodes = maxLeafValidx, max_depth = maxDepthValidx)
    maxEstimators.fit(x_train, y_train.ravel())

    # Make predictions
    maxEstimatorsPred = maxEstimators.predict(data_fea[validation_idx, :])

    # Assess performance
    maxEstimators_val_score = maxEstimators.score(
        data_fea[validation_idx, :], data_gnd[validation_idx, :])
    maxEstimators_train_score = maxEstimators.score(data_fea[train_idx, :], data_gnd[train_idx, :])

    # Push the performace and value of C into list
    plotMaxEstimatorsVal.append(maxEstimators_val_score)
    plotMaxEstimatorsTrain.append(maxEstimators_train_score)
    plotI3.append(i)


# Find (x,y) of best validation score
maxEstimatorsVal = max(plotMaxEstimatorsVal)
maxEstimatorsValidx = plotMaxEstimatorsVal.index(maxEstimatorsVal)
maxplotI3idx = plotI3[maxEstimatorsValidx]
print(maxplotI3idx)
print(maxEstimatorsVal)

# Plot the Score of both validation and training
plt.figure(3)
plt.semilogx(plotI3, plotMaxEstimatorsVal, color="blue", label="Val_Score")
plt.semilogx(plotI3, plotMaxEstimatorsTrain, color="red", label="Train_Score")
plt.legend(loc="lower right")
plt.text(maxplotI3idx, maxEstimatorsVal, ' {} , {}'.format(maxplotI3idx, maxEstimatorsVal))
plt.xlabel("# of Trees Increasing Continously")
plt.ylabel("Score Values")

# Show Graph
plt.show()