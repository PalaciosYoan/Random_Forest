import scipy.io
import numpy as np
import pandas as pd
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
#number of trees
#maxleafs
#depth
df = pd.DataFrame(columns=['id', 'numTrees', 'currenmaxtLeaf', 'currDepth', 'Val_score','Train_score', 'minLeaf', 'maxFea', 'minSplit', 'min_impurity_decrease', 'warm_start'])
counter = 1
# # for numTrees in range(1,52,10):
numTrees = 1
for numleafs in range(2, 63,10):
    for numDepth in range(1,61, 10):
        for minLeaf in range(1, 61, 10):
            for maxFea in range(1, 61, 10):
                for minSplit in range(2, 63, 10):
                    for minImpurity in range(0, 60, 10):
                        for warmStart in [False, True]:
                            random_forest = RandomForestClassifier(n_estimators=numTrees,min_samples_leaf=minLeaf, max_features = maxFea, random_state=0, max_depth = numDepth, max_leaf_nodes=numleafs, min_samples_split=minSplit, min_impurity_decrease=minImpurity, warm_start=warmStart)
                            random_forest.fit(x_train, y_train.ravel())
                            currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
                            currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
                            df = df.append({'id':counter, 'numTrees':numTrees, 'currenmaxtLeaf':numleafs, 'currDepth':numDepth, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'minLeaf':minLeaf, 'maxFea':10, 'minSplit':minSplit, 'min_impurity_decrease':minImpurity, 'warm_start':warmStart}, ignore_index=True)
                            counter+=1
    print("Done with few things")

df = pd.DataFrame(columns=['id', 'numTrees', 'currentLeaf', 'currDepth', 'Val_score','Train_score'])
# counter = 1

# for numleafs in range(2, 103, 10):
#     for numDepth in range(1,100, 10):
#         random_forest = RandomForestClassifier(n_estimators=1, random_state=0, max_depth = numDepth, max_leaf_nodes=numleafs)
#         random_forest.fit(x_train, y_train.ravel())
#         currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
#         currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
#         df = df.append({'id':counter, 'numTrees':1, 'currentLeaf':numleafs, 'currDepth':numDepth, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train}, ignore_index=True)
#         counter+=1
# df = pd.DataFrame(columns=['id', 'numTrees', 'Val_score', 'Train_score'])
# counter = 1
# for nTrees in range(1, 500, 10):
#     random_forest = RandomForestClassifier(n_estimators=nTrees,min_samples_leaf=1, max_features = 10, random_state=0, max_depth = 11, max_leaf_nodes=32, min_samples_split=2, min_impurity_decrease=0, warm_start=False)
#     random_forest.fit(x_train, y_train.ravel())
#     currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
#     currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
#     df = df.append({'id':counter, 'numTrees':nTrees, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train}, ignore_index=True)
#     counter += 1
df.to_csv('c.csv', index=False)
plt.figure(1)
plt.plot(df['id'], df['Train_score'], color="blue", label="Val_Score")
plt.plot(df['id'], df['Val_score'], color="red", label="Train_Score")
plt.legend(loc="lower right")
plt.xlabel("Max Depth Value Increasing Continously")
plt.ylabel("Score Values")
print("Done")
# Show Graph
plt.show()

# ax = plt.gca()
# df.plot(kind='line', x='id', y='Train_score', color='blue', ax=ax)
# df.plot(kind='line', x='id', y='Val_score', color='red', ax=ax)

# plt.show()

# for i in range(1, 1000): #max deapth
#     # random forest model initialization and training
#     maxDepth = RandomForestClassifier(
#         n_estimators=1, bootstrap=False, random_state=0, max_depth = i)
#     maxDepth.fit(x_train, y_train.ravel())
    
#     # Make predictions
#     maxDepthPred = maxDepth.predict(data_fea[validation_idx, :])

#     # Assess performance
#     maxDepth_val_score = maxDepth.score(
#         data_fea[validation_idx, :], data_gnd[validation_idx, :])
#     maxDepth_train_score = maxDepth.score(data_fea[train_idx, :], data_gnd[train_idx, :])


#     # Push the performace and value of C into list
#     plotMaxDepthVal.append(maxDepth_val_score)
#     plotMaxDepthTrain.append(maxDepth_train_score)
#     plotI.append(i)


# # Find (x,y) of best validation score
# maxDepthVal = max(plotMaxDepthVal)
# maxDepthValidx = plotMaxDepthVal.index(maxDepthVal)
# maxplotIidx = plotI[maxDepthValidx]
# print(maxplotIidx)
# print(maxDepthVal)

# # Plot the Score of both validation and training
# plt.figure(1)
# plt.semilogx(plotI, plotMaxDepthVal, color="blue", label="Val_Score")
# plt.semilogx(plotI, plotMaxDepthTrain, color="red", label="Train_Score")
# plt.legend(loc="lower right")
# plt.text(maxplotIidx, maxDepthVal, ' {} , {}'.format(maxplotIidx, maxDepthVal))
# plt.xlabel("Max Depth Value Increasing Continously")
# plt.ylabel("Score Values")

# # Show Graph
# plt.show()

# plotI2 = []
# plotMaxLeafVal = []
# plotMaxLeafTrain = []
# for i in range(2,2000): #max leaf
#     # random forest model initialization and training
#     maxLeaf = RandomForestClassifier(
#             n_estimators=i, bootstrap=True, random_state=0, max_leaf_nodes = i, max_depth = maxDepthValidx)
#     maxLeaf.fit(x_train, y_train.ravel())

#     # Make predictions
#     maxLeafPred = maxLeaf.predict(data_fea[validation_idx, :])

#     # Assess performance
#     maxLeaf_val_score = maxLeaf.score(
#         data_fea[validation_idx, :], data_gnd[validation_idx, :])
#     maxLeaf_train_score = maxLeaf.score(data_fea[train_idx, :], data_gnd[train_idx, :])

#     # Push the performace and value of C into list
#     plotMaxLeafVal.append(maxLeaf_val_score)
#     plotMaxLeafTrain.append(maxLeaf_train_score)
#     plotI2.append(i)


# # Find (x,y) of best validation score
# maxLeafVal = max(plotMaxLeafVal)
# maxLeafValidx = plotMaxLeafVal.index(maxLeafVal)
# maxplotI2idx = plotI2[maxLeafValidx]
# print(maxplotI2idx)
# print(maxLeafVal)

# # Plot the Score of both validation and training
# plt.figure(2)
# plt.semilogx(plotI2, plotMaxLeafVal, color="blue", label="Val_Score")
# plt.semilogx(plotI2, plotMaxLeafTrain, color="red", label="Train_Score")
# plt.legend(loc="lower right")
# plt.text(maxplotI2idx, maxLeafVal, ' {} , {}'.format(maxplotI2idx, maxLeafVal))
# plt.xlabel("Max Leaf Node Value Increasing Continously")
# plt.ylabel("Score Values")

# # Show Graph
# plt.show()


# plotI3 = []
# plotMaxEstimatorsVal = []
# plotMaxEstimatorsTrain = []
# for i in range(1,500): #num of trees
#     # random forest model initialization and training
#     maxEstimators = RandomForestClassifier(
#             n_estimators=i, bootstrap=False, random_state=0, max_leaf_nodes = maxLeafValidx, max_depth = maxDepthValidx)
#     maxEstimators.fit(x_train, y_train.ravel())

#     # Make predictions
#     maxEstimatorsPred = maxEstimators.predict(data_fea[validation_idx, :])

#     # Assess performance
#     maxEstimators_val_score = maxEstimators.score(
#         data_fea[validation_idx, :], data_gnd[validation_idx, :])
#     maxEstimators_train_score = maxEstimators.score(data_fea[train_idx, :], data_gnd[train_idx, :])

#     # Push the performace and value of C into list
#     plotMaxEstimatorsVal.append(maxEstimators_val_score)
#     plotMaxEstimatorsTrain.append(maxEstimators_train_score)
#     plotI3.append(i)


# # Find (x,y) of best validation score
# maxEstimatorsVal = max(plotMaxEstimatorsVal)
# maxEstimatorsValidx = plotMaxEstimatorsVal.index(maxEstimatorsVal)
# maxplotI3idx = plotI3[maxEstimatorsValidx]
# print(maxplotI3idx)
# print(maxEstimatorsVal)

# # Plot the Score of both validation and training
# plt.figure(3)
# plt.semilogx(plotI3, plotMaxEstimatorsVal, color="blue", label="Val_Score")
# plt.semilogx(plotI3, plotMaxEstimatorsTrain, color="red", label="Train_Score")
# plt.legend(loc="lower right")
# plt.text(maxplotI3idx, maxEstimatorsVal, ' {} , {}'.format(maxplotI3idx, maxEstimatorsVal))
# plt.xlabel("# of Trees Increasing Continously")
# plt.ylabel("Score Values")

# Show Graph
plt.show()