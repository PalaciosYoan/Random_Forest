from turtle import color
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


def main():
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
    # df = pd.DataFrame(columns=['id', 'numTrees', 'currenmaxtLeaf', 'currDepth', 'Val_score','Train_score', 'minLeaf', 'maxFea', 'minSplit', 'min_impurity_decrease', 'warm_start'])
    # counter = 1
    # # # for numTrees in range(1,52,10):
    # numTrees = 1
    # print("running!")
    # for numleafs in range(2, 63,10):
    #     for numDepth in range(1,61, 10): #done
    #         for minLeaf in range(1, 61, 10):
    #             for maxFea in range(1, 61, 10):
    #                 for minSplit in range(2, 63, 10):
    #                     for minImpurity in range(0, 60, 10):
    #                         for warmStart in [False, True]:
    #                             random_forest = RandomForestClassifier(n_estimators=numTrees,min_samples_leaf=minLeaf, max_features = maxFea, random_state=0, max_depth = numDepth, max_leaf_nodes=numleafs, min_samples_split=minSplit, min_impurity_decrease=minImpurity, warm_start=warmStart)
    #                             random_forest.fit(x_train, y_train.ravel())
    #                             currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
    #                             currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
    #                             df = df.append({'id':counter, 'numTrees':numTrees, 'currenmaxtLeaf':numleafs, 'currDepth':numDepth, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'minLeaf':minLeaf, 'maxFea':10, 'minSplit':minSplit, 'min_impurity_decrease':minImpurity, 'warm_start':warmStart}, ignore_index=True)
    #                             counter+=1
    #     print("Done with few things")

    # df = pd.DataFrame(columns=['id', 'numTrees', 'currentLeaf', 'currDepth', 'Val_score','Train_score'])
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
    # for nTrees in range(1, 61):
    #     random_forest = RandomForestClassifier(n_estimators=nTrees,min_samples_leaf=1, max_features = 10, random_state=0, max_depth = 11, max_leaf_nodes=32, min_samples_split=2, min_impurity_decrease=0, warm_start=False)
    #     random_forest.fit(x_train, y_train.ravel())
    #     currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
    #     currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
    #     df = df.append({'id':counter, 'numTrees':nTrees, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train}, ignore_index=True)
    #     counter += 1
    #     print(nTrees)
    
    '''Finding the best Depth for our tree'''
    
    #define needed columns
    df = pd.DataFrame(columns=['id', 'max_depth', 'Val_score', 'Train_score','Test_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for currDepth in range(1,20):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0, max_depth = currDepth)
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'max_depth':currDepth, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train,'Test_score': test_score, 'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./max_depth/max_depth10.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    maxDepthVal = df['max_depth'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(1)
    plt.plot(df['max_depth'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['max_depth'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['max_depth'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(id, bestValScore, color='green')
    
    plt.title("Number of Depth vs Score")
    plt.annotate(f"({maxDepthVal}, {bestValScore})", (maxDepthVal, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num Depth")
    plt.ylabel("Score Values")
    
    ## Time vs depth
    plt.figure(9)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of Depth vs Time")
    plt.plot(df['max_depth'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(maxDepthVal, timeval, color='green')
    

    #################################################################################################
    
    '''Finding min_samples_split can be between 4 & 2 or 24(this one is the best of best, iterations of 39)'''
    df = pd.DataFrame(columns=['id', 'min_samples_split', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for minSplit in range(2, 10):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0, min_samples_split=minSplit, max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'min_samples_split':minSplit, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'Test_score': test_score, 'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./min_samples_split/min_samples_split.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval1 = df['min_samples_split'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(2)
    plt.plot(df['min_samples_split'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['min_samples_split'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['min_samples_split'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval1, bestValScore, color='green')
    
    plt.title("Number of min_samples_split vs Score")
    plt.annotate(f"({bestval1}, {bestValScore})", (bestval1, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num min_samples_split")
    plt.ylabel("Score Values")
    
    plt.figure(10)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of min_samples_split vs Time")
    plt.plot(df['min_samples_split'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval1, timeval, color='green')
    #################################################################################################
    
    '''Finding max_leaf_nodes '''
    
    df = pd.DataFrame(columns=['id', 'max_leaf_nodes', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for numleafs in range(2, 63,10):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0, max_leaf_nodes=numleafs, min_samples_split=int(bestval1), max_depth = maxDepthVal)
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'max_leaf_nodes':numleafs, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train,'Test_score': test_score, 'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./max_leaf_nodes/max_leaf_nodes.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval = df['max_leaf_nodes'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(3)
    plt.plot(df['max_leaf_nodes'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['max_leaf_nodes'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['max_leaf_nodes'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval, bestValScore, color='green')
    
    plt.title("Number of max_leaf_nodes vs Score")
    plt.annotate(f"({bestval}, {bestValScore})", (bestval, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num max_leaf_nodes")
    plt.ylabel("Score Values")
    
    plt.figure(11)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of max_leaf_nodes vs Time")
    plt.plot(df['max_leaf_nodes'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval, timeval, color='green')
    #################################################################################################
    
    ''' Finding min_samples_leaf '''
    
    df = pd.DataFrame(columns=['id', 'min_samples_leaf', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for minLeaf in range(1, 40):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0,min_samples_leaf=minLeaf, max_leaf_nodes=int(bestval), min_samples_split=int(bestval1), max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'min_samples_leaf':minLeaf, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train,'Test_score': test_score, 'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./min_samples_leaf/min_samples_leaf.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval2 = df['min_samples_leaf'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(4)
    plt.plot(df['min_samples_leaf'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['min_samples_leaf'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['min_samples_leaf'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval2, bestValScore, color='green')
    
    plt.title("Number of min_samples_leaf vs Score")
    plt.annotate(f"({bestval2}, {bestValScore})", (bestval2, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num min_samples_leaf")
    plt.ylabel("Score Values")

    plt.figure(12)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of min_samples_leaf vs Time")
    plt.plot(df['min_samples_leaf'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval2, timeval, color='green')
    #################################################################################################1
    
    ''' Finding max_features '''
    
    df = pd.DataFrame(columns=['id', 'max_features', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for maxFea in range(1, 20):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0,max_features=maxFea,min_samples_leaf=int(bestval2), max_leaf_nodes=int(bestval), min_samples_split=int(bestval1), max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'max_features':maxFea, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train,'Test_score': test_score, 'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./max_features/max_features.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval3 = df['max_features'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(5)
    plt.plot(df['max_features'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['max_features'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['max_features'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval3, bestValScore, color='green')
    
    plt.title("Number of max_features vs Score")
    plt.annotate(f"({bestval3}, {bestValScore})", (bestval3, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num max_features")
    plt.ylabel("Score Values")
    

    plt.figure(13)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of max_features vs Time")
    plt.plot(df['max_features'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval3, timeval, color='green')
    #################################################################################################2
    
    ''' Finding min_impurity_decrease '''
    
    df = pd.DataFrame(columns=['id', 'min_impurity_decrease', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for minImpurity in range(0, 6):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=1, random_state=0,min_impurity_decrease=minImpurity,min_samples_leaf=int(bestval2),max_features= int(bestval3), max_leaf_nodes=int(bestval), min_samples_split=int(bestval1), max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'min_impurity_decrease':minImpurity, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'Test_score': test_score,'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./min_impurity_decrease/min_impurity_decrease.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval5 = df['min_impurity_decrease'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(6)
    plt.plot(df['min_impurity_decrease'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['min_impurity_decrease'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['min_impurity_decrease'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval5, bestValScore, color='green')
    
    plt.title("Number of min_impurity_decrease vs Score")
    plt.annotate(f"({bestval5}, {bestValScore})", (bestval5, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num min_impurity_decrease")
    plt.ylabel("Score Values")
    
    plt.figure(14)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of min_impurity_decrease vs Time")
    plt.plot(df['min_impurity_decrease'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval5, timeval, color='green')
    #################################################################################################3
    
    ''' Finding n_estimators & warm_start = False '''
    
    df = pd.DataFrame(columns=['id', 'n_estimators', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for numTree in range(1, 20):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=numTree, random_state=0, warm_start=False,min_impurity_decrease=int(bestval5), min_samples_leaf=int(bestval2),max_features= int(bestval3), max_leaf_nodes=int(bestval), min_samples_split=int(bestval1), max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'n_estimators':numTree, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'Test_score': test_score,'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1

    df.to_csv('./n_estimators_warm_false/n_estimators_warm_false.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval6 = df['n_estimators'][bestValScoreidx]
    id = df['id'][bestValScoreidx]

    plt.figure(7)
    plt.plot(df['n_estimators'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['n_estimators'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['n_estimators'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval6, bestValScore, color='green')
    
    plt.title("Number of n_estimators & warm_start = False vs Score")
    plt.annotate(f"({bestval6}, {bestValScore})", (bestval6, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num n_estimators")
    plt.ylabel("Score Values")

    plt.figure(15)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of n_estimators & warm_start = False vs Time")
    plt.plot(df['n_estimators'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval6, timeval, color='green')
    # #################################################################################################4
    
    ''' Finding n_estimators & warm_start = True '''
    
    df = pd.DataFrame(columns=['id', 'n_estimators', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for numTree in range(1, 60):
        now = datetime.now()
        random_forest = RandomForestClassifier(n_estimators=numTree, random_state=0, warm_start=True,min_impurity_decrease=int(bestval5), min_samples_leaf=int(bestval2),max_features= int(bestval3), max_leaf_nodes=int(bestval), min_samples_split=int(bestval1), max_depth = int(maxDepthVal))
        random_forest.fit(x_train, y_train.ravel())
        end = datetime.now()
        currentModelScore_val = random_forest.score(data_fea[validation_idx, :], data_gnd[validation_idx, :])
        currentModelScore_train = random_forest.score(data_fea[train_idx, :], data_gnd[train_idx, :])
        test_score = random_forest.score(data_fea[test_idx, :], data_gnd[test_idx, :])
        df = df.append({'id':counter, 'n_estimators':numTree, 'Val_score':currentModelScore_val, 'Train_score':currentModelScore_train, 'Test_score': test_score,'train_time(sec)':(end-now).total_seconds()}, ignore_index=True)
        counter += 1
    df.to_csv('./n_estimators_warm_true/n_estimators_warm_true.csv', index=False)
    bestValScoreidx = df['Val_score'].idxmax()
    bestValScore = df['Val_score'][bestValScoreidx]
    bestval7 = df['n_estimators'][bestValScoreidx]
    id = df['id'][bestValScoreidx]
    print(f"best model's value: n_estimators={bestval7}, random_state=0, warm_start=True or False (same result),min_impurity_decrease={int(bestval5)}, min_samples_leaf={int(bestval2)},max_features= {int(bestval3)}, max_leaf_nodes={int(bestval)}, min_samples_split={int(bestval1)}, max_depth = {int(maxDepthVal)}")
    plt.figure(8)
    plt.plot(df['n_estimators'], df['Train_score'], color="red", label="Train_Score")
    plt.plot(df['n_estimators'], df['Val_score'], color="blue", label="Val_score")
    plt.plot(df['n_estimators'], df['Test_score'], color="pink", label="Test_Score")
    plt.scatter(bestval7, bestValScore, color='green')
    
    plt.title("Number of n_estimators & warm_start = True vs Score")
    plt.annotate(f"({bestval7}, {bestValScore})", (bestval7, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num n_estimators")
    plt.ylabel("Score Values")

    plt.figure(16)
    timeval = df['train_time(sec)'][bestValScoreidx]
    plt.title("Number of n_estimators & warm_start = True vs Time")
    plt.plot(df['n_estimators'], df['train_time(sec)'], color="red", label="train_time")
    plt.scatter(bestval7, timeval, color='green')

    # print("Done")
    # # Show Graph
    # plt.show()

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

main()