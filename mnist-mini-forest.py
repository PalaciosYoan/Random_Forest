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
    plt.savefig('./figures/1.png')
    

    #################################################################################################
    
    '''Finding min_samples_split can be between 4 & 2 or 24(this one is the best of best, iterations of 39)'''
    df = pd.DataFrame(columns=['id', 'min_samples_split', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for minSplit in range(2, 13):
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
    plt.savefig('./figures/2.png')
    
    #################################################################################################
    
    '''Finding max_leaf_nodes '''
    
    df = pd.DataFrame(columns=['id', 'max_leaf_nodes', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for numleafs in range(2, 43):
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
    plt.savefig('./figures/3.png')

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
    plt.savefig('./figures/4.png')

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
    plt.savefig('./figures/5.png')

    #################################################################################################2
    
    ''' Finding min_impurity_decrease '''
    
    df = pd.DataFrame(columns=['id', 'min_impurity_decrease', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for minImpurity in range(0, 30, 2):
        minImpurity = minImpurity/100
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
    plt.savefig('./figures/6.png')

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
    plt.scatter(bestval6, df['Test_score'][bestValScoreidx], color='green')
    plt.annotate(f"({bestval6}, {df['Test_score'][bestValScoreidx]})", (bestval6, df['Test_score'][bestValScoreidx]))
    
    plt.title("Number of n_estimators & warm_start = False vs Score")
    plt.annotate(f"({bestval6}, {bestValScore})", (bestval6, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num n_estimators")
    plt.ylabel("Score Values")
    plt.savefig('./figures/7.png')


    # #################################################################################################4
    
    ''' Finding n_estimators & warm_start = True '''
    
    df = pd.DataFrame(columns=['id', 'n_estimators', 'Val_score', 'Train_score', 'train_time(sec)', 'Test_score'])
    counter = 1 #this is the unique id to identify each model, not needed but added cause why not
    for numTree in range(1, 20):
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
    plt.scatter(bestval7, df['Test_score'][bestValScoreidx], color='green')
    plt.annotate(f"({bestval7}, {df['Test_score'][bestValScoreidx]})", (bestval7, df['Test_score'][bestValScoreidx]))
    plt.title("Number of n_estimators & warm_start = True vs Score")
    plt.annotate(f"({bestval7}, {bestValScore})", (bestval7, bestValScore))
    plt.legend(loc="lower right")
    plt.xlabel("Num n_estimators")
    plt.ylabel("Score Values")
    plt.savefig('./figures/8.png')

    # Show Graph
    plt.show()

main()