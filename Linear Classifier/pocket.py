




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('classification.txt', header = None)
df.insert(0, 'x0',1)
learning_rate = 0.01


def get_violated_and_sign(clf,X,w):
    clf['pred'] = np.dot(X,w)
    clf.loc[(clf['pred']>=0), 'pred'] = 1
    clf.loc[(clf['pred']<0), 'pred'] = -1
    violated = not all(clf['pred'] == clf['y'])
    return violated, clf



def pocket(df):
    #initialization
    w = np.random.rand(4)
    
    #data preparation
    clf = df.drop(4, axis = 1).copy()
    clf.rename(columns = {0:'x1',1:'x2', 2:'x3', 3:'y'}, inplace = True)
    X = clf.drop('y', axis = 1)
    
    #first iteration. check if violated constraints exist
    violated, clf = get_violated_and_sign(clf,X,w)
    accuracy = (clf['pred'] == clf['y']).value_counts()[1]/len(clf)
    count = 1
    #the rest iteration
    while violated & (count <7000):
        count += 1
        #pick a vioated constraint i
        i = clf[clf['pred']!= clf['y']].sample(1).iloc[0]
        xi = i[0:4]
        yi = i[4]
        pred = i[5]
        #case 1
        if(pred <0) & (yi == 1):
            w = w + learning_rate * xi
        #case 2
        elif(pred >=0) & (yi == -1):
            w = w - learning_rate * xi
        
        #check if violated constraints exist
        X = clf.drop(['y', 'pred'], axis = 1)
        violated, clf= get_violated_and_sign(clf,X,w)        
        accuracy_this = (clf['pred'] == clf['y']).value_counts()[1]/len(clf)
        
        accuracy=max(accuracy, accuracy_this)
    return np.array(w), accuracy, count



w, accuracy, iteration = pocket(df)
print('In Pocket learning algorithm\nIteration time: {}\nWeights: {}\nAccuracy: {}'
     .format(iteration, w, accuracy))



