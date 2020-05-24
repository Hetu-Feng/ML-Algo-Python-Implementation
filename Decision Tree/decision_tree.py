

import pandas as pd
import re, pprint
import numpy as np


# In[349]:


def prepare(file):
    data = pd.read_csv(file)
    data = data.replace(r'[0-9]+: ',value = '', regex = True)
    data= data.replace(r';',value = '',regex = True)
    data = data.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]','',x))
    data = data.rename(columns=lambda x: x.lower())
    data = data.replace(r'^\s+','', regex = True)
    return data


# In[352]:


class Node:
    def __init__(self, df):
        self._df = df                    # The whole dataset
        self._pred = self._df.keys()[-1] # The attribute we want to predict
        self._row_count = self._df.shape[0]
        self._col_count = self._df.shape[1]
        self._node = ''
        self._children = {}
        self._tree_layer = {}
    
    
    
    '''following three functions calculate the information gain'''
    def information(self):
        #This function calculate the entropy of the current dataset
        ent = 0
        pred_values = self._df[self._pred].unique()
        for pred_value in pred_values:
            p_i = self._df[self._pred].value_counts()[pred_value]/self._row_count
            ent += -p_i*np.log2(p_i)
        return ent
        
    
    def entropy(self, attribute):
        #This function calculate the entropy of a chosen attribute
        pred_values = self._df[self._pred].unique() # prediction, basically yes and no, but may have more values
        attr_values = self._df[attribute].unique() # What values the attribute have
        ent_attr = 0
        
        for attr_value in attr_values: # This level iterates through the value of a given attribute
            ent_pred = 0            
            sub_df = self._df[self._df[attribute] == attr_value] #extract a subset of data for a specific value of the attr
            sub_row_count = sub_df.shape[0] # the row of the subset that df[attribute] is a certain value
            
            for pred_value in pred_values: # This level iterates through the value of prediction
                value_count = sub_df[sub_df[self._pred] == pred_value].shape[0] # how many rows have the specific pred_value
                p_i = value_count/sub_row_count+0.000000000000000001 # The probability
                ent_pred += -p_i * np.log2(p_i) #entropy sum up
            
            p_ia = sub_row_count/self._row_count+0.0000000000000001 # The probability that the subset take in the whole dataset
            ent_attr += p_ia * ent_pred # Sum up partial entropy 
        return ent_attr # the entropy of this attribute
        
        
        
    def get_split_node(self):
        # This function calculate the information gain and return the split node
        attributes = self._df.columns.tolist()[:-1]
        igs = []
        info = self.information()
        for attribute in attributes:
            igs.append(info - self.entropy(attribute))  
        self._node = attributes[np.argmax(igs)] # update split node
#        print(igs)
        return self._node
            
                
    '''following functions construct tree structure'''
    
    def get_sub_df(self, value):
        '''sub_df is instances who has the same value on self._node
           since self._node has been used to construct a layer of tree, it won't be used in the future.
           Therefore we remove it from sub_df and pass the copy of the rest sub_df to the next layer
        '''
        return self._df[self._df[self._node] == value].drop(self._node, axis = 1).copy() 
    
    
    
    def split_df(self):
        if self._df.shape[1] > 1:
            node = self.get_split_node()
            values = self._df[node].unique() # values of nodes
            self._tree_layer[node] = {}
            for value in values: # for each value of the node
                sub_df = self.get_sub_df(value) # get the subset of the next layer
                child = Node(sub_df)                
                self._children[node+':'+value], next_layer = child.split_df()
                self._tree_layer[node] = next_layer
            return self._children, self._tree_layer
    
        else:
            self._children[self._pred] = self._df[self._pred].unique()[0]
            return self._children, self._pred
    
    def print_tree(self):
        pprint.pprint(self._children)
    
    def make_pred(self, X:'{attribute: value}'):    
        
        pred_df ={}
        for x in X:
            pred_df[x] = pd.Series(X[x])
        X_d = pd.DataFrame(pred_df)
        X_d = X_d.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]','',x))
        X_d = X_d.rename(columns=lambda x: x.lower())
        
        count = len(X)
        l = self._tree_layer.copy()
        t = self._children.copy()
        for i in range(count):
            for key in l:
                node = key+':'+X_d[key][0]
                try:
                    l = l[key]
                    t = t[node]
                except:
                    t = t[list(t.keys())[0]]
        return t


# In[353]:


data = prepare('dt_data.txt')
tree = Node(data)
tree.split_df()
print("------This is the Decision Tree------\n")
tree.print_tree()


# In[354]:

X = {'occupied':'Moderate', 'price': 'Cheap', 
     'music':'loud', 'location': 'City-Center', 'VIP': 'No', 'favorite beer':'No'}

print("\n------This is test input------")

print(X)

print("\n------Prediction made by the model------")
print(tree.make_pred(X))


# In[337]:




