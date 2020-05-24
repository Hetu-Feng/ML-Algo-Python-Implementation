


import pandas as pd
import re, pprint
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# In[10]:


data = pd.read_csv('dt_data.txt')
data = data.replace(r'[0-9]+: ',value = '', regex = True)
data= data.replace(r';',value = '',regex = True)
data = data.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]','',x))
data = data.replace(r'^\s+','', regex = True)


data["Occupied"].replace({"High":3, "Moderate":2, "Low":1}, inplace = True)
data["Price"].replace({"Expensive":3, "Normal":2, "Cheap":1}, inplace = True)
data["Music"].replace({"Loud":1, "Quiet":0}, inplace = True)
data["Location"].replace({"City-Center":1,"Mahane-Yehuda":2, "Talpiot":3, "Ein-Karem":4,"German-Colony":5}, inplace=True)
data["VIP"].replace({"Yes":1, "No":0},inplace = True)
data["FavoriteBeer"].replace({"Yes":1, "No":0},inplace = True)
data["Enjoy"].replace({"Yes":1, "No":0},inplace = True)



feature_col = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'FavoriteBeer']
x_train = data[feature_col]
y_train = data.Enjoy

clf = DecisionTreeClassifier()
# Decision Tree Classifer
clf = clf.fit(x_train,y_train)


# In[14]:


# test data
data_t = [{"Occupied":"Moderate", "Price":"Cheap", "Music":"Loud", "Location":"City-Center", "VIP": "No", "FavoriteBeer":"No"}]
data_test = pd.DataFrame(data_t)  

#clean test data
data_test["Occupied"].replace({"High":3, "Moderate":2, "Low":1}, inplace = True)
data_test["Price"].replace({"Expensive":3, "Normal":2, "Cheap":1}, inplace = True)
data_test["Music"].replace({"Loud":1, "Quiet":0}, inplace = True)
data_test["Location"].replace({"City-Center":1,"Mahane-Yehuda":2, "Talpiot":3, "Ein-Karem":4,"German-Colony":5}, inplace=True)
data_test["VIP"].replace({"Yes":1, "No":0},inplace = True)
data_test["FavoriteBeer"].replace({"Yes":1, "No":0},inplace = True)

#make prediction
feature= ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'FavoriteBeer']
x_test = data_test[feature]
pred = clf.predict(x_test)

print("------Predicted result by Scikit-Learn DecisionTreeClassifier-------")
print(pred)


# In[ ]:




