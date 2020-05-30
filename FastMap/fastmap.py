



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_df(file):
    data = np.loadtxt(file)
    df = {}
    for i in data:
        a, b, d = i[0], i[1], i[2]
        if a in df.keys():
            df[a][b] = i[2]
        else:
            df[a] = {}
            df[a][b] = i[2]
        if b in df.keys():
            df[b][a] = i[2]
        else:
            df[b] = {}
            df[b][a] = i[2]
    return data, df


def x_i(a,b,i, dist_function):
    xi = (dist_function(a,i)**2+dist_function(a,b)**2-dist_function(b,i)**2)/2/dist_function(a,b)
    return xi



def other_i(a,b,df):
    result = list(df.keys())
    result.remove(a)
    result.remove(b)
    return result



def all_x_i(a,b,df, dist_function,coordinate, dist):
    others = other_i(a,b,df)
    for i in others:
        if i in coordinate.keys():
            coordinate[i].append(x_i(a,b,i,dist_function))
        else:
            coordinate[i] = [x_i(a,b,i,dist_function)]
    if a not in coordinate.keys():
        coordinate[a] = [0]
    else:
        coordinate[a].append(0)
    if b not in coordinate.keys():
        coordinate[b] = [dist]
    else:
        coordinate[b].append(dist)
    return coordinate



def find_a_b(a, dist_function, df,counter):  
    pairs = list(df[a].keys())
    dist_dict = {}
    for j in pairs:
        dist_dict[j] = dist_function(a,j)
#        print("pair {},{} has distance {}: ".format(a,j, dist_dict[j]))
    max_dist = max(dist_dict.values())
    for p, d in dist_dict.items():
        if d == max_dist:
            b = p
            break
#    print('current b is {}, distance is {}'.format(b, max_dist))
    if counter == 4:
        return a, b, max_dist
    else:
        return find_a_b(b, dist_function, df,counter+1)



def first_function(a,b):
    return df[a][b]



def fastmap(df,dist_function,k, coordinate):
    counter = 0
    
    if k == 2:
        return coordinate
    else:
        
        a = float(np.random.randint(10)+1)
        a, b, dist = find_a_b(a,dist_function, df, counter)
        x_i = all_x_i(a,b,df,dist_function,coordinate,dist)
        def new_dist_function(i, j):
            if (i == a) & (j == b):
                return 0
            elif (i == b) & (j == a):
                return 0
            else:
                return (dist_function(i,j)**2 - (x_i[i][k]-x_i[j][k])**2)**0.5
        
        return fastmap(df, new_dist_function, k+1, x_i)



data,df = generate_df('fastmap-data.txt')
X = fastmap(df, first_function, 0, {})
fp = open('fastmap-wordlist.txt','r')
print(X)
words = []
for line in fp.readlines() :
	word = line.strip()
	words.append(word)
for i in range(1,11):
	plt.plot(X[i][0], X[i][1],'xk') 
	plt.annotate(words[i-1], xy = (X[i][0], X[i][1]))

print('Graph saved to folder as words.png')
plt.savefig('words.png')




