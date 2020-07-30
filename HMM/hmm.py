
# Team Member: Xiao Yang, Hetu Feng


import numpy as np
from collections import defaultdict



def load_data(file):
    file = open(file)
    result = [] 
    for line in file:
        temp = line.split('\n')
        result.append(temp[:-1])

    grid = np.zeros((10,10))
    tower = np.zeros((4,2))
    noisy = np.zeros((11,4))  
    for i in range(0,10):
        grid[i] = np.array(result[i+2][0].split(' ')).astype(int)

    for i in range(0,4):
        tower[i] = np.array(result[i+16][0].split(' ')[-2:]).astype(int)

    for i in range(0,11):
        temp = result[i+24][0].split(' ')
        if '' in temp:
            temp.remove('')
        noisy[i] = np.array(temp).astype(float)
    return grid, tower, noisy





def euc_dist(x, y):
    distance = [] #
    # top-left, top-right, bottom-left, bottom-right
    for xt, yt in tower:
        dist = np.sqrt( (x-xt)**2 + (y-yt)**2 )
        distance.append(dist)
    return distance





def action_prob(x, y):
    available = []   
    #up
    if ((x-1)>=0) and (grid[x-1][y]==1):
        available.append((x-1,y))
    #down
    if ((x+1)<=9) and (grid[x+1][y]==1):
        available.append((x+1,y))
    #left
    if ((y-1)>=0) and (grid[x][y-1]==1):
        available.append((x,y-1))
    #right
    if ((y+1)<=9) and (grid[x][y+1]==1):
        available.append((x,y+1))
    
    if len(available):
        return available, 1.0/len(available)
    else:
        return available, 0.0





def transition_probability():
    tp = np.zeros((10,10))
    neighbors = {}
    for i in range(10):
        for j in range(10):
            if grid[i][j]:
                neighbors[(i,j)],tp[i][j] = action_prob(i,j)
    
    return tp, neighbors





def cell_emission_probability(distances:'a list of 4 euclidian distance'):
    temporary_list = []
    for d in distances:
        if d == 0:
            temporary_list.append(0)
        else:
            temporary_list.append((1.3 - 0.7) / .1 + 1)
        probability = 1
        for i in temporary_list:
            probability = probability * (1 / i)
            
    return probability





def emission_probability(): #the probability for a hidden having a certain observation, chain of conditional probability
    ep = np.zeros((11,10,10))
    for i in range(10):
        for j in range(10):
            if grid[i][j]:
                distance = euc_dist(i,j) # list of 4 euclidian distance top-left, top-right, bottom-left, bottom-right
                for k,o in enumerate(observation):
                    count = 0
                    for d in range(4):
                        a,b = 0.7*distance[d], 1.3*distance[d]
                        if a<=o[d]<=b:
                            count +=1
                    if count==4:
                        ep[k][i][j] = cell_emission_probability(distance)
                    
    return ep





def viterbi(ep, tp, neighbors):
    viterbi_matrix = np.zeros((11,10,10)) #path probability matrix
    backpointer = defaultdict(dict)
    # initialization step
    for i in range(10):
        for j in range(10):
            if grid[i][j]:
                viterbi_matrix[0][i][j] = ep[0][i][j]
                backpointer[0][(i,j)]=0
    #recursion step
    for timestep in range(1, 11):
        for i in range(10):
            for j in range(10):
                if grid[i][j]:
                    neighbor_list = neighbors[(i,j)]
                    values = np.zeros(len(neighbor_list))
                    index = 0
                    for x, y in neighbor_list:
                        values[index] = viterbi_matrix[timestep-1][x][y]*tp[x][y]*ep[timestep-1][x][y] 
                        index+=1
                    max_index = values.argmax()
                    viterbi_matrix[timestep][i][j] = values[max_index]
                    backpointer[timestep][(i,j)] = neighbor_list[max_index]
    
    #termination step
    bestpathpointer = viterbi_matrix[-1].argmax()
    
    i = bestpathpointer%(10*10)//10
    j = bestpathpointer-i*10
    
    bestpathprob = viterbi_matrix[10][i][j]

    bestpath = [(i,j)]
    for timestep in range(10, 0, -1):
        bestpath.append((backpointer[timestep][bestpath[-1]]))
    
    return bestpath[::-1], bestpathprob




if __name__=='__main__':
    grid, tower, observation = load_data('hmm-data.txt')
    free_states = [(x,y) for x in range(10) for y in range(10) if grid[x][y]==1]
    tp, neighbors = transition_probability()
    ep = emission_probability()
    path, prob= viterbi(ep, tp, neighbors)
    print('The path is:\n',path)




