#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random


class Bandit:
    def __init__(self):
        
        # The mean calculated for the bandit till now
        self.mean = 0
        
        # The number of times a bandit/floor is selected
        self.N = 0
        
    # function that calculates the utility s- start floor, e - exit floor, f - elevator floor
    def pull(self, s, e, f):
        
        # quadratic utility/penalty, probabilities are taken care of in the run_experiment section
        r = -np.power((abs(s - f) + abs(s - e) + 1)* 7, 2)
        return r

    def update(self, x):
        
        # update the number of selection of each floor
        self.N += 1
        
        # update the mean including the new reward x using incremental averaging
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
 
def run_experiment_eps(eps, N):
    
    random.seed(5)
    
    # creating a list of objects of class Bandit to represent each floor
    bandits = [Bandit(), Bandit(), Bandit(), Bandit(), Bandit(), Bandit()]
    
    # creating dictionary to deal with indexing and floor numbering differences index:floor
    d = {0:1, 1:2 ,2:3 ,3:4 ,4:5 ,5:6}
   
    # list containing list of lists of rewards for each action (floor selected) over all iterations
    points = []
 
    for i in range(N):
        # start floor is uniformly selected
        s = np.random.choice([0,1,2,3,4,5], p = [(1/6), (1/6), (1/6), (1/6), (1/6), (1/6)])
        
        # probability of exit floor given start floor is floor no * p, 0 when same as start floor
        # probability value of exiting at each floor given starting floor is calculated and given
        if s == 0:
            e = np.random.choice([0,1,2,3,4,5], p = [0, (2/20), (3/20), (4/20), (5/20), (6/20)])
        if s == 1:
            e = np.random.choice([0,1,2,3,4,5], p = [(1/19), 0, (3/19), (4/19), (5/19), (6/19)])
        if s == 2:
            e = np.random.choice([0,1,2,3,4,5], p = [(1/18), (2/18), 0, (4/18), (5/18), (6/18)])
        if s == 3:
            e = np.random.choice([0,1,2,3,4,5], p = [(1/17), (2/17), (3/17), 0, (5/17), (6/17)])
        if s == 4:
            e = np.random.choice([0,1,2,3,4,5], p = [(1/16), (2/16), (3/16), (4/16), 0, (6/16)])
        if s == 5:    
            e = np.random.choice([0,1,2,3,4,5], p = [(1/15), (2/15), (3/15), (4/15), (5/15), 0])
        
        # implementation of epsilon-greedy algorithm to select elevator floor
        p = np.random.uniform()
        if p < eps:

            # randomly choose to pull one of the six bandits/floors when random no less than epsilon
            j = np.random.choice([0,1, 2, 3, 4, 5], p = [(1/6), (1/6), (1/6), (1/6), (1/6), (1/6)])
        else:

            # choose the bandit/floor whose mean reward is highest till now
            j = np.argmax([b.mean for b in bandits])

        # pass values of start, exit, elevator floor to calculate utility
        x = bandits[j].pull(d[s], d[e], d[j])
        
        # updates no of selections N and mean reward for each bandit/floor
        bandits[j].update(x)

        # mean estimates for each floor/bandit an iteration
        b_points = [b.mean for b in bandits]
        
        # to store all the floor mean estimates for every iteration
        points.append(b_points)

    return points 


# run elevator scheduling for 500 iterations with epsilon value 0.1
c_1 = run_experiment_eps(0.1, 500)

# list of lists containing all rewards of a partcular floor over all iterations. 6 lists representing each floor
pl = [[], [], [], [], [], []]

# code to generate a list of lists. Each lists inside the list contains all the rewards for a particular floor over 500 iterations
for l in range(len(c_1)):
    for m in range(len(c_1[0])):
        if m == 0:
            pl[m].append(c_1[l][m])
        if m == 1:
            pl[m].append(c_1[l][m])
        if m == 2:
            pl[m].append(c_1[l][m])
        if m == 3:
            pl[m].append(c_1[l][m])
        if m == 4:
            pl[m].append(c_1[l][m])
        if m == 5:
            pl[m].append(c_1[l][m])

            
# plotting rewards of each floor/action over all the iterations            
plt.plot(pl[0], label = 'floor 1')
plt.plot(pl[1], label = 'floor 2')
plt.plot(pl[2], label = 'floor 3')
plt.plot(pl[3], label = 'floor 4')
plt.plot(pl[4], label = 'floor 5')
plt.plot(pl[5], label = 'floor 6')
plt.legend()
