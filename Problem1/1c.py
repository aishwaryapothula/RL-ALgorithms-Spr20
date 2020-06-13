#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
import matplotlib as plt


# In[99]:


class Bandit:
    def __init__(self):

        self.rewards = []

    # s -start floor, e - exit floor, f - floor elevator is on when no one calls
    # s_prob - probability of start floor, es_prob - probability of exit floor given start floor
    
    # utility as a function of se, sc and waiting time. The summation for each floor happens in run_experiment
    def utility(self, s_prob, es_prob, s, e ,f):
        r = - (s_prob * es_prob *((abs(s - f) + abs(s - e) + 1)* 7))
        return r

   


# In[100]:


def run_experiment(case):
    
    # creating a list of objects of class Bandit to represent each floor
    bandits = [Bandit(), Bandit(), Bandit(), Bandit(), Bandit(), Bandit()]
    
    # creating dictionary to deal with indexing and floor numbering differences while calculating utility index:floor
    d = {0:1, 1:2 ,2:3 ,3:4 ,4:5 ,5:6}
    
    
    if case == 'morning':
        b_sums = []
        
        # for each floor f calculating the penalties for all combinations of start and exit floors
        for f in range(6):
            s = 0
            s_prob = 1
            es_prob = 1/5
            p_vals = []
            
            for i in [1,2,3,4,5]:
                
                e = i
                # calculating penalty
                x = bandits[f].utility(s_prob, es_prob, d[s], d[e], d[f])
                # collection of penalties of all combinations of start and exit floors for each floor
                p_vals.append(x)
                
            # summing the penalties of all combinations of start and exit floors for each floor    
            pp = sum(p_vals)
            
            # list of penalties of all 6 floors 
            b_sums.append(pp)
            
        return b_sums
                
    if case == 'noon':
        b_sums = []
        
        # for each floor f calculating the penalties for all combinations of start and exit floors
        for f in range(6):
            e = 0
            es_prob = 1
            s_prob = 1/5
            p_vals = []
            
            for i in [1,2,3,4,5]:
                
                s = i
                # calculating penalty
                x = bandits[f].utility(s_prob, es_prob, d[s], d[e], d[f])
                # collection of penalties of all combinations of start and exit floors for each floor
                p_vals.append(x)
                
            # summing the penalties of all combinations of start and exit floors for each floor
            pp = sum(p_vals)
            
            # list of penalties of all 6 floors 
            b_sums.append(pp)
            
        return b_sums

    if case == 'afternoon':

        c1_penalties = []
        c2_penalties = []

        for f in range(6):
            
            s = 1
            s_prob = 1
            es_prob = 1/5
            p1_vals = []
            
            for i in [0,2,3,4,5]:
                e = i
                # calculating penalty
                x = bandits[f].utility(s_prob, es_prob, d[s], d[e], d[f])
                # collection of penalties of all combinations of start and exit floors for each floor
                p1_vals.append(x)
                
            # summing the penalties of all combinations of start and exit floors for each floor
            pp1 = sum(p1_vals)
                
            # list of penalties of all 6 floors under case1    
            c1_penalties.append(pp1)
                
        for f in range(6):
            
            e = 0
            es_prob = 1
            s_prob = 1/5
            p2_vals = []
            
            for i in [1,2,3,4,5]:
                
                s = i
                # calculating penalty
                x = bandits[f].utility(s_prob, es_prob, d[s], d[e], d[f])
                # collection of penalties of all combinations of start and exit floors for each floor
                p2_vals.append(x)
                
            # summing the penalties of all combinations of start and exit floors for each floor    
            pp2 = sum(p2_vals)   
            
            # list of penalties of all 6 floors under case2 
            c2_penalties.append(pp2)
            
        # doing an average to represent 50% chance of either case happening  
        # combined penalties of all 6 floors
        combined_penalties = (np.array(c1_penalties) + np.array(c2_penalties))/2
        
        return combined_penalties 


# In[101]:


c_1 = run_experiment('afternoon')


# In[102]:


print(c_1)
floor = np.argmax(c_1)
print("Ideal floor for the elevator to be on in this case is {}". format(floor + 1))


# In[ ]:




