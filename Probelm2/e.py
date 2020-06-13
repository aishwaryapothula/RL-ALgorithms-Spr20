# Aishwarya Pothula - 1001743470

import numpy as np
import matplotlib.pyplot as plt


class GridWorldEnvironment:
    def __init__(self, l, b, obstacles, target, start, orientation):
        self.l = l
        self.b = b
        self.obstacles = obstacles
        self.target = target
        self.start = start
        self.orientation = orientation
        self.table = self.create_table()
        self.nt_states, self.t_state = self.t_nt_states(self.table)
        self.observations = []
        self.observation = [start, orientation]
        self.prev_state, self.new_state, self.state = None, None, None
        self.pos = [[0, 0], "n"]
        
    # defining terminal and non-terminal states
    def t_nt_states(self, table):
        tab = np.argwhere(table)
        t = tab[-1]
        nt = np.delete(tab, len(tab) - 1, axis=0)
        return nt, t
        
        
    # creates tabels of specifies shapes
    # creates an initial version of the environemnt by displaying a numpy array
    # in which 7 represents target, 2 represents start state, 5 represents obstacles
    def create_table(self):
        table = np.ones((self.l, self.b))
        table[self.target[0], self.target[1]] = 7
        table[self.start[0], self.start[1]] = 2
        for obstacle in self.obstacles:
            table[obstacle[0], obstacle[1]] = 5
        return table

    def step(self, a):
        obv, neg, done = self.move(a, self.start, self.orientation)
        reward = self.reward(neg, done)
        state = self.obv_to_state(obv)
        return state, reward, done
        
        
    # splitting the ins struction received by "_" coz ward actions and turn actions have common updations to the position of the agent
    def move(self, ins, cord, ort):
        import random

        r = np.random.uniform()
        where, how = ins.split("_")[0], ins.split("_")[1]
        cord, ort = self.pos
        
        # updating position according to unreliability probability, obstacles and out of grid
        if how == "w":
            idx = {"f": 0, "b": 1}
            # designing grid movement. movement based on grid indices not x,y axes
            mv = {
                "n": [[cord[0] - 1, cord[1]], [cord[0] + 1, cord[1]]],
                "e": [[cord[0], cord[1] + 1], [cord[0], cord[1] - 1]],
                "w": [[cord[0], cord[1] - 1], [cord[0], cord[1] + 1]],
                "s": [[cord[0] + 1, cord[1]], [cord[0] - 1, cord[1]]],
            }
          
            new_cord = mv[ort][idx[where]]
            new_pos = [new_cord, ort]
            new_pos = random.choices(
                population=[new_pos, self.pos], weights=[0.8, 0.2]
            )[0]
           
            cleaned_nts = [list(i) for i in self.nt_states]
            
            if new_pos[0] in self.obstacles or new_pos[0] not in cleaned_nts:
                condtarget = np.argwhere(self.table == 7)[0]
                
              
                new_pos_ar = np.asarray(new_pos[0])
               
                if np.array_equal(new_pos_ar,condtarget) == False:
                   
                    return self.pos, True, False
                   
                    
                if np.array_equal(new_pos_ar,condtarget) == True:
                    self.pos = new_pos
                    return new_pos, False, True
            else:
                self.pos = new_pos
                return self.pos, False, False
                
        # for turn actions, changing orientation according to unreliability probability
        if how == "t":
            idx = {"l": 0, "r": 1}
            mv = {"n": ["w", "e"], "s": ["e", "w"], "e": ["n", "s"], "w": ["s", "n"]}
            new_ort = mv[self.pos[1]][idx[where]]
            new_pos = [self.pos[0], new_ort]
            new_pos = random.choices(population=[new_pos, self.pos], weights=[0.9, 0.2])
            self.pos = new_pos[0]
            return new_pos[0], False, False
            
    # assigning observations to states
    def obv_to_state(self, new_pos):
        orts = {"n": 0, "e": 1, "w": 2, "s": 3}
        [x, y], ort = new_pos
        state_id = x + (y * self.l) + (self.b * self.l * orts[ort])
        return state_id
        
    # generates rewards according to discussed policy.
    # 2nd parameters specifies reward for out of grid or obstacle
    # 3rd specifies parameters for target position reward
    def reward(self, neg, done):
        if done == True:
            reward = 100
        if neg == True:
            reward = -100
        if neg == False and done == False:
            reward = 0
        return reward
        
    # resets the position of the agent to the starting postion
    def reset(self):
       self.prev_state, self.new_state, self.state = None, None, None
       self.pos = [[0, 0], "n"]
