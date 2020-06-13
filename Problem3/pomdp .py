import random, sys, math 
import numpy

import random

class POMDP:
    def __init__(self, q_table, epsilon, columns, rows, actions , orts, obstacle, belief_hash, bel, b_tmp, episodes, alpha, gamma):

        self.q_table = q_table
        self.epsilon = epsilon
        self.columns = columns
        self.rows = rows
        self.actions = actions
        self.orts = orts
        self.obstacle = obstacle
        self.cos = {}
        self.sin = {}
        self.belief_hash = belief_hash 
        self.prob = prob
        self.episodes = episodes
        self.nn_state = None

        self.alpha = alpha
        self.gamma = gamma
        self.set_rotations()

    def set_rotations(self):
        self.cos[0], self.cos[90], self.cos[180], self.cos[270], self.sin[360] = 1, 0, -1, 0, 1
        self.sin[0], self.sin[90], self.sin[180], self.sin[270], self.sin[360]= 0, 1, 0, -1, 0
        return

    def reward(self, state):
        if self.obstacle == 0:
            if (state[0] < 0 or state[0] >= self.rows) or (state[1] < 0 or state[1] >= self.columns):
                return -100
            elif state[0] == self.rows - 1 and state[1] == self.columns - 1:
                return 100
            else:
                return 0
        else:
            if (state[0] < 0 or state[0] >= self.rows) or (state[1] < 0 or state[1] >= self.columns):
                return -100
            elif state[1] == self.columns - 1 and state[0] == self.rows - 1:
                return 100
            else:
                if (state[1] <= 4 and state[1] >= 3) and (state[0] <= 4 and state[0] >= 3):
                    return -100
                else:
                    return 0
    
    def right_turn(self, angle, radian, state):

        angle = angle + 270
        if angle >= 360:
            angle = angle - 360
        radian = angle
        next_state= [act + bi for act, bi in zip(state, [self.cos[radian], self.sin[radian],0])]
        next_state[2] = int(round(angle/90))
        reward = self.reward(next_state)

        return next_state, reward 

    def left_turn(self, angle, radian, state):
        angle = angle + 90
        if angle >= 360:
            angle = angle - 360
        radian = angle
        next_state= [act + bi for act, bi in zip(state, [self.cos[radian], self.sin[radian],0])]
        next_state[2] = int(round(angle/90))
        reward = self.reward(next_state)

        return next_state, reward 

    def forward(self, radian, state):
        next_state = [act + bi for act, bi in zip(state, [self.cos[radian], self.sin[radian],0])]
        reward = self.reward(next_state)
        return next_state, reward

    def backward(self, angle, radian, state):
        angle = angle + 180
        if angle >= 360:
            angle = angle - 360
        radian = angle
        next_state = [act + bi for act, bi in zip(state, [self.cos[radian], self.sin[radian],0])]
        next_state[2] = int(round(angle/90))
        reward = self.reward(next_state)

        return next_state, reward

    def env(self, state, action):
        angle = state[2] *90
        radian = angle
        r = random.random()

        if action == 0:
            if r < 0.2:
                return (state, 0)
            else:
                next_state, reward = self.forward(radian, state)
                reward = self.reward(next_state)
                if reward == -100:
                    return (state, reward)
                else:
                    return (next_state, reward)

        if action == 1:
            if r < 0.1:
                return (state, 0)
            else:
                next_state, reward = self.left_turn(angle, radian, state)    
                if reward == -100:
                    return (state, reward)
                else:
                    return (next_state, reward)

        if action == 2: 

            if r < 0.2:
                return (state, 0)

            else:
                next_state, reward = self.backward(angle, radian, state)   
                if reward == -100:
                    return (state, reward)
                else:
                    return (next_state, reward)

        if action == 3: 
            if r < 0.1:
                return (state, 0)
            else:
                next_state, reward = self.right_turn(angle, radian, state)
                if reward == -100:
                    return (state, reward)
                else:
                    return (next_state, reward)

    def take_action(self, state):
        random_number = random.random()
        if random_number < self.epsilon: return random.randint(0, actions - 1)
        else:
            s, q_table_max = self.q_table[state[0]][state[1]][state[2]][:], self.q_table[state[0]][state[1]][state[2]][:][0]

            for ix in list(range(actions)):
                if q_table_max <= s[ix]:
                    q_table_max = s[ix]

            index_max = [ix for ix in list(range(actions)) if s[ix] == q_table_max]
            pick = random.randint(0, len(index_max) - 1)
            return index_max[pick]

    def transition_prob(self, nn_state, state, action):
        # p_s_prime_given_s_and_a
        angle, radian = state[2] * 90, state[2] * 90
    
        for ix in list(range(len(state))):
            if ix == 0 and (nn_state[ix] > (rows - 1)  or state[ix] < 0 or nn_state[ix] < 0 or  state[ix] > (rows - 1)):
                return 0
            if ix == 1 and (nn_state[ix] > (columns - 1) or state[ix] < 0 or nn_state[ix] < 0  or state[ix] > (columns - 1)):
                return 0

        if action == 0:
            next_state, reward = self.forward(radian, state)
            if reward == -100:
                next_state = state

        if action == 1:
            next_state, reward = self.left_turn(angle, radian, state)
            if reward == -100:
                next_state = state

        if action == 2:
            next_state, reward = self.backward(angle, radian, state)
            if reward == -100:
                next_state = state

        if action == 3:
            next_state, reward = self.right_turn(angle, radian, state)
            if reward == -100:
                next_state = state

        if (action == 2 or action == 0) and (next_state == nn_state and next_state != state):
            prob = 0.8
        elif (action == 3 or action == 1) and (next_state == nn_state and next_state != state):
            prob = 0.9
        elif (action == 2 or action == 0) and (state == nn_state and next_state != state):
            prob = 0.2
        elif (action == 3 or action == 1) and (state == nn_state and next_state != state):
            prob = 0.1
        elif next_state == state and next_state == nn_state:
            prob = 1
        else:
            prob = 0
        return prob

    def sigma_transition(self, s,action):
        sigma = 0
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (0, 1, 2, 3):
                    state_index = [s[0] + i, s[1] + j, k]
                    prob = self.transition_prob(s, state_index, action)
                    if  prob != 0.0:
                        sigma += (prob * self.belief_hash.get((state_index[0], state_index[1], state_index[2]),0))
        return sigma

    def update_belief_states(self, s, nn_state, action):

        pi = self.sigma_transition(nn_state ,action) * self.prob
        states_array = self.get_states(s, nn_state,action)
        ns = len(states_array)

        if ns != 0:
            p = (1 - self.prob) / float(ns)
        else:
            pi = self.sigma_transition(nn_state, action)
            p = 0
        
        p_list = [p * self.sigma_transition(states_array[s], action) for s in range(ns)]

        summation = float(sum(p_list, pi))

        self.belief_copy = {}

        for s in list(range(ns)):
            self.belief_copy[(states_array[s][0], states_array[s][1], states_array[s][2])] = float(p_list[s]) / summation

        self.belief_copy[(nn_state[0], nn_state[1], nn_state[2])] = float(pi)/summation

        self.belief_hash = self.belief_copy

        return

    def q_table_state_actions(self, action):
        postion = list(self.belief_hash.keys())
        array = list(self.belief_hash.values())

        q_sum = 0

        for ix in list(range(len(postion))):
            q_sum += self.q_table[postion[ix][0]][postion[ix][1]][postion[ix][2]][action] * array[ix] 
        return q_sum

    def go_to_next_state(self, state, action):

        angle = state[2] * 90
        radian = angle

        for ix in list(range(len(state))):
            if ix == 0 and (state[ix] < 0 or self.nn_state[ix] < 0 or self.nn_state[ix] > (rows-1) or state[ix] > (rows-1)):
                return 0
            if ix == 1 and (state[ix] < 0 or self.nn_state[ix] < 0  or self.nn_state[ix] > (columns-1) or state[ix] > (columns-1)):
                return 0

        if action == 0:
            next_state, reward = self.forward(radian, state)
            if reward == -100:
                next_state = state

        if action == 1: # action turn left
            next_state, reward = self.left_turn(angle, radian, state)

            if reward == -100:
                next_state = state

        if action == 2: # action backward
            next_state, reward = self.backward(angle, radian, state)
            if reward == -100:
                next_state = state

        if action == 3: # action turn right
            next_state, reward = self.right_turn(angle, radian, state)
            if reward == -100:
                next_state = state

        return next_state

    def get_states(self, state, nn_state, action):

        array = []

        if state != nn_state:
            array.append(state)

        for act in list(range(self.actions)):
            s = self.go_to_next_state(state, act)
            if s != nn_state and s not in array:
                array.append(s)

        for ix in list(range(len(array))):
            for j in range(4):
                s = self.go_to_next_state(array[ix], j)
                if  s != nn_state and s not in array:
                    array.append(s)

        for ix in list(range(self.actions)):
            s = self.go_to_next_state(nn_state, ix)
            if s != nn_state and s not in array:
                array.append(s)

        return array


    def train(self):
        current_state = init_state
        episode = 0
        while self.episodes < 2000:
            episode = episode + 1
            step = 0
            goal = 0
            rewards = []
            current_state = init_state
            self.belief_hash = {}
            self.belief_hash[(0,0,0)] = 1
            while goal == 0:
                action = self.take_action(current_state)
                out_of_env = self.env(current_state, action)
                self.nn_state = out_of_env[0]
                reward = out_of_env[1]
                # print(reward)
                rewards.append(reward)
                q_state = self.q_table_state_actions(action)
                oh_belief =  self.belief_hash[(current_state[0],current_state[1],current_state[2])]

                self.update_belief_states(current_state, self.nn_state, action)

                nsa = [self.q_table_state_actions(ix) for ix in range(4)]
                act = self.q_table[current_state[0]][current_state[1]][current_state[2]][action]
                self.q_table[current_state[0]][current_state[1]][current_state[2]][action] = act + (reward + self.gamma * max(nsa) - q_state) * oh_belief * self.alpha

                step = step + 1

                current_state = self.nn_state

                if reward == 100:
                    goal = 1
                    print("episode", episode, "steps", step, "reward", sum(rewards)/step)

if __name__ == "__main__":
    columns, rows = 25, 15 
    obstacle = 1 
    actions, orts = 4, 4

    init_state = [0,0,0]
    episodes = 30
    step, goal = 0, 0
    alpha, gamma, epsilon = 0.1,  0.95, 0.01
    prob = 0.7

    belief_hash = {}
    q_table = numpy.zeros((rows,columns,orts,actions))

    model = POMDP(
        q_table=q_table,
        epsilon=epsilon,
        columns=columns, 
        rows=rows, 
        actions=actions, 
        orts=orts, 
        obstacle=obstacle, 
        bel={}, 
        belief_hash=belief_hash, 
        episodes=episodes, 
        b_tmp={}, 
        alpha=alpha, 
        gamma=gamma)
    model.train()
