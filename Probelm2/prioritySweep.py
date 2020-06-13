# Aishwarya Pothula - 1001743470

import numpy as np
import matplotlib.pyplot as plt
# importing the environment file
import e

l = 15
b = 25
obstacles = [[2, 6], [6, 12], [12, 13], [7, 9]]
target = [14, 24]
start = [0, 0]
orientation = "n"

alpha = 0.1
gamma = 0.95  # discount
epsilon = 0.005
theta = 0.0001
max_steps = 500
trials = 1


class PQ:
    
    def __init__(self,theta,n,alpha,gamma,epsilon,max_steps,l,b,obstacles,target,start,orientation):
        
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.theta = theta
        self.max_steps = max_steps
        self.l = l
        self.b = b
        self.obstacles = obstacles
        self.target = target
        self.start = start
        self.orientation = orientation
        self.steps = 0
        self.its = []
        self.p = 0
        self.q_table = np.zeros((self.l * self.b * 4, 4))
        self.m_table = self.create_model(self.l * self.b * 4, 4)
        # the priority queue's datastructure
        self.pq = [{"p":0,"SA":[0,10]}]
        self.predecessor = []
        self.env = e.GridWorldEnvironment(l=self.l,b=self.b,obstacles=self.obstacles,target=self.target,start=self.start,
            orientation=self.orientation)
        
    def create_model(self, n_states, n_actions):
        self.rewards = np.zeros((n_states, n_actions))
        self.transitions = np.zeros((n_states, n_actions))

    def learn(self):
        s = 0
        reward_list = []
        cum_reward = [0]

        for _ in range(max_steps):
            p = 0
            self.steps += 1
            a_dic = {"f_w": 0, "b_w": 1, "l_t": 2, "r_t": 3}
            b = self.choose_action(s)
            s_prime, reward, done = self.env.step(b)
            a = a_dic[b]
            self.add_to_model(s, a, s_prime, reward)
            self.track_list(s, a, s_prime, reward)
            p = abs(reward + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[s][a])
            self.add_to_pq(s,a,p)
            self.planning(p)
            s = s_prime

            if done == True:
                print("episode")
                self.its.append(self.steps)
                self.steps = 0
                self.env.reset()
                s = 0

            cum_reward.append(cum_reward[-1] + reward)
        
        return np.array(cum_reward[1:])


    # adds s,a along with priority to the priority queue
    def add_to_pq(self,s,a,p):

        ff, i = self.ptheta_check(s)
        if p > self.theta and len(self.pq) == 0:
            self.pq.append({"priort": p, "SA":[s, a]})

        if p > self.theta and ff == "a":
            if self.pq[0]["SA"][1] == 10:
                self.pq.pop(0)
            self.pq.append({"priort": p, "SA":[s, a]})
            self.desc_sort()
            
        # if s = s then replace a, p
        if p > self.theta and ff == "b":
            self.pq[i]["SA"][1] = a
            self.pq[i]["priort"] = p
            self.desc_sort()

    def desc_sort(self):

        self.pq = sorted(self.pq, key=lambda k: k["priort"], reverse = True)


    # to check if s alreadye exists
    def ptheta_check(self,s ):
        if len(self.pq ) > 0:

            for i in range(len(self.pq)):
                if s != self.pq[i]["SA"][0]:
                    return i, "a"

                if s == self.pq[i]["SA"][0]:
                    return i, "b"
                

    
    def planning(self,p): 
        for _ in range(self.n):
            while(self.pq and self.pq[0]["SA"][1] != 10):
                sa_pop = self.pq.pop(0)
                s = sa_pop["SA"][0]
                a = sa_pop["SA"][1]
                reward = int(self.rewards[s][a])
                s_prime = int(self.transitions[s][a])
                self.q_table[s][a] = self.q_table[s][a] + self.alpha * (reward + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[s][a])
                sar = self.match(s)
                for each in sar:
                    r_bar = each[2]
                    if each[0] == -1 and each[1] == -1:
                        p = abs(reward + self.gamma * np.max(self.q_table[s_prime]))
                    else:
                        p = abs(reward + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[each[0]][each[1]])
                    self.add_to_pq(each[0], each[1], p)



    def choose_action(self, state):
        r = np.random.uniform()
        a_dic = {0: "f_w", 1: "b_w", 2: "l_t", 3: "r_t"}
        if r < self.eps:
            action = np.random.choice(["f_w", "b_w", "l_t", "r_t"])
        else:
            action = a_dic[np.argmax(self.q_table[state])]
        return action


    def add_to_model(self,s, a, s_prime, reward):
        self.transitions[s][a] = s_prime
        # self.transitions[s][-1] = 1
        # self.action_table[s][a] = 1
        self.rewards[s][a] = reward
    # to get a list of dictionaries containing the trajectory for each state
    def track_list(self, s, a, s_prime, reward):
        i = 0
        for i in range(len(self.predecessor)):
            self.predecessoring(i, s, a, s_prime, reward)

        if len(self.predecessor) == 0:
            self.predecessor.append({"s_lead":s, "s,a,r":[(-1,-1,reward)]})
            
    def predecessoring(self,i,s, a, s_prime, reward): 
        if s != self.predecessor[i]["s_lead"]:
            self.predecessor.append({"s_lead":s_prime, "s,a,r":self.predecessor[-1]["s,a,r"]})
            self.predecessor[-1]["s,a,r"].append((s,a,reward))

    def match(self, s):
        for i in range(len(self.predecessor)):
            if s == self.predecessor[i]["s_lead"]:
                return self.predecessor[i]["s,a,r"]

    def plot_data(self, y):

        """ y is a 1D vector """
        x = np.arange(y.size)
        _ = plt.plot(x, y, "-")
        plt.show()

    def multi_plot_data(self, data, names):
        for i, y in enumerate(data):
            x = np.arange(y.size)
            plt.plot(x, y, "-", markersize=2, label=names[i])
        plt.legend(loc="lower right", prop={"size": 16}, numpoints=5)
        plt.show()






PQ_r = np.zeros((trials, max_steps))
PQ_50_r = np.zeros((trials, max_steps))
PQ_100_r = np.zeros((trials, max_steps))
PQ_realworld = np.zeros((trials, max_steps))
PQ_50_realworld = np.zeros((trials, max_steps))
PQ_100_realworld = np.zeros((trials, max_steps))

for t in range(trials):

    # PQ 5
    n = 0
    agent = PQ(theta,n,alpha,gamma,epsilon,max_steps,l,b,obstacles,target,start,orientation)
    PQ_r[t] = agent.learn()
    PQ_realworld = np.asarray(agent.its)


    # PQ 50
    n = 50
    agent = PQ(theta,n,alpha,gamma,epsilon,max_steps,l,b,obstacles,target,start,orientation)
    PQ_50_r[t] = agent.learn()
    PQ_50_realworld = np.asarray(agent.its)


    # Q-Learning
    n = 100
    agent =PQ(theta,n,alpha,gamma,epsilon,max_steps,l,b,obstacles,target,start,orientation)
    PQ_100_r[t] = agent.learn()
    PQ_100_realworld = np.asarray(agent.its)


    # Average across trials
    PQ_r = np.mean(PQ_r, axis=0)
    PQ_50_r = np.mean(PQ_50_r, axis=0)
    PQ_100_r = np.mean(PQ_100_r, axis=0)

data = [PQ_r, PQ_50_r, PQ_100_r]
names = ["PQ, n=0", "PQ, n=50", "PQ, n =100"]
agent.multi_plot_data(data, names)

data_realworld = [PQ_realworld, PQ_50_realworld, PQ_100_realworld]
names_realworld = ["PQ_realworld", "PQ_50_realworld", "PQ_50_realworld"]
agent.multi_plot_data(data_realworld, names_realworld)



































































