# Aishwarya Pothula - 1001743470

# importing the environment file
import e
import numpy as np
import matplotlib.pyplot as plt

l = 15
b = 25
obstacles = [[2, 6], [6, 12], [12, 13], [7, 9]]
target = [14, 24]
start = [0, 0]
orientation = "n"



alpha = 0.1  # learning rate
gamma = 0.95  # discount
epsilon = 0.01
max_steps = 10000
trials = 1

class TS:
    def __init__(
        self,
        n,
        alpha,
        gamma,
        epsilon,
        max_steps,
        l,
        b,
        obstacles,
        target,
        start,
        orientation,
    ):

        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon
        self.max_steps = max_steps
        self.l = l
        self.b = b
        self.obstacles = obstacles
        self.target = target
        self.start = start
        self.orientation = orientation
        self.steps = 0
        self.its = []
        self.env = e.GridWorldEnvironment(
            l=self.l,
            b=self.b,
            obstacles=self.obstacles,
            target=self.target,
            start=self.start,
            orientation=self.orientation,
        )
        self.q_table = np.zeros((self.l * self.b * 4, 4))
        self.a_dic = {0: "f_w", 1: "b_w", 2: "l_t", 3: "r_t"}
        self.create_model(self.l * self.b * 4, 4)

   # initializing necessary tables for planning
    def create_model(self, n_states, n_actions):
        self.transitions = np.zeros((n_states, n_actions+1))
        self.rewards = np.zeros((n_states, n_actions))
        ## action table for sampling actions in sample
        self.action_table = np.zeros((n_states, n_actions))
   
    # chooses an action based on greedy epsilon policy
    def choose_action(self, state):
        r = np.random.uniform()
        a_dic = {0: "f_w", 1: "b_w", 2: "l_t", 3: "r_t"}
        if r < self.eps:
            action = np.random.choice(["f_w", "b_w", "l_t", "r_t"])
        else:
        
            action = a_dic[np.argmax(self.q_table[state])]
        return action

    # updates the tables created for planning with s_prime, r values
    def add(self, s, a, s_prime, reward):
        self.transitions[s][a] = s_prime
        ##aishwarya##
        self.transitions[s][-1] = 1
        ##aishwarya##
        self.action_table[s][a] = 1
        self.rewards[s][a] = reward
        
    # samples random previously visited states with the help of flag.
    # Last col of self.transitions is is for flag. flag =1 for visited states
    def trajectory_sample(self):
        a_dic = {0: "f_w", 1: "b_w", 2: "l_t", 3: "r_t"}
        r = np.random.uniform()
        
        flags = self.transitions[:,-1]
        s = np.random.choice(list(np.where(flags == 1)[0]))
        
        # chooses randomly from taken actions of visited states
        if r < self.eps:
           act = self.action_table[s][:]
           acti = np.random.choice(list(np.where(act == 1)[0]))
           action = a_dic[acti]
           
        else:
        # chooses the action with the best value for that state from q_table
        # thus creating a trajectory
        # This is the only firrereing part from DynaQ
            action = a_dic[np.argmax(self.q_table[s])]
           
        return s, action
        
        
     # this is to generate s_prime and r within planning. r, s_prime = model(s,a)
    def step(self, s, a):
        a_dic = {"f_w": 0, "b_w": 1, "l_t": 2, "r_t": 3}
        b = a_dic[a]
        s_prime = self.transitions[s][b]
        rewards = self.rewards[s][b]
        return s_prime, rewards
        
        
 # this is where all the model / offline part happens
    def planning(self):
        a_dic = {0: "f_w", 1: "b_w", 2: "l_t", 3: "r_t"}

        for _ in range(self.n):
            a_dic = {"f_w": 0, "b_w": 1, "l_t": 2, "r_t": 3}
            st, a = self.trajectory_sample()
            s = int(st)
            s_p, reward = self.step(s, a)
            s_prime = int(s_p)
            b = a_dic[a]
            
             # q_table update by model
            self.q_table[s][b] = self.q_table[s][b] + self.alpha * (
                reward + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[s][b]
            )
            
    def learn(self):
        

        # s = 0 because start position is [(0,0),0] and s = a + 15 * b + 15 * 25 *c
        s = 0
        reward_list = []
        cum_reward = [0]
       
        for _ in range(self.max_steps):
            # incrementing this variable to count steps
            self.steps += 1
            a_dic = {"f_w": 0, "b_w": 1, "l_t": 2, "r_t": 3}
            b = self.choose_action(s)
            s_prime, reward, done = self.env.step(b)
            # a needs to an integer
            a = a_dic[b]
            self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                reward + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[s][a]
            )
            
             # model behaviour starts
            self.add(s, a, s_prime, reward)
            self.planning()
            s = s_prime
            if done == True:
                print("episode")
                self.its.append(self.steps)
                self.steps = 0
                self.env.reset()
                s = 0

            cum_reward.append(cum_reward[-1] + reward)

        return np.array(cum_reward[1:])

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


TS_r = np.zeros((trials, max_steps))
TS_50_r = np.zeros((trials, max_steps))
TS_100_r = np.zeros((trials, max_steps))
TS_realworld = np.zeros((trials, max_steps))
TS_50_realworld = np.zeros((trials, max_steps))
TS_100_realworld = np.zeros((trials, max_steps))
for t in range(trials):

    # Q-Learning
    n = 0
    agent = TS(
        n, alpha, gamma, epsilon, max_steps, l, b, obstacles, target, start, orientation
    )
    
    TS_r[t] = agent.learn()
    TS_realworld = np.asarray(agent.its)

    # DynaQ 50
    n = 50
    agent = TS(
        n, alpha, gamma, epsilon, max_steps, l, b, obstacles, target, start, orientation
    )
    TS_50_r[t] = agent.learn()
    TS_50_realworld = np.asarray(agent.its)

    # DynaQ 100
    n = 100
    agent = TS(
        n, alpha, gamma, epsilon, max_steps, l, b, obstacles, target, start, orientation
    )
    TS_100_r[t] = agent.learn()
    TS_100_realworld = np.asarray(agent.its)

    # Average across trials
    TS_r = np.mean(TS_r, axis=0)
    TS_50_r = np.mean(TS_50_r, axis=0)
    TS_100_r = np.mean(TS_100_r, axis=0)

data = [TS_r, TS_50_r, TS_100_r]
names = ["TS, n=0", "TS, n=50", "TS, n =100"]
agent.multi_plot_data(data, names)

data_realworld = [TS_realworld, TS_50_realworld, TS_100_realworld]
names_realworld = ["TS", "TS_50_realworld", "TS_50_realworld"]
agent.multi_plot_data(data_realworld, names_realworld)
