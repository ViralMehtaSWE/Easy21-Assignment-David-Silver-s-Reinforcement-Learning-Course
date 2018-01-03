import numpy as np
import random

Gamma = 1
stick_done = False
V = {}
Q = {}
N = {}
wins=0
losses=0
draws=0

def get_epsilon(n):
    n0 = 100
    epsilon = n0/(n0+n)
    return epsilon

def get_reward(s):
    dealer_sm, player_sm = s
    if((player_sm < 1) or (player_sm > 21)): return -1
    elif((dealer_sm < 1) or (dealer_sm > 21)): return 1
    elif((stick_done == True) and (dealer_sm > player_sm)): return -1
    elif((stick_done == True) and (dealer_sm < player_sm)): return 1
    else: return 0

def is_burst(s):
    dealer_sm, player_sm = s
    if((player_sm < 1) or (player_sm > 21)): return True
    elif((dealer_sm < 1) or (dealer_sm > 21)): return False
    return False

def step(s,a):
    #s is state, a tuple: (dealer’s firstcard 1–10, the player’s sum 1–21)
    #a is action, a boolean, hit(False) or stick(True).
    dealer_sm, player_sm = s
    if(a == True): #stick option
        while((dealer_sm < 17) and (dealer_sm > 0)):
            val = random.randint(1,10)
            prob = random.random()
            if(prob < 1/3):
                dealer_sm -= val
            else:
                dealer_sm += val
        next_state = (dealer_sm, player_sm)
        reward = get_reward(next_state)
    else:
        val = random.randint(1,10)
        prob = random.random()
        if(prob >= 2/3):
            player_sm -= val
        else:
            player_sm += val
        next_state = (dealer_sm, player_sm)
        reward = get_reward(next_state)
    return (next_state,reward)

def get_initial_state():
    s = (random.randint(1,10), random.randint(1,10))
    return s

def get_Q_val(sa):
    if(sa not in Q):
        Q[sa] = 0
    return Q[sa]    
    
def get_next_action(state, test = False):
    if(test == False):
        if(state in N): prob = get_epsilon(N[state])
        else: prob = 0.5
        if(random.random() < prob):
            return(random.randint(0,1) == 1)
    sa1 = (state, True)
    sa0 = (state, False)
    if(get_Q_val(sa0) < get_Q_val(sa1)): return True
    else: return False

def update_N(state):
    if(state not in N):
        N[state] = 1
    N[state] += 1

def update_Q(sa, reward):
    if(sa not in Q):
        Q[sa] = 0
    state, action = sa
    update_N(state)
    Q[sa] = ((N[state]-1)*Q[sa] + reward)/(N[state])
    
def train(num_episodes = 100000):
    global stick_done
    print("Training Started.....")
    for i in range(1, num_episodes + 1):
        state = get_initial_state()
        path = []
        G = 0
        p_Gamma = 1
        stick_done = False
        while((stick_done == False) and (is_burst(state) == False)):
            action = get_next_action(state)
            if(action == True): stick_done = True
            path.append((state, action))
            state, reward = step(state, action)
            G += p_Gamma*reward
            p_Gamma *= Gamma
            assert(abs(G)<=1)
        length = len(path)
        for j in range(length):
            update_Q(path[j], G)
        if((i%10000==0) and (i>0)):
            for tc in range(100000):
                test()
            total = wins + draws + losses
            print("Current Episode =", i, "accuracy stats (Win, Draw, Loss) =", (wins/total, draws/total, losses/total))
    print("Training Completed!")
        
def test():
    global stick_done
    global wins
    global draws
    global losses
    state = get_initial_state()
    #print(state)
    stick_done = False
    while((stick_done == False) and (is_burst(state) == False)):
        action = get_next_action(state, test = True)
        if(action == True): stick_done = True
        prev_state = state
        state, reward = step(state, action)
        if(action == False):
            action_str = "Hit"
        else:
            action_str = "Stick"
        #print(prev_state, action_str, reward, state)
    if(reward==1): wins+=1
    elif(reward==0): draws+=1
    else: losses+=1

train()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
states = [[i[0][0],i[0][1]] for i in Q]
V = [(i[0],i[1],max([Q[(tuple(i),False)],Q[(tuple(i),True)]])) for i in states]
V=np.array(list(set(V)))
print(V.shape)
X = V[:,0]
Y = V[:,1]
Z = V[:,2]
import pandas as pd
df = pd.DataFrame({'x': X, 'y': Y, 'z': Z})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(df.x, df.y, df.z)#, cmap=cm.jet, linewidth=0.1)
plt.show()
