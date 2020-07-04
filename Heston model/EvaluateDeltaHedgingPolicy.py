#This is the modified Algorithm 3 in the thesis
#This is used to evaluate the delta-hedging policy, instead of seeking for the optimal policy

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import scipy.stats
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

# Envrionment
class Env:
    def __init__(self,MAX_t,MAX_S,MAX_H,MAX_v,Strick_price,start_state,SLowerBd,vLowerBd):
        self.MAX_t,self.MAX_S,self.MAX_H,self.MAX_v = MAX_t,MAX_S,MAX_H,MAX_v
        self.start_state = start_state
        self.Strick_price = Strick_price
        self.state = self.start_state
        self.SLowerBd = SLowerBd
        self.vLowerBd = vLowerBd #0.01
        self.c = 0
        
    def step(self,action,r,sigma,rho,kappa,theta,delta_t,OptionIniP):
        t,S,H,v = self.state
        t += 1
        current_holding = H*10
        new_holding = action*10
        if t == 1:
            reward = OptionIniP*100-(self.SLowerBd+S)*new_holding-self.c*abs(OptionIniP*100+(self.SLowerBd+S)*new_holding)
            S,v = self.next_Sv(S,v,r,sigma,rho,kappa,theta,delta_t)
            new_state = (t,S,action,v)
        elif ((t>1) and (t<self.MAX_t)):
            reward = (self.SLowerBd+S)*(current_holding-new_holding)-self.c*abs((self.SLowerBd+S)*(current_holding-new_holding))
            S,v = self.next_Sv(S,v,r,sigma,rho,kappa,theta,delta_t)
            new_state = (t,S,action,v)
        else:
            call_payoff = (self.SLowerBd+S-self.Strick_price) if (self.SLowerBd+S-self.Strick_price > 0) else 0
            reward = (self.SLowerBd+S)*current_holding-self.c*abs((self.SLowerBd+S)*current_holding)-call_payoff*100 # *100 because we have 100 call options
            new_state = self.start_state
        
        self.state = new_state
        return reward,new_state
        
    def next_Sv(self,S0,v0,r,sigma,rho,kappa,theta,delta_t):
        v0 = v0/20 + self.vLowerBd
        randS = np.random.normal(0,1,1)
        randv = rho*randS+np.sqrt(1-rho*rho)*np.random.normal(0,1,1)
        S = (S0+self.SLowerBd)*np.exp((r-0.5*v0)*delta_t+np.sqrt(v0)*np.sqrt(delta_t)*randS)
        v = v0*np.exp((1/v0)*(kappa*(theta-v0)-0.5*sigma*sigma)*delta_t+sigma/np.sqrt(v0)*np.sqrt(delta_t)*randv)
        S = round(S[0])# S is a np.array
        # aussme S is bdd, say 40~60
        S = S if (S<=self.SLowerBd+self.MAX_S-1) else self.SLowerBd+self.MAX_S-1
        S = S if (S>=self.SLowerBd) else self.SLowerBd
        v = round((v[0]-self.vLowerBd)*20)
        v = v if (v<=self.MAX_v-1) else self.MAX_v-1
        v = v if (v>=0) else 0     
        return int(S-self.SLowerBd), int(v)
    
    
class AgentQRDQN:
    def __init__(self,env):
        self.action_number = env.MAX_H
        self.quantile_number = 5
        self.MAX_t = env.MAX_t
        self.MAX_S = env.MAX_S
        self.MAX_H = env.MAX_H
        self.MAX_v = env.MAX_v
        self.SLowerBd = env.SLowerBd
        self.vLowerBd = env.vLowerBd
        self.state_dim = self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v
        self.batch_size = 10
        self.learning_rate = 0.001
        self.memory_size = 1000
        self.memory_counter = 0
        self.MEMORY_POOL = deque(maxlen=self.memory_size)
        self.epsilon = 0.1
        self.gamma = 1
        self.tau = np.array([(2*(i-1)+1)/(2*self.quantile_number) for i in range(1, self.quantile_number+1)])
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.qtarget_update_preiod = 5 # every what number of steps then update q_target network once
        self.q = self.create_MLP() # network used to be trained
        self.q_target = self.create_MLP() # networks to do Q(s,a) approximation
    
    def create_MLP(self):
        return tf.keras.Sequential([
            Input([self.state_dim, ]),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_number * self.quantile_number, activation='linear'),
            Reshape([self.action_number, self.quantile_number])
        ])
    
    def train_MLP(self,sample):
        state = np.array([data[0] for data in sample])
        action = np.array([data[1] for data in sample])
        reward = np.array([data[2] for data in sample])
        next_state = np.array([data[3] for data in sample])
        state_boolean = np.zeros((len(state),self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v))
        for i in range(len(state)):
            state_boolean[i][state[i][0]] = 1
            state_boolean[i][state[i][1]+self.MAX_t] = 1
            state_boolean[i][state[i][2]+self.MAX_t+self.MAX_S] = 1
            state_boolean[i][state[i][3]+self.MAX_t+self.MAX_S+self.MAX_H] = 1
        bellman_target = []
        for i in range(self.batch_size):
            bellman_target.append(np.ones(self.quantile_number)*reward[i])
        bellman_target = tf.constant(bellman_target)
        
        #calculate the theta to be updated
        with tf.GradientTape() as tape:
            theta_tobeupdated = self.q(state_boolean) # this is a tf, not np.array
            # loss is to measure gaps between bellman_target and theta_tobeupdated
            loss = self.huber_lossfunction(bellman_target,theta_tobeupdated,action)
        gradients = tape.gradient(loss,self.q.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.q.trainable_variables))
        return loss.numpy()
        
    def huber_lossfunction(self,target,tf_approx,action):
        action = action.astype("int")
        actionVec = np.zeros((len(action),self.action_number)) # [1,0,0...] represents action"0"; [0,1,0..] represents action"1"...
        for i in range(len(action)):
            actionVec[i][action[i]]=1
        actionVec = actionVec.astype("float64")
        actionTf = tf.expand_dims(actionVec, -1)
        theta_tobeupdated = tf.reduce_sum(actionTf*tf_approx,axis=1) # theta i (x,a)
        theta_tobeupdated_tile = tf.tile(tf.expand_dims(theta_tobeupdated,-1),[1,1,self.quantile_number])
        target_tile = tf.tile(tf.expand_dims(target,1),[1,self.quantile_number,1])       
        huber_loss = self.huber_loss(tf.expand_dims(target_tile, -1), tf.expand_dims(theta_tobeupdated_tile, -1))
        error = tf.math.subtract(target_tile, theta_tobeupdated_tile)
        self.tau = self.tau.astype("float64")
        tau = tf.tile(tf.expand_dims(self.tau, 0), [self.batch_size,1])
        tau_tile = tf.tile(tf.expand_dims(tau,-1),[1,1,self.quantile_number])
        inv_tau = tf.tile(1.0-tf.expand_dims(self.tau, 0), [self.batch_size,1])
        inv_tau_tile = tf.tile(tf.expand_dims(inv_tau,-1),[1,1,self.quantile_number])
        loss= tf.where(tf.less(error, 0.0),inv_tau_tile*huber_loss,tau_tile*huber_loss)
        return tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss,axis=2),axis=1))
    
    def update_q_traget(self):
        weights = self.q.get_weights()
        self.q_target.set_weights(weights)      
        
    def update_MEMORY_POOL(self,state,action,reward,next_state):
        loss = np.nan
        if self.memory_counter % self.qtarget_update_preiod == 0:
            self.update_q_traget()
            
        if self.memory_counter > self.batch_size:
            sample = random.sample(self.MEMORY_POOL,self.batch_size)
            loss = self.train_MLP(sample)
            if self.memory_counter % 100 == 0:
              print(loss)
        
        self.MEMORY_POOL.append([state,action,reward,next_state])
        self.memory_counter += 1
        return loss
       
    # the eponsion can be changed
    # argmax the mean of reward         
    def make_action(self,state,K,r,delta_t): # the eponsion can be changed
        t,S,H,v = state
        sigma = np.sqrt(v/20 + self.vLowerBd)
        if t == self.MAX_t-1:
            action = 0  # in the last step, clean the portfolio
        else:        
            time_remain = delta_t*(self.MAX_t-t-1)
            dPlus = (np.log((S+self.SLowerBd)/K)+(r+sigma*sigma/2)*time_remain) / (sigma*time_remain**(1/2))
            Delta = scipy.stats.norm(0,1).cdf(dPlus)
            action = int(round(Delta*10))
        return action
        

def run_RL(env,agent,number_of_steps,r,sigma,rho,kappa,theta,delta_t,start_state,K):
    state = env.start_state
    trajectory = [[(0,0,0,0),0,0,(0,0,0,0)] for j in range(env.MAX_t)]
    loss_vec = np.array([])
    TimeRemain = (env.MAX_t-1)*delta_t
    S0 = start_state[1]+env.SLowerBd
    v0 = np.sqrt(start_state[3]/20 + env.vLowerBd)
    dPlus = (np.log(S0/K)+(r+v0*v0/2)*TimeRemain)/(v0*TimeRemain**(1/2))
    dMins = (np.log(S0/K)+(r-v0*v0/2)*TimeRemain)/(v0*TimeRemain**(1/2))
    OptionIniP = S0*scipy.stats.norm(0,1).cdf(dPlus)-K*np.exp(-r*TimeRemain)*scipy.stats.norm(0,1).cdf(dMins) 
    for i in range(number_of_steps):
        action = agent.make_action(state,K,r,delta_t)
        reward,new_state = env.step(action,r,sigma,rho,kappa,theta,delta_t,OptionIniP)
        trajectory[state[0]][0] = state
        trajectory[state[0]][1] = action
        trajectory[state[0]][2] = reward
        trajectory[state[0]][3] = new_state
        
        if new_state[0]==0:
            for j in range(env.MAX_t-2,-1,-1):
                trajectory[j][2] += trajectory[j+1][2]

            for k in range(env.MAX_t):
                loss = agent.update_MEMORY_POOL(trajectory[k][0],trajectory[k][1],trajectory[k][2],trajectory[k][3])
                loss_vec = np.append(loss_vec,loss)

        state = new_state
    return agent.q,loss_vec  


#main
strick_price = 50
MAX_t,MAX_S,MAX_H,MAX_v = 10,21,11,6
number_of_steps = 5*10**3
SLowerBd,vLowerBd = 40,0.01
K,r,sigma,delta_t = 50,0,0.9,1/252
rho,kappa,theta = -0.7,5,0.2

start_state_action = (0,10,0,2,0)
start_state = start_state_action[:-1]
start_state_boolean = np.zeros((1,MAX_t+MAX_S+MAX_H+MAX_v))
start_state_boolean[0][start_state[0]] = 1
start_state_boolean[0][start_state[1]+MAX_t] = 1
start_state_boolean[0][start_state[2]+MAX_t+MAX_S] = 1
start_state_boolean[0][start_state[3]+MAX_t+MAX_S+MAX_H] = 1

M = 200
quantileVec = np.zeros((M,5))
for i in range(M):
  envGBM = Env(MAX_t,MAX_S,MAX_H,MAX_v,strick_price,start_state,SLowerBd,vLowerBd)
  QRDQN = AgentQRDQN(envGBM)
  trained_model,loss_vec = run_RL(envGBM,QRDQN,number_of_steps,r,sigma,rho,kappa,theta,delta_t,start_state,K)
  quantile = trained_model.predict([start_state_boolean])
  print(quantile)
  plt.figure(1, figsize=(12,8))
  plt.grid()
  plt.plot(loss_vec)
  plt.ylabel('loss',fontsize=20)
  plt.xlabel("steps",fontsize=20)
  plt.title("LOSS",fontsize=25)
  plt.show()
  quantileVec[i] = quantile[0,5]
  print(quantile[0,5])
plt.figure(1, figsize=(12,8))
plt.grid()
plt.hist(quantileVec[:,0],bins=10)
plt.title("0.1 Quantile",fontsize=25)
plt.tick_params(labelsize=20)
plt.show()

plt.figure(1, figsize=(12,8))
plt.grid()
plt.hist(quantileVec[:,2],bins=10)
plt.title("0.5 Quantile",fontsize=25)
plt.tick_params(labelsize=20)
plt.show()
