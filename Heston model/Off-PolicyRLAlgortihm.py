#This is the modified Algorithm 2 in the thesis

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import scipy.stats

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
        self.c = 0.01
        
    def step(self,action,r,sigma,rho,kappa,theta,delta_t):
        t,S,H,v = self.state
        t += 1
        if t == 1:
            reward = -(self.SLowerBd+S)*action-self.c*abs((self.SLowerBd+S)*action)
            S,v = self.next_Sv(S,v,r,sigma,rho,kappa,theta,delta_t)
            H = action
            new_state = (t,S,H,v)
        elif ((t>1) and (t<self.MAX_t)):
            reward = (self.SLowerBd+S)*(H-action)-self.c*abs((self.SLowerBd+S)*(H-action))
            S,v = self.next_Sv(S,v,r,sigma,rho,kappa,theta,delta_t)
            H = action
            new_state = (t,S,H,v)
        else:
            call_payoff = (self.SLowerBd+S-self.Strick_price) if (self.SLowerBd+S-self.Strick_price > 0) else 0
            reward = (self.SLowerBd+S)*H-self.c*abs((self.SLowerBd+S)*H)-call_payoff*100 # *100 because we have 100 call options
            new_state = self.start_state
        
        self.state = new_state
        return reward,new_state
        
    def next_Sv(self,S0,v0,r,sigma,rho,kappa,theta,delta_t):
        v0 = v0/100 + self.vLowerBd
        randS = np.random.normal(0,1,1)
        randv = rho*randS+np.sqrt(1-rho*rho)*np.random.normal(0,1,1)
        S = (S0+self.SLowerBd)*np.exp((r-0.5*v0)*delta_t+np.sqrt(v0)*np.sqrt(delta_t)*randS)
        v = v0*np.exp((1/v0)*(kappa*(theta-v0)-0.5*sigma*sigma)*delta_t+sigma/np.sqrt(v0)*np.sqrt(delta_t)*randv)
        S = round(S[0])# S is a np.array
        # aussme S is bdd, say 40~60
        S = S if (S<=self.SLowerBd+self.MAX_S-1) else self.SLowerBd+self.MAX_S-1
        S = S if (S>=self.SLowerBd) else self.SLowerBd
        v = round(v[0]*100)
        v = v if (v<=self.vLowerBd*100+self.MAX_v-1) else self.vLowerBd*100+self.MAX_v-1
        v = v if (v>=self.vLowerBd*100) else self.vLowerBd*100      
        return int(S-self.SLowerBd), int(v-self.vLowerBd)
    
    
class AgentQRDQN:
    def __init__(self,env):
        self.action_number = env.MAX_H
        self.quantile_number = 5
        self.MAX_t = env.MAX_t
        self.MAX_S = env.MAX_S
        self.MAX_H = env.MAX_H
        self.MAX_v = env.MAX_v
        self.state_dim = self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v
        self.batch_size = 10
        self.learning_rate = 0.0001
        self.memory_size = 1000
        self.memory_counter = 0
        self.MEMORY_POOL = [[(0,0,0,0),0,0,(0,0,0,0)] for j in range(self.memory_size)]
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
            Dense(self.action_number * self.quantile_number, activation='linear'),
            Reshape([self.action_number, self.quantile_number])
        ])
    
    def train_MLP(self,sample):
        state = np.array([data[0] for data in sample])
        action = np.array([data[1] for data in sample])
        reward = np.array([data[2] for data in sample])
        next_state = np.array([data[3] for data in sample])
        state_boolean = np.zeros((len(state),self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v))
        next_state_boolean = np.zeros((len(next_state),self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v))
        for i in range(len(state)):
            state_boolean[i][state[i][0]] = 1
            state_boolean[i][state[i][1]+self.MAX_t] = 1
            state_boolean[i][state[i][2]+self.MAX_t+self.MAX_S] = 1
            state_boolean[i][state[i][3]+self.MAX_t+self.MAX_S+self.MAX_H] = 1
        for i in range(len(next_state)):
            next_state_boolean[i][next_state[i][0]] = 1
            next_state_boolean[i][next_state[i][1]+self.MAX_t] = 1
            next_state_boolean[i][next_state[i][2]+self.MAX_t+self.MAX_S] = 1
            next_state_boolean[i][next_state[i][3]+self.MAX_t+self.MAX_S+self.MAX_H] = 1
        # calculate bellman target
        target_theta = self.q_target.predict(next_state_boolean) # this theta is used to calculate bellman target, shpae is batchSize*actionNumber*quantileNumber
        next_action = np.argmax(np.mean(target_theta,axis=2),axis=1) # the action leads to max "mean of quantiles"
        bellman_target = []
        for i in range(self.batch_size):
            if state[i][0] == self.MAX_t-1: # if this is the last step in one trajectory
                bellman_target.append(np.ones(self.quantile_number)*reward[i])
            else:
                bellman_target.append(reward[i]+self.gamma*target_theta[i][next_action[i]])
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
        if self.memory_counter % self.qtarget_update_preiod == 0:
            self.update_q_traget()
            
        if self.memory_counter > self.batch_size:
            sample = random.sample(self.MEMORY_POOL,self.batch_size)
            loss = self.train_MLP(sample)
            if self.memory_counter % 100 == 0:
                print(loss)
        
        memory_index = self.memory_counter % self.memory_size
        self.MEMORY_POOL[memory_index] = [state,action,reward,next_state]
        self.memory_counter += 1
       
    # the eponsion can be changed
    # argmax the mean of reward         
    def make_action(self,state): # the eponsion can be changed
        t,S,H,v = state
        if t == self.MAX_t-1:
            return 0
        else:           
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_number)
            else:
                state_boolean = np.zeros((1,self.MAX_t+self.MAX_S+self.MAX_H+self.MAX_v))
                state_boolean[0][state[0]] = 1
                state_boolean[0][state[1]+self.MAX_t] = 1
                state_boolean[0][state[2]+self.MAX_t+self.MAX_S] = 1
                state_boolean[0][state[3]+self.MAX_t+self.MAX_S+self.MAX_H] = 1
                quantile = self.q.predict([state_boolean])[0]
                return np.argmax(np.mean(quantile,axis=1))
            
    def DeltaHedging_action(self,state,K,r,delta_t): # the eponsion can be changed
        t,S,H,v = state
        sigma = np.sqrt(v/100)
        if t == self.MAX_t-1:
            action = 0  # in the last step, clean the portfolio
        else:        
            time_remain = delta_t*(self.MAX_t-t-1)
            dPlus = (np.log((S+self.SLowerBd)/K)+(r+sigma*sigma/2)*time_remain) / (sigma*time_remain**(1/2))
            Delta = scipy.stats.norm(0,1).cdf(dPlus)
            action = int(round(Delta*100))
        return action

        

def run_RL(env,agent,number_of_steps,r,sigma,rho,kappa,theta,delta_t,start_state):
    state = env.start_state
    for i in range(number_of_steps):
        action = agent.make_action(state)
        reward,new_state = env.step(action,r,sigma,rho,kappa,theta,delta_t)
        agent.update_MEMORY_POOL(state,action,reward,new_state)
        state = new_state
    return agent.q

def test_env(env,number_of_steps,r,sigma,rho,kappa,theta,delta_t):
    for i in range(number_of_steps):
        action = 0
        reward,new_state = env.step(action,r,sigma,rho,kappa,theta,delta_t)
        print(new_state)

#main
strick_price = 50
MAX_t,MAX_S,MAX_H,MAX_v = 10,21,101,50
number_of_steps = 10**5
SLowerBd,vLowerBd = 40,0.01
K,r,sigma,delta_t = 50,0,0.9,3/252
rho,kappa,theta = -0.7,5,0.2

start_state_action = (0,10,0,9,0)
start_state = start_state_action[:-1]
start_state_boolean = np.zeros((1,MAX_t+MAX_S+MAX_H+MAX_v))
start_state_boolean[0][start_state[0]] = 1
start_state_boolean[0][start_state[1]+MAX_t] = 1
start_state_boolean[0][start_state[2]+MAX_t+MAX_S] = 1
start_state_boolean[0][start_state[3]+MAX_t+MAX_S+MAX_H] = 1

envGBM = Env(MAX_t,MAX_S,MAX_H,MAX_v,strick_price,start_state,SLowerBd,vLowerBd)
#test_env(envGBM,number_of_steps,r,sigma,rho,kappa,theta,delta_t)
QRDQN = AgentQRDQN(envGBM)
trained_model = run_RL(envGBM,QRDQN,number_of_steps,r,sigma,rho,kappa,theta,delta_t,start_state)
quantile = trained_model.predict([start_state_boolean])
