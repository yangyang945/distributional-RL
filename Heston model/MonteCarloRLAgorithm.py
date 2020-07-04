
import numpy as np
import pandas as pd
import scipy.stats

# Envrionment
# to hedge 100*(sell a Call), K=S0=50, 10 periods of time, 3 tarding days per period
class Env:
    def __init__(self,MAX_t,MAX_S,MAX_H,Strick_price,start_state,SLowerBd):
        self.MAX_t,self.MAX_S,self.MAX_H = MAX_t,MAX_S,MAX_H
        self.start_state = start_state
        self.Strick_price = Strick_price
        self.state = self.start_state
        self.SLowerBd = SLowerBd
        self.c = 0.01
        
    def step(self,action,r,sigma,delta_t):
        t,S,H = self.state
        t += 1
        if t == 1:
            reward = -(self.SLowerBd+S)*action-self.c*abs((self.SLowerBd+S)*action)
            S = self.next_S(S,r,sigma,delta_t)
            H = action
            new_state = (t,S,H)
        elif ((t>1) and (t<self.MAX_t)):
            reward = (self.SLowerBd+S)*(H-action)-self.c*abs((self.SLowerBd+S)*(H-action))
            S = self.next_S(S,r,sigma,delta_t)
            H = action
            new_state = (t,S,H)
        else:
            call_payoff = (self.SLowerBd+S-self.Strick_price) if (self.SLowerBd+S-self.Strick_price > 0) else 0
            reward = (self.SLowerBd+S)*H-self.c*abs((self.SLowerBd+S)*H)-call_payoff*100 # *100 because we have 100 call options
            new_state = self.start_state
        
        self.state = new_state
        return reward,new_state
        
    def next_S(self,S,r,sigma,delta_t):
        S = (S+self.SLowerBd)*np.exp((r-sigma*sigma/2)*delta_t+sigma*delta_t**(1/2)*np.random.normal(0,1,1))
        S = round(S[0])# S is a np.array
        # aussme S is bdd, say 40~60
        S = S if (S<=self.SLowerBd+self.MAX_S-1) else self.SLowerBd+self.MAX_S-1
        S = S if (S>=self.SLowerBd) else self.SLowerBd
        return int(S-self.SLowerBd)
    


# Agent

class AgentDeltaHedging:
    def __init__(self,MAX_t,MAX_S,MAX_H,start_state_action,SLowerBd):
        self.MAX_t,self.MAX_S,self.MAX_H = MAX_t,MAX_S,MAX_H
        self.SLowerBd = SLowerBd
        self.start_state_action = start_state_action
        self.state_action = [0 for i in range(self.MAX_t)]
        self.state_action[0] = self.start_state_action
        self.sum_action_state_value = np.zeros(shape=(self.MAX_t,self.MAX_S,self.MAX_H,self.MAX_H)) # sum of "sum of rewards" for all episode
        self.visit_number = np.zeros(shape=(self.MAX_t,self.MAX_S,self.MAX_H,self.MAX_H))# number of visit at each state
        self.reward = np.zeros(shape=(self.MAX_t,1))
    
    def make_action(self,reward,state,K,r,sigma,delta_t):
        t,S,H = state
        if t == self.MAX_t-1:
            action = 0  # in the last step, clean the portfolio
        else:        
            time_remain = delta_t*(self.MAX_t-t-1)
            dPlus = (np.log((S+self.SLowerBd)/K)+(r+sigma*sigma/2)*time_remain) / (sigma*time_remain**(1/2))
            Delta = scipy.stats.norm(0,1).cdf(dPlus)
            action = int(round(Delta*100))
        
        if t==0:
            sumReward = 0
            for i in range(self.MAX_t-1,-1,-1):
                sumReward += self.reward[i]
                t,S,H,a = self.state_action[i]
                self.sum_action_state_value[t][S][H][a] += sumReward
            self.state_action = [0 for i in range(self.MAX_t)]
            self.reward = np.zeros(shape=(self.MAX_t,1))
            
        self.state_action[t]=(t,S,H,action)
        self.visit_number[t][S][H][action] += 1
        self.reward[t] = reward
        return action     
                
        
# Main      

def run_RL2(env,agent,number_of_steps,K,r,sigma,delta_t):
    action = agent.start_state_action[-1]
    for i in range(number_of_steps):
        reward,new_state = env.step(action,r,sigma,delta_t)
        action = agent.make_action(reward, new_state,K,r,sigma,delta_t)
    return agent.sum_action_state_value, agent.visit_number # the ratio of them are sample average (estimator of expectation)
       
# Inputs
strick_price = 50
MAX_t,MAX_S,MAX_H = 63,101,101
start_state = (0,50,0)
start_state_action = (0,10,0,0)
number_of_steps = 10**6
SLowerBd = 0
K,r,sigma,delta_t = 50,0,0.3,1/252
    
GBMenv = Env(MAX_t,MAX_S,MAX_H,strick_price,start_state,SLowerBd)
DHRL = AgentDeltaHedging(MAX_t,MAX_S,MAX_H,start_state_action,SLowerBd)
sumQ,number = run_RL2(GBMenv,DHRL,number_of_steps,K,r,sigma,delta_t)
(sumQ/number)[0][10][0]
  
