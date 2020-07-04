import numpy as np

def BTgenerator(S0):
    rand = np.random.randint(0,2,1)[0]
    S = S0+1 if rand==1 else S0-1
    return int(S)

def BTDeltaHedge(Svec):
    N = len(Svec)
    Dvec = np.zeros(N)
    Dvec[0] = 1/2
    Dvec[1] = 3/4 if Svec[1]==51 else 1/4
    if Svec[2] == 52:
        Dvec[2] = 1
    if Svec[2] == 50:
        Dvec[2] = 1/2
    if Svec[3] == 53 or Svec[3] == 51:
        Dvec[3] = 1
        
    return Dvec

def HedgeProfit(Svec,Dvec,K,V0,c,L):
    Reward = np.zeros(len(Svec))
    Reward[0] = V0-Svec[0]*Dvec[0]-c*abs(Svec[0]*Dvec[0])
    for i in range(1,len(Svec)-1):
        Reward[i] = -Svec[i]*(Dvec[i]-Dvec[i-1])-c*abs(Svec[i]*(Dvec[i]-Dvec[i-1]))
    Vpayoff = Svec[-1]-K if (Svec[-1]-K>0) else 0
    Reward[-1] = -Vpayoff-Svec[-1]*(Dvec[-1]-Dvec[-2])-c*abs(Svec[-1]*(Dvec[-1]-Dvec[-2]))
    Profit = sum(Reward)*100
    return Profit, Reward

N = 5
S=50
V0 = 3/4
Sepisode = np.zeros((5))
Sepisode[0] = S
for i in range(1,N):
    S = BTgenerator(S)
    Sepisode[i] = S
Depisode = BTDeltaHedge(Sepisode)
profit, reward = HedgeProfit(Sepisode,Depisode,50,V0,0,100)
