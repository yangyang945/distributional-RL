import numpy as np
import scipy.stats

def HestonGenerator(S0,v0,r,sigma,rho,kappa,theta,delta_t):
    randS = np.random.normal(0,1,1)
    randv = rho*randS+np.sqrt(1-rho*rho)*np.random.normal(0,1,1)
    S = S0*np.exp((r-0.5*v0)*delta_t+np.sqrt(v0)*np.sqrt(delta_t)*randS)
    v = v0*np.exp((1/v0)*(kappa*(theta-v0)-0.5*sigma*sigma)*delta_t+sigma/np.sqrt(v0)*np.sqrt(delta_t)*randv)
    return S,v

def DeltaH_Heston(StockPriceVec,K,r,sigma,delta_t):
    N = len(StockPriceVec)
    DeltaVec = np.zeros(shape=(N-1,1))
    for i in range(N-1):
        TimeRemain = (N-i-1)*delta_t
        dPlus = (np.log(StockPriceVec[i]/K)+(r+sigma[i]*sigma[i]/2)*TimeRemain) / (sigma[i]*TimeRemain**(1/2))
        DeltaVec[i] = scipy.stats.norm(0,1).cdf(dPlus)
    return DeltaVec

# c is the coefficent(weight) of the trasncation cost,K is the strike price,L is the number of Call Optioins
# assume this is to hedge "short a call"
# assume no discount on the rewards
def HedgeProfit(StockPriceVec,DeltaVec,c,K,L):
    N = len(StockPriceVec)
    Reward = np.zeros(shape=(N,1))
    Reward[0] = -StockPriceVec[0]*round(DeltaVec[0][0]*L)-c*abs(StockPriceVec[0]*round(DeltaVec[0][0]*L))
    for i in range(1,N-1):
        Reward[i] = StockPriceVec[i]*(round(L*DeltaVec[i-1][0])-round(L*DeltaVec[i][0]))-c*abs(StockPriceVec[i]*(round(L*DeltaVec[i-1][0])-round(L*DeltaVec[i][0])))
    CallPayoff = StockPriceVec[-1]-K if (StockPriceVec[-1]-K>0) else 0
    Reward[-1] = StockPriceVec[-1]*round(L*DeltaVec[-1][0])-c*abs(StockPriceVec[-1]*round(L*DeltaVec[-1][0]))-CallPayoff*L
    NetProfit = sum(Reward)
    return NetProfit,Reward
 
M = 10
N = 10**3
allS = np.zeros((N,M))
allv = np.zeros((N,M))
allD = np.zeros((N,M))
AvgProfit = np.zeros((N))
for i in range(N):
    S0,v0,r,sigma,rho,kappa,theta,delta_t,K = 50,0.09,0,0.9,-0.7,5,0.2,3/252,50
    for j in range(M):
        allS[i][j] = S0
        allv[i][j] = v0
        S0,v0 = HestonGenerator(S0,v0,r,sigma,rho,kappa,theta,delta_t)
    D = DeltaH_Heston(allS[i],K,r,np.sqrt(allv[i]),delta_t)
    #D = DeltaHedge(allS[i],K,r,np.sqrt(v0),delta_t)
    Profit,Reward = HedgeProfit(allS[i][:,np.newaxis],D,0,50,100)
    AvgProfit[i] = Profit[0]
    D = np.append(D,0)
    allD[i] = D
    if i%100 == 0:
        print("{}".format(i))

    
allS = allS.reshape((M*N,1))  
allv = allv.reshape((M*N,1))   
np.max(allS)
np.min(allS)
np.quantile(allS,0.999)
np.quantile(allS,0.001)

np.max(allv)
np.min(allv)
np.quantile(allv,0.999)

np.quantile(AvgProfit,0.1)
np.quantile(AvgProfit,0.3)
np.quantile(AvgProfit,0.5)
np.quantile(AvgProfit,0.7)
np.quantile(AvgProfit,0.9)
