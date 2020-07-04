import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def GBMGenerator(S0,r,sigma,delta_t,N,UpperB,LowerB):
    StockPriceVec = np.zeros(shape=(N,1))
    NoramlRandomVec = np.random.normal(0,1,N-1)
    StockPriceVec[0]=S0
    for i in range(N-1):
        stockprice = round((StockPriceVec[i]*np.exp((r-sigma*sigma/2)*delta_t+sigma*delta_t**(1/2)*NoramlRandomVec[i]))[0])
        stockprice = stockprice if (stockprice<=UpperB) else UpperB
        stockprice = stockprice if (stockprice>=LowerB) else LowerB
        StockPriceVec[i+1] = stockprice
    return StockPriceVec

# Dalta rounding to the nearest 5%!!
def DeltaHedge(StockPriceVec,K,r,sigma,delta_t):
    N = len(StockPriceVec)
    DeltaVec = np.zeros(shape=(N-1,1))
    for i in range(N-1):
        TimeRemain = (N-i-1)*delta_t
        dPlus = (np.log(StockPriceVec[i]/K)+(r+sigma*sigma/2)*TimeRemain) / (sigma*TimeRemain**(1/2))
        Delta = scipy.stats.norm(0,1).cdf(dPlus)
        RoundDelta = round(Delta[0]/0.01)
        DeltaVec[i] = RoundDelta*0.01
    return DeltaVec

# c is the coefficent(weight) of the trasncation cost,K is the strike price,L is the number of Call Optioins
# assume this is to hedge "short a call"
# assume no discount on the rewards
def HedgeProfit(StockPriceVec,DeltaVec,c,K,L):
    N = len(StockPriceVec)
    Reward = np.zeros(shape=(N,1))
    TimeRemain = (N-1)*delta_t
    dPlus = (np.log(StockPriceVec[0]/K)+(r+sigma*sigma/2)*TimeRemain)/(sigma*TimeRemain**(1/2))
    dMins = (np.log(StockPriceVec[0]/K)+(r-sigma*sigma/2)*TimeRemain)/(sigma*TimeRemain**(1/2))
    OptionIniP = StockPriceVec[0]*scipy.stats.norm(0,1).cdf(dPlus)-K*np.exp(-r*TimeRemain)*scipy.stats.norm(0,1).cdf(dMins)
    Reward[0] = OptionIniP*L-StockPriceVec[0]*round(DeltaVec[0][0]*L)-c*abs(OptionIniP*L+StockPriceVec[0]*round(DeltaVec[0][0]*L))
    for i in range(1,N-1):
        Reward[i] = StockPriceVec[i]*(round(L*DeltaVec[i-1][0])-round(L*DeltaVec[i][0]))-c*abs(StockPriceVec[i]*(round(L*DeltaVec[i-1][0])-round(L*DeltaVec[i][0])))
    CallPayoff = StockPriceVec[-1]-K if (StockPriceVec[-1]-K>0) else 0
    Reward[-1] = StockPriceVec[-1]*round(L*DeltaVec[-1][0])-c*abs(StockPriceVec[-1]*round(L*DeltaVec[-1][0]))-CallPayoff*L
    NetProfit = sum(Reward)
    return NetProfit,Reward

#test1
S0,r,sigma,delta_t,K=50,0,0.3,1/252,50
N = 5
M = 10**3
AvgProfit = np.zeros((M))
for i in range(M):
    S = GBMGenerator(S0,r,sigma,delta_t,N,100,0)
    D = DeltaHedge(S,K,r,sigma,delta_t)
    Profit,Reward = HedgeProfit(S,D,0,50,100)
    AvgProfit[i] = Profit
    if i%100 == 0:
        print("{}".format(i))
plt.figure(1, figsize=(12,8))
plt.grid()
plt.hist(AvgProfit,bins=40)
plt.title("deltaHedging",fontsize=25)
plt.tick_params(labelsize=20)
plt.show()
#AvgProfit.mean()
np.quantile(AvgProfit,0.1)
np.quantile(AvgProfit,0.3)
np.quantile(AvgProfit,0.5)
np.quantile(AvgProfit,0.7)
np.quantile(AvgProfit,0.9)
#test2        
allS = np.zeros((M,N,1))
for i in range(M):
    S = GBMGenerator(S0,r,sigma,delta_t,N,100,0)
    allS[i] = S
allS = allS.reshape((M*N,1))    
np.max(allS)
np.min(allS)
np.quantile(allS,0.001)
np.quantile(allS,0.999)
