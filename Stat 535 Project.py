"""
Created on Thu Nov 14 19:02:07 2019

@author: User
"""

'''STAT 535 NPV is Life
Alvaro Callejas
Addis Gunst
Alexander Kaim
Jesse Pulselli
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KernelDensity

start_time=time.time()

import itertools as itrS
sed=70

def sim(p=0.5, ph=0.7, pl=0.8, n=20,sd=70):
    """Funtion that simulates data, ph represents probability of going from 
    high to high, pl represents probability of going from low to low, n is the
    length of the realization, p is probability of observing high on t0."""
    random.seed(sd)
    te=np.array(random.sample(range(1000),n-1))/1000#random draw
    random.seed(sd)
    rea=random.choices('HL',weights=(p,1-p),k=1) #initial realization at t0
    if rea[0]=='H':
        jp=p
    else:
        jp=round((1-p),2)   
    for i in range(n-1): #compute realization from t1 to tn
        temp=te[i]
        if rea[-1]=='L': #using probability of low to low to be compared with draw
            if temp<=pl:
                rea.append('L')
                jp=jp*pl
            else:
                rea.append('H')
                jp=jp*round((1-pl), 2)
        else: #using probability of high to high to be compared with draw
            if temp<=ph:
                rea.append('H')
                jp=jp*ph
            else:
                rea.append('L')
                jp=jp*round((1-ph), 2)
    return (rea, jp)

  
def NPV(B,C,BV,CV,d,pb,pc):
    """Function that computes the net present value using revenue, cost, 
    discount rate, revenue values, and cost values."""
    b=[BV[0] if x=='L' else BV[1] for x in B] #decoding revenue
    c=[CV[0] if x=='L' else CV[1] for x in C] #decoding cost
    z=[b_i - c_i for b_i, c_i in zip(b, c)] #profit at each time
    npv=np.npv(d, z)
    pnpv=pb*pc
    return (npv,pnpv)
        
def ind_sim(n,CV,BV,N,p,d):
    """Function that simulates data where the variables are independent."""     
    dic={}
    dic2={}
    for i in range(N):
        Bt=random.choices('HL', weights=(p,1-p), k=n)
        pb=[round((1-p), 5) if x=='L' else p for x in Bt] 
        Ct=random.choices('HL', weights=(p,1-p), k=n)
        pc=[round((1-p), 5) if x=='L' else p for x in Ct] 
        [npvt,pr]=NPV(Bt,Ct,BV,CV,d,np.prod(pb),np.prod(pc))
        if npvt in dic.keys():
            dic[npvt] += 1
        else:
            dic[npvt] = 1
            dic2[npvt] =pr
    return (dic, dic2)

def dep_sim(real,npr,BV,CV,n):
    """Function that calculate the theoretical probability asuming dependency
    among diferent realizations"""
    s_npv=""
    pb1=1
    pc1=1
    while s_npv!=real:
        Bt=random.choices('HL', weights=(.5,.5), k=n)
        Ct=random.choices('HL', weights=(.5,.5), k=n)
        [s_npv,ppr]=NPV(Bt,Ct,BV,CV,d,pb1,pc1)
    pb=join_p(Bt,npr)
    pc=join_p(Ct,npr)
    return (pb*pc)


# Function that calculates joint probability    
def join_p(Bt,npr):
    n=len(Bt)
    p=npr[0]
    ph=npr[1]
    pl=npr[2]
    if Bt[0]=='H':
        jp=p
    else:
        jp=round((1-p),2)   
    for i in range(n-1): #compute realization from t1 to tn
        if Bt[i]=='L': #using probability of low to low to be compared with draw
            if Bt[i+1]=='L':
                jp=jp*pl
            else:
                jp=jp*round((1-pl), 2)
        else: #using probability of high to high to be compared with draw
            if  Bt[i+1]=='H':
                jp=jp*ph
            else:
                jp=jp*round((1-ph), 2)
    return(jp)

random.seed(5)   
seed_r=np.array(random.sample(range(10000),1000))
seed_c=np.array(random.sample(range(10000),1000))
prob=[i/100 for i in range(0,101,10)]
n=6
it=0
d=0.04 #discount rate 
BV=[12000, 36000] #revenue values
CV=[6000, 18000] #cost values


lkp=[]
rea_pro=[]

start_time=time.time()
for j in range(len(seed_r)):
    print(j)
    sdd_r=seed_r[j]
    sdd_c=seed_c[j]
    [B,pb]=sim(p=0.5, ph=0.7, pl=0.8, n=n,sd=sdd_r) #simulation of revenue
    [C,pc]=sim(p=0.5, ph=0.7, pl=0.8, n=n,sd=sdd_c) #simulation of cost
    [real, pnpv]=NPV(B,C,BV,CV,d,pc,pb)
    f=[]
    jo_p=[]
    sim_num=10000
    for pr in prob:
        [x,y] = ind_sim(n,CV,BV, sim_num, pr,d)
        if real in x:
            f.append(x[real]/sim_num)
            jo_p.append(y[real])
        else:
            f.append(0)
            jo_p.append(None)
    lkp.append(jo_p[np.argmax(f)])
    rea_pro.append(pnpv)

print(time.time()-start_time)

relation= np.matrix([(x-y)/y for x, y in zip(lkp, rea_pro)])
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(relation.transpose())
bins = np.linspace(-5, 225, 100)[:, np.newaxis]
log_dens = np.exp(kde.score_samples(bins))

plt.plot(bins[:, 0],log_dens,'r')
plt.axvline(np.mean(log_dens),color='g',linestyle ='--',linewidth=6)
plt.axvline(np.median(log_dens),color='b',linestyle =':',linewidth=2)
plt.legend(['Probability density function', 'Mean', 'Median'])
plt.ylabel('Probability density')
plt.xlabel('Porcentual difference between actual and estimated probability of the sim. NPV')
plt.title('Probability density funtion diff. between actual and estimated probability of the sim. NPV')
plt.rcParams.update({'font.size': 55})

plt.plot(prob,f,'r', label='Empirical probability')
plt.plot(prob,jo_p,'b',label='Teorical probability')
plt.axvline(prob[np.argmax(f)],color='g',linestyle ='--')
plt.axhline(pnpv,color='y',linestyle ='--')
plt.legend(['Likelihood', 'Theorical probability', 'Maximum likelihood', 'Join Probability of Sim NPV'])
plt.ylabel('Probability of simulated NPV')
plt.xlabel('pH parameter')
plt.title('Estimated vs. actual probability of simulated NPV')
plt.rcParams.update({'font.size': 55})
