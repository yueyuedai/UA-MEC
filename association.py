# This program is to generate 10 SBS and 1 Macrocell and 50 Users

######## Create the 5*4 APs ##########
#AP & User's location
import math
import numpy as np
import numpy
import sys
import random
import matplotlib.pyplot as plt
#import cvxopt.modeling
from cvxopt.modeling import variable

def Distance(M,N,b0):
    # The MBS is located at (500m,500m)
    location_MBS = [[500.0,500.0]]

    # The SBS is uniform distributed at U(50m,950m)
    location_SBS = []
    for i in range(M):
        location_SBS.append([random.uniform(50.0,950.0),random.uniform(50.0,950.0)])

    # The mobile user is uniform distributed at U(0m,1000m)
    location_MU = []
    for i in range(N):
        location_MU.append([random.uniform(0,1001.0),random.uniform(0,1001.0)])

    '''
    #Plot the scene
    plt.plot(location_MBS[0][0],location_MBS[0][1],'go')

    for j in range(M):
        plt.plot(location_SBS[j][0],location_SBS[j][1],"bd")

    for j in range(N):
        plt.plot(location_MU[j][0],location_MU[j][1],"r*")

    plt.show()
    '''
    return location_MBS,location_MU,location_SBS


def Daterate_MBS_MU(BS,U,P,noise,Bandwidth,X_ij):
    rate=[]
    l_ij = np.sum(X_ij,axis=0) 
    l_MBS_ij = l_ij[0]
    for i in range(len(U)):
        rate.append([])
        for j in range(len(BS)):
            #First, compute the distance between BS and MU,1*40, 10*40
            dist = ((BS[j][0]-U[i][0])**2+(BS[j][1]-U[i][1])**2)**0.5 # 0-x;1-y 
            
            #Second, compute the channel gain
            gain = 128.1+37.6*math.log10(dist/1000)
            
            #Third, compute the SINR without interference
            gain1 = 10.0**(gain/10.0)
            SINR= (P[i]*gain1)/noise
            
            #Forth, compute the uplink data rate
            r = Bandwidth*(math.log(1+SINR)/math.log(2))#Mbps
            r1 = r/l_MBS_ij
            rate[i].append(r1)
    return rate


def channel_gain_SBS_MU(BS,U):
    channel_gain=[]
    for i in range(len(U)):
        channel_gain.append([])
        for j in range(len(BS)):
            #First, compute the distance between BS and MU,1*40, 10*40
            dist = ((BS[j][0]-U[i][0])**2+(BS[j][1]-U[i][1])**2)**0.5 # 0-x;1-y 
            #Second, compute the channel gain
            gain = 140.7+36.7*math.log10(dist/1000)
            channel_gain[i].append(gain)
    return channel_gain

def Daterate_SBS_MU(M,N,channelgain_SBS_MU,Interference_ij,P,noise,Bandwidth,X_ij):
    rate=[]
    l_ij = np.sum(X_ij,axis=0) 
    l_SBS_ij = l_ij[1:M+1]
    for i in range(N):
        rate.append([])
        for j in range(M):
            #First, obtain the channel gain
            gain = 10**(channelgain_SBS_MU[i][j]/10)
            
            #Second, compute the SINR with interference
            SINR= (P[i]*gain)/(noise+Interference_ij[i][j])
            
            #Third, compute the uplink data rate
            r = Bandwidth*(math.log(1+SINR)/math.log(2))#Mbps
            r1 = r/l_SBS_ij[j]
            rate[i].append(r1)
    return rate


def Communication_datarate (M,N,b0,location_MBS,location_MU,location_SBS,X_ij,noise,Bandwidth_MBS,Bandwidth_SBS,P_i):
    #The SINR and date rate calculation
       # The large-scale change model is 
          # MBS-MU:128.1+37.6log10(r)
          # SBS-MU:140.7+36.7log10(r)

    # Frist, compute the SINR of MSB-MU without Interference
    Date_rate_MBS_MU =  Daterate_MBS_MU(location_MBS,location_MU,P_i,noise,Bandwidth_MBS,X_ij)

    # Second, compute the SINR of MSB-MU without Interference
    #    1. Interfernce
    #    2. Data rate and According to user association, we reformulate the data rate.
    channelgain_SBS_MU = channel_gain_SBS_MU(location_SBS,location_MU)

    #1. Interfernce
    Interference_ij = []
    for i in range(N):
        Interference_ij.append([])
        for j in range(M):
            I_ij = 0
            for i1 in range(N):
                if i1 != i:
                    I_ij = I_ij + P_i[i1]*channelgain_SBS_MU[i1][j]
            Interference_ij[i].append(I_ij)
    
    #2. Data rate
    Date_rate_SBS_MU = Daterate_SBS_MU(M,N,channelgain_SBS_MU,Interference_ij,P_i,noise,Bandwidth_SBS,X_ij)
    return Date_rate_MBS_MU,Date_rate_SBS_MU

def computation_sequential(M,N,b0,R_i0,R_ij,K,T_i_max,d_i_k,P_i,f_i_k,f_ij_k,d_ij_k,d1_ij_k,c_i_k,F_i,F_j,f_0,e_j,e_0,r_0,sigma,delta):
    # The computation model includes local computing, SBS computation, MBS computation
    # Local Computing:
    T_i_L = []# The task execution time via local computing
    E_i_L = []# The task energy consumption via local computing
    for i in range(N):
        for j in range(M+1):
            if d_ij_k[i][j] !=[]:#d_ij_k[i][j] !=[] means x_ij = 1 that is MU i is associated with BS j.
                T_i_L_k = []
                E_i_L_k = []
                for k in range(K):
                    T_i_L_k.append((d_i_k[i][k]-d_ij_k[i][j][k])*c_i_k[i][k]/(f_i_k[i][k]))
                    E_i_L_k.append(sigma*(d_i_k[i][k]-d_ij_k[i][j][k])*c_i_k[i][k]*((f_i_k[i][k])**2))
        T_i_L.append(T_i_L_k)
        E_i_L.append(E_i_L_k)
    
    #SBS Computing: (Note the index of the SBS is from [0,M).   )
    T_ij_SBS = []# The task execution time via SBS computing
    E_ij_SBS = []# The task energy consumption via SBS computing
    for i in range(N):
        T_ij_SBS.append([])
        E_ij_SBS.append([])
        for j in range(M):
            T_ij_SBS[i].append([])
            E_ij_SBS[i].append([])
            for k in range(K):
                j1 = j+1
                if X_ij[i][j1] == 1:#d_ij_k[i][j] !=[] means x_ij = 1 that is MU i is associated with BS j.
                    #First, calculate the delay
                    t1 = (d_ij_k[i][j1][k])/(R_ij[i][j])# the wireless transmission time 
                    t2 = (d_ij_k[i][j1][k]-d1_ij_k[i][j1][k])*c_i_k[i][k]/f_ij_k[i][j][k]
                    t3 = (d1_ij_k[i][j1][k])/r_0
                    t4 = d1_ij_k[i][j1][k]*c_i_k[i][k]/f_0
                    t = t1+t2+t3+t4
                    #print t1,t2
                    #Second, calculate the energy
                    e1 = (P_i[i]*d_ij_k[i][j1][k])/(R_ij[i][j])# # the wireless transmission energy
                    e2 =(100*10**(3)*8)*c_i_k[i][k]*e_j
                    e3 = delta*(d1_ij_k[i][j1][k])/r_0
                    e4 = d1_ij_k[i][j1][k]*c_i_k[i][k]*e_0
                    e = e1+e2+e3+e4
                else:
                    t = 0.0
                    e = 0.0
                T_ij_SBS[i][j].append(t)
                E_ij_SBS[i][j].append(e)
    
    # MBS computing
    T_i0_MBS = []# The task execution time via MBS computing
    E_i0_MBS = []# The task energy consumption via MBS computing
    for i in range(N):
        T_i0_MBS.append([])
        E_i0_MBS.append([])
        for j in range(b0):
            T_i0_MBS[i].append([])
            E_i0_MBS[i].append([])
            for k in range(K):
                j1 = j
                if X_ij[i][j1] == 1:#d_ij_k[i][j] !=[] means x_ij = 1 that is MU i is associated with BS j.
                    #First, calculate the delay
                    t1 = (d_ij_k[i][j1][k])/(R_ij[i][j])# the wireless transmission time 
                    t4 = d1_ij_k[i][j1][k]*c_i_k[i][k]/f_0
                    t = t1+t4
                    #print t1,t2
                    #Second, calculate the energy
                    e1 = (P_i[i]*d_ij_k[i][j1][k])/(R_ij[i][j])# # the wireless transmission energy
                    e4 = d1_ij_k[i][j1][k]*c_i_k[i][k]*e_0
                    e = e1+e4
                else:
                    t = 0.0
                    e = 0.0
                T_i0_MBS[i][j].append(t)
                E_i0_MBS[i][j].append(e)
    return T_i_L,E_i_L,T_ij_SBS,E_ij_SBS,T_i0_MBS,E_i0_MBS

def User_association(N,M,b0,K,T_L_k,T_SBS_k,E_SBS_k,T_MBS_k,E_MBS_k,f_ij_k):
    # Define the varible
    length = M+b0+K

    #w_i = []
    q_i = []
    for i in range(N):
        #w_i.append([])
        q_i.append([])
        for j in range(M+b0):
            #w_i[i].append(variable(1,'x'))
            q_i[i].append(variable(1,'x'))
        for j in range(K):
            #w_i[i].append(variable(1,'t'))
            q_i[i].append(variable(1,'t'))
        q_i[i].append(1.0)
        #w_i.append(variable(K,'t'))
    '''
    print length


    print len(q_i),len(q_i[0])
    print q_i
    '''

    #Define the b_i~b_ij^f
    E_i0_MBS = []
    E_ij_SBS = []
    for  i in range (N):
        E_i0_MBS.append([])
        for j in range(M):
            E_i0_MBS[i].append(sum(k for k in E_MBS_k[i][0]))

    
    for  i in range (N):
        E_ij_SBS.append([])
        for j in range(M):
            E_ij_SBS[i].append(sum(k for k in E_SBS_k[i][j]))

    #QCQP Transformation
    b_i = [] #   N*(M+K+1)
    b_i_ti = [] #   N*(M+K+1), 
    b_ik_L = []  #  (N*K)*(M+K+1) for each user this vector is the same,so we just produce one of them
    b_ik_BS = [] # (N*K)*(M+K+1)
    b_i_x = []
    b_ij_f = []
    # This is for b_i,b_i_ti,b_i_x
    for i in range(N):
        b_i.append([])
        b_i_ti.append([])
        b_i_x.append([])
        for j in range(b0):
            b_i[i].append(E_i0_MBS[i][b0])
            b_i_ti[i].append(0.0)
            b_i_x[i].append(1.0)
        for j in range(M):
            b_i[i].append(E_ij_SBS[i][j])
            b_i_ti[i].append(0.0)
            b_i_x[i].append(1.0)
        for k in range(K):
            b_i[i].append(0.0)
            b_i_ti[i].append(1.0)
            b_i_x[i].append(0.0)

    # This is just for b_ik_L,
    for  k in range(K):
        b_ik_L.append([])
        for j in range(M+k+1):
            b_ik_L[k].append(0.0)
        b_ik_L[k].append(-1.0)
        for k1 in range(K-1-k):
            b_ik_L[k].append(0.0)
    
     #This is just for b_ik_BS
    for i in range(N):
        b_ik_BS.append([])
        for k in range(K):
            b_ik_BS[i].append([])
            for j in range(b0):
                b_ik_BS[i][k].append(T_MBS_k[i][j][k])
            for j in range(M):
                b_ik_BS[i][k].append(T_SBS_k[i][j][k])
            for k1 in range(k+1-1):
                b_ik_BS[i][k].append(0.0)
            b_ik_BS[i][k].append(-1.0)
            for k2 in range(K-1-k):
                b_ik_BS[i][k].append(0.0)

    # This is for the b_ij_f
    for i in range(N):
        b_ij_f.append([])
        for j in range(M):
            b_ij_f[i].append([])
            for j1 in range(j+1):
                b_ij_f[i][j].append(0.0)

            f_ij = 0
            for k in range(K):
                f_ij = f_ij+f_ij_k[i][j][k]
            b_ij_f[i][j].append(f_ij)
            for j2 in range(M-j-1+K):
                b_ij_f[i][j].append(0.0)
        
    '''
    print "SBS computing "
    print "SBS Energy",E_SBS_k

    print "MBS computing "
    print "MBS Energy",E_MBS_k
    '''
    #q_iA_iq_i
    A_i = [] #   N*(M+K+1)
    A_i_ti = [] #   N*(M+K+1), 
    A_ik_L = []  #  (N*K)*(M+K+1) for each user this vector is the same,so we just produce one of them
    A_ik_BS = [] # (N*K)*(M+K+1)
    A_i_x = []
    A_ij_f = []
    A_i_e = []
    bi = np.transpose(b_i)
    '''
    print len(b_i),len(b_i[0])
    print b_i
    print len(bi),len(bi[0])
    print bi
    '''
    z1 = [] #(M+K+1)*(M+K+1)
    z2 = numpy.zeros(N) #(4*1) add at bi
    for i in range(M+K+1):
        z1.append(numpy.zeros(M+K+1))
    
    bi = np.vstack((bi,z2))

    
    for i in range(N):
       A_i_1 = np.vstack((z1,0.5*np.array(b_i[i])))
       A_1 = np.c_[A_i_1,bi[:,i]]
       A_i.append(A_1)
    print len(A_i_1),len(A_i_1[0])
    print A_i_1
    print len(A_i),len(A_i[0]),len(A_i[0][0])
    print A_i
  
  

###############################
# The parameter about the scene, where are one macro base station, 10 small base station, and 40 mobile users.
N = 4 # The number of Mobile Users
b0 = 1 # The number of Macro Base Station
M = 2 # The number of Small Base Station
K = 3

# The parameter for the channel 
noise =  10**(-14)#Watt 10^(-11)mWatt
Bandwidth_MBS = 10*10.0**(6) #10 MHz
Bandwidth_SBS = 5*10.0**(6)  #5 MHz
p_i = 1 #Watt

# The parameter about the application which is consisted by 3 separable tasks. Each task is uniformed, U[200,500]KB.
F_i = 1*10**(9) #1GHz = 1Giga cycles
F_j = 20*10**(9) #20GHz
f_0 = 10*10**(9) #10GHz, this is a pre-determined value

e_j = 1*10**(-9) #Watt/GHz
e_0 = 1*10**(-9) #Watt/GHz
r_0 = 1*10**(9) #Gbps
sigma = 1*10**(-24-3)#translating to Watt has to multiply 10^(-3) 
delta = 1*10**(-3) #1mW = 10^(-3)Watt



T_i_max = []#The delay constraint
for i in range(N):
    T_i_max.append( random.uniform(0.5,1) )# second


d_i_k = []# The input data size, KB
c_i_k = [] #500~1000cycles per bit
for i in range(N):
    d_i_k.append([]) #the i-th MU
    c_i_k.append([])
    for k in range(K):
        d_i_k[i].append(random.uniform(200000*8,500000*8))# KB
        c_i_k[i].append(random.uniform(500,1000))# CPU cycles/bit


#Inicialize the user association
X_ij = [[1,0,0],[0,1,0],[0,0,1],[1,0,0]]# This is the association inicialization which may be changed later.

#Inicailize the computation resource allcoation
f_i_k = []
f_ij_k = []
for i in range(N):
    f_i_k.append([])
    for k in range(K):
        f_i_k[i].append(F_i)

loa = np.sum(X_ij,axis=0)
load = loa[1:M+1]

for i in range(N):
    f_ij_k.append([])
    for j in range(M):
        f_ij_k[i].append([])
        for  k in range(K):
            f_ij_k[i][j].append(F_j/(load[j]*K))

#Inicialize the power allocation
P_i = []
for i in range(N):
    P_i.append(p_i) 

#Inicialize the offloading partition
d_ij_k = []
#d_i0_k = []

d1_ij_k = []
#d1_i0_k = []

for i in range(N):
    #d_i0_k.append([])
    #d1_i0_k.append([])
    d_ij_k.append([])
    d1_ij_k.append([])
    for j in range(M+1):
        #d_i0_k[i].append([])
        #d1_i0_k[i].append([])
        d_ij_k[i].append([])
        d1_ij_k[i].append([])
        for k in range(K):
            if (X_ij[i][j] == 1):
                d = d_i_k[i][k]/2
                d_ij_k[i][j].append(d)
                d1_ij_k[i][j].append(d/2)


# The scene,  channel model, and data rate--seen the communication model
loc_MBS,loc_MU,loc_SBS = Distance(M,N,b0)
R_i0,R_ij = Communication_datarate (M,N,b0,loc_MBS,loc_MU,loc_SBS,X_ij,noise,Bandwidth_MBS,Bandwidth_SBS,P_i)# Unit, Mbps

#print "MBS_MU rate",R_i0
#print "SBS_MU rate",R_ij

# The computation model 
T_L_k,E_L_k,T_SBS_k,E_SBS_k,T_MBS_k,E_MBS_k = computation_sequential(M,N,b0,R_i0,R_ij,K,T_i_max,d_i_k,P_i,f_i_k,f_ij_k,d_ij_k,d1_ij_k,c_i_k,F_i,F_j,f_0,e_j,e_0,r_0,sigma,delta)
'''
print "Local "
print "Local Time",T_L
print "n"
print "Local Energy",E_L

print "SBS computing "
print "SBS Time",T_SBS
print "n"
print "SBS Energy",E_SBS

print "MBS computing "
print "MBS Time",T_MBS
print "n"
print "MBS Energy",E_MBS
'''
# We decompose the optimization problem into two subproblems:
# 1.User association
# 2.Joint optimization of offloading and resource

#User association
User_association(N,M,b0,K,T_L_k,T_SBS_k,E_SBS_k,T_MBS_k,E_MBS_k,f_ij_k)
