# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:18:36 2021

@author: Léa
"""

import numpy as numpy
from scipy.stats import norm
import math
import scipy
import matplotlib.pyplot as plt
from numpy import matrix 
import pandas as pd
from numpy import linalg 
from scipy.ndimage.interpolation import shift
import copy
from scipy import interpolate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time as ti
import random
from numpy import random as rnd



def calibrationBS(X):
    """
    Calibrate implied vol.

    Parameters
    ----------
    X : TYPE double[]
        DESCRIPTION. BS parameters

    Returns
    -------
    TYPE double
        DESCRIPTION. difference of volatility

    """
    return X[5]-BlackScholes(X[0], X[1], X[2], X[3], X[4])

def BlackScholes(sigma, K, S, r, T):
    """
    Calculate an option price with BS formula.

    Parameters
    ----------
    sigma : TYPE double
        DESCRIPTION. vol
    K : TYPE doube
        DESCRIPTION. strike
    S : TYPE double
        DESCRIPTION. spot
    r : TYPE double
        DESCRIPTION. IR
    T : TYPE double
        DESCRIPTION. time until maturity

    Returns
    -------
    factor : TYPE double
        DESCRIPTION.option price

    """
    if T == 0:
        return max(S-K,0)
    else :
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1-(sigma * math.sqrt(T))
        factor = S * scipy.stats.norm(0, 1).cdf(d1) - K * math.exp(-r * T) * scipy.stats.norm(0, 1).cdf(d2)
        return factor
    
def newton_raphston(fct, x, N, i):
    """
    Calibrate the implied vol.

    Parameters
    ----------
    fct : TYPE fonction
        DESCRIPTION.fonction to reduce
    x : TYPE double[]
        DESCRIPTION. parameters of fonction
    N : TYPE int 
        DESCRIPTION.itteration max
    i : TYPE int
        DESCRIPTION. step

    Returns
    -------
    TYPE double
        DESCRIPTION. implied vol

    """
    m = numpy.array([i, 0, 0, 0, 0, 0])
    x[0] = x[0]-fct(x) * (i) / (fct(x)-fct(x-m))
    n = 1
    while fct(x) != 0 and abs(fct(x)) > 0.00001 and n < N and fct(x) != "false":
        x[0] = x[0] - fct(x) * (i) / (fct(x) - fct(x-m))
        n = n + 1
    return x[0]


def vol_imply_vect(price, spot, r, T, strike):
    """
    From a price vector, this function generates a vector of implied vol.

    Parameters
    ----------
    price : TYPE double[]
        DESCRIPTION.
    spot : TYPE double
        DESCRIPTION.
    r : TYPE double
        DESCRIPTION.
    T : TYPE double
        DESCRIPTION.
    strike : TYPE double
        DESCRIPTION.

    Returns
    -------
    vol_imp : TYPE  double[]
        DESCRIPTION. vector of implied vol

    """
    vol_imp = numpy.zeros(len(strike))
    for i in range(0, len(price)):
        x = [0.2, strike[i], spot, r, T, price[i]]
        vol_imp[i] = newton_raphston(calibrationBS, x, 100, 0.01)
    return vol_imp


def cvxt_interpolation(strike, vol, suite):
    """
    Interpolate mutliple point with cvxt.

    Parameters
    ----------
    strike : TYPE double[]
        DESCRIPTION.
    vol : TYPE double[]
        DESCRIPTION.
    suite : TYPE int[]
        DESCRIPTION. array of index to interpolate the points

    Returns
    -------
    list_params : TYPE double[,]
        DESCRIPTION. array of parameters

    """
    list_params = []
    i = 0
    mat = matrix([[strike[i + 2] ** 3, strike[i + 2] ** 2, strike[i + 2], 1],
                  [strike[i] ** 3, strike[i] ** 2, strike[i], 1],
                  [strike[i + 1] ** 3, strike[i + 1] ** 2, strike[i + 1], 1],
                  [6 * strike[i + 2], 2, 0, 0]])
    fir_part_deriv = (vol[i+3]-vol[i+2]) / (strike[i+3]-strike[i+2])
    sec_part_deriv = (vol[i+2]-vol[i+1]) / (strike[i+1]-strike[i])
    cvxt = (fir_part_deriv-sec_part_deriv) / (strike[i+3]-strike[i+1])
    y = matrix([[vol[i+2]], [vol[i]], [vol[i+1]], [cvxt]])
    res = numpy.ndarray.tolist((mat.I * y))
    abc = [item for sublist in res for item in sublist]
    list_params.append(abc)
    for i in suite:
        if i != 0:
            mat = matrix([[strike[i + 2] ** 2, strike[i + 2], 1],
                          [strike[i] ** 2, strike[i], 1],
                          [2, 0, 0]])
            fir_part_deriv = (vol[i+2]-vol[i+1]) / (strike[i+2]-strike[i+1])
            sec_part_deriv = (vol[i+1]-vol[i]) / (strike[i+1]-strike[i])
            cvxt = (fir_part_deriv-sec_part_deriv) / (strike[i+1]-strike[i])
            y = matrix([[vol[i + 2]], [vol[i]], [cvxt]])
            res = numpy.ndarray.tolist((mat.I * y))
            listabc = [item for sublist in res for item in sublist]
            listabc.insert(0, 0)
            list_params.append(listabc)
    return list_params



def polynome_order_three(params, suite, X, step, index_last_point):
    """
    Calculate the polynome degree 3.

    Parameters
    ----------
    params : TYPE double[,]
        DESCRIPTION. matrix coeficient polynomes
    suite : TYPE
        DESCRIPTION. index of the second method interpolated
    X : TYPE double[]
        DESCRIPTION. strikes
    step : TYPE
        DESCRIPTION. precision
    index_last_point : TYPE int
        DESCRIPTION. index last point of the polynome

    Returns
    -------
    Y : TYPE double[]
        DESCRIPTION. curve

    """
    i = 0
    lastsuite = 0
    p = -1
    Y = numpy.zeros(len(X))
    list_index = suite.copy()
    list_index.append(index_last_point)
    occurence = int(1 / step)
    for i in (list_index):
        for k in range(lastsuite, i):
            for j in range(occurence):
                variable = params[p, 0] * X[k * occurence + j] ** 3
                variable = variable + params[p, 1] * X[k * occurence + j] ** 2
                variable = variable + params[p, 2] * X[k * occurence + j]
                variable = variable + params[p, 3]
                Y[k * int(1 / step) + j] = variable
        p = p + 1
        lastsuite = i
    return Y



#Part 2
def interpolation_Y_axis(X, Y, suite, i, j):
    params = []
    mat = matrix([[X[i] ** 3,X[i] ** 2,X[i],1],
                  [X[i+1] ** 3,X[i+1] ** 2,X[i+1],1],
                  [X[i + 2] ** 3,X[i + 2] ** 2,X[i + 2],1],
                  [X[i + 3] ** 3,X[i + 3] ** 2,X[i + 3],1]])
    
    y = matrix([[Y[0][j]], [Y[1][j]], [Y[2][j]],[Y[3][j]]])
    res = numpy.ndarray.tolist((mat.I * y))
    abc = [item for sublist in res for item in sublist]
    params.append(abc)   
    return params


def cholesky(vect,number,hurst):
    cov = numpy.zeros([number, number])
    if hurst == 0:
        print(hurst)
    for i in range(0,number):
        for j in range(0,i + 1):
            t = i-j
            cov[i,j] = (abs(t-1) ** (2 * hurst) - 2 * abs(t) ** (2 * hurst) + abs(t + 1) ** (2 * hurst)) / 2
    cho = numpy.linalg.cholesky(cov) 
    return (numpy.dot(cho, numpy.array(vect).transpose()))


def brownian_motion(nb,taille,hurst):
    vect = numpy.random.normal(0,1,taille)
    if hurst != 1 / 2:
        vect = cholesky(vect,taille,hurst)
    test = vect * (taille) ** hurst
    return numpy.insert(test, [0], 0)



def neldermead(funct, X,step,stop_stagne,stop, ittBreak,X_params):
    nb_params = len(X)
    F0 = funct(X,X_params)
    m = 0
    k = 0
    refl = 1
    exp = 2
    contr = -1 / 2
    red = 0.5
    params = [[X, F0]]
    for i in range(nb_params):
        vect = copy.copy(X)
        vect[i] = vect[i] + step
        params.append([vect,funct(vect,X_params)])
    while 1:
        params.sort(key = lambda x: x[1])
        F = params[0][1]
        if ittBreak<= k :
            print(k)
            return params[0]
        k = k + 1
        if F<F0-stop_stagne:
            m = 0
            F0 = F
        else:
            m = m + 1
        if m >= stop:
            print(k)
            return params[0]
            
        centroid = [0.] * nb_params
        for cen in params[:-1]:
            for i, c in enumerate(cen[0]):
                centroid[i] += c / (len(params)-1)
        newParam_refl = centroid + refl * (centroid - numpy.array(params[-1][0]))
        refl_r = funct(newParam_refl,X_params)
        if params[0][1] <= refl_r < params[-2][1]:
            del params[-1]
            params.append([newParam_refl, refl_r])
            continue
        if refl_r < params[0][1]:
            newParam_exp = centroid + exp * (centroid - numpy.array(params[-1][0]))
            exp_e = funct(newParam_exp,X_params)
            if exp_e < refl_r:
                del params[-1]
                params.append([newParam_exp,exp_e])
                continue
            else:
                del params[-1]
                params.append([newParam_refl, refl_r])
                continue
        newParam_contr = centroid + contr * (centroid -  numpy.array(params[-1][0]))
        contr_c = funct(newParam_contr,X_params)
        if contr_c < params[-1][1]:
            del params[-1]
            params.append([newParam_contr, contr_c])
            continue
        par = params[0][0]
        params2 = []
        for li in params:
            newParam_red = par + red * (li[0] - numpy.array(par))
            red_r = funct(newParam_red,X_params)
            params2.append([newParam_red, red_r])
        params = params2
        
def montecarlo_stochastic(volvol, v0, r, T, spot, strike, n, N,hurst):
    """
    Calculate option price with monte carlo method.

    Parameters
    ----------
    volvol : TYPE double
        DESCRIPTION. vol of variance model
    v0 : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    spot : TYPE
        DESCRIPTION.
    strike : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    hurst : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE double
        DESCRIPTION. option rice

    """
    dt = T / n
    N = int(N)
    n = int(n)
    hurst = abs(hurst)
    S = numpy.repeat(float(spot),(n))
    var = numpy.repeat(float(v0),(n))
    price = 0
    for i in range(1, int(N)):
        W1 = numpy.random.rand(n)
        W2 = brownian_motion(n,n-1, hurst)
        for j in range(1,int(n)):
            var[j] = v0 * numpy.exp(volvol * numpy.sqrt(numpy.absolute(var[j-1] * dt * i) ) * W2[j])
            S[j] = S[0] * numpy.exp((r-numpy.absolute(var[j] / 2)) * dt * i + numpy.sqrt((numpy.absolute(var[j] * dt * i))) * W1[j])
        if not math.isnan(numpy.maximum((S[len(S)-1]-strike),0) * numpy.exp(-r * T)) and not math.isinf(numpy.maximum((S[len(S)-1]-strike),0) * numpy.exp(-r * T)):
            price = price + (numpy.maximum((S[len(S)-1]-strike),0)) * numpy.exp(-r * T)
        else : 
            N=N-1
    return price/N  


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 3 méthodes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def recuit(Xp,start,stop,Xparams,dist):
    X = [i for i in range(0,len(Xp))]
    X_sol = copy.copy(X)
    X_dist = dist(Xp,Xparams)
    #X_sol.append(X_sol[0])
    i = 0
    while start>stop:
        value = [abs(np.random.normal(Xp[i],0.1)) for i in range(len(Xp))]
        if(value[2]<0):
            value[2] = abs(value[2])
        if (value[2])>1 :
            value[2] = abs(1 / value[2])
        if dist(value,Xparams)<X_dist:
            X_sol = copy.copy(value)
            X_dist = dist(value,Xparams)
            X = copy.copy(value)
            X_sol.append(X_sol[0])
        else:
            if rnd.rand()<numpy.exp(-abs(dist(value,Xparams)-dist(Xp,Xparams)) / start):
                Xp = copy.copy(value)
            #refroidissement
            start = start * 0.95
        i = i + 1
    print(i)
    return(value)

def funct_recuit(X,Y): #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    return abs(Y[9] - montecarlo_stochastic(Y[0], Y[1], X[1], Y[2], X[0], Y[3],  Y[4],  Y[5],  Y[6],  Y[7],Y[8], X[2]))




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 3 
We test the performance of the calibration recuit algorithm vs Nelder Mead
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
strike = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
price = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]
r = 0
T = 1
spot = 100
step = 0.01

rho = 1
v0 = 0.13
hurst = 0.25
n = 30
N = 500
returnmean = 1
volvol = 0.2
vbar = 1
maturite = 7 / 12
striketest = 98.3
r = 0

stri = round((striketest-95) * 10)
matu = round(maturite * 100) 


spot = 100
step = 0.01
price9m = [11.79,8.95,8.07,7.03,6.18,6.04,5.76,5.50,5.50,5.39]
price6m = [10.71,8.28,6.91,6.36,5.29,5.07,4.76,4.47,4.35,4.14]
price3m = [8.67,7.14,5.98,4.93,4.09,3.99,3.43,3.01,2.72,2.53]
strike = [95,96,97,98,99,100,101,102,103,104]
price12m = [12.40,9.59,8.28,7.40,6.86,6.58,6.52,6.49,6.47,6.46]
m = 0
sumtime = 0
sumtimenelder = 0
allprice = [price9m,price6m,price3m,price12m]
MSEnelder = 0
MSErec = 0
indexStrike = 33
indexMat = 58 

rho = 1
v0 = 0.13
hurst = 0.25
n = 20
N = 500
returnmean = 1
volvol = 0.2
vbar = 1


vol9m = vol_imply_vect(price9m,spot,r,T,strike)
vol6m = vol_imply_vect(price6m,spot,r,T,strike)
vol3m = vol_imply_vect(price3m,spot,r,T,strike)
vol12m = vol_imply_vect(price12m,spot,r,T,strike)

suite = [0,2,3,4,5,6]
step = 0.1
X = [strike[0] + i * step for i in range((len(strike)-1) * int(1 / step))]

poly9m = cvxt_interpolation(strike,vol9m,suite)
poly6m = cvxt_interpolation(strike,vol6m,suite)
poly3m = cvxt_interpolation(strike,vol3m,suite)
poly12m = cvxt_interpolation(strike,vol12m,suite)

VOL9m = polynome_order_three(numpy.array(poly9m),suite,X,step,9)
VOL6m = polynome_order_three(numpy.array(poly6m),suite,X,step,9)
VOL3m = polynome_order_three(numpy.array(poly3m),suite,X,step,9)
VOL12m = polynome_order_three(numpy.array(poly12m),suite,X,step,9)   
suite = [0]
step2 = 0.01
nappe = numpy.zeros(shape = (90,100))
time = [1 / 4,1 / 2,9 / 12,1]
Xtime = [0 + i * 0.01 for i in range(int(1 / 0.01))]

for i in range(len(VOL9m)):
    y_interpolation = interpolation_Y_axis(time,
                                   [VOL3m,VOL6m,VOL9m,VOL12m],
                                   suite,
                                   0,
                                   i)
    nappe[i,:] = polynome_order_three(numpy.array(y_interpolation),
                               suite,
                               Xtime,
                               step2,
                               1)
    
nappeprice = numpy.zeros(shape = (90,100))
for i in range(len(VOL12m)):
    for j in range(len(Xtime)):
        nappeprice[i,j] = BlackScholes(nappe[i,j],X[i],spot,r,Xtime[j])


for priceMM in allprice:
    m = m + 3
    for i in range(0,len(priceMM)):
        Xparams = [0, 0,0, r, m / 12 , spot,  strike[i], n, N,priceMM[i]]
        x = [v0,volvol,hurst]
        tic = ti.perf_counter()
        res = recuit(x,200,25,Xparams,funct_recuit)
        toc = ti.perf_counter()
        sumtime = sumtime + toc-tic
        print("Time REC :",str(toc-tic))
        tic = ti.perf_counter()   
        v022 = neldermead(funct_recuit, x,step,0.0001,60, 60,Xparams)
        toc = ti.perf_counter()
        sumtimenelder = sumtimenelder + toc-tic
        print("Time nelder :",str(toc-tic))
        MSErec = MSErec + abs(priceMM[i]-montecarlo_stochastic(0, 0,  res[1], 0, res[0], r, m / 12 , spot,strike[i], n, N, res[2]) ) ** 2
        if MSErec == np.NaN :
            print("stop")
        MSEnelder = MSEnelder + abs( priceMM[i]-montecarlo_stochastic(0, 0,  v022[0][1], 0, v022[0][0], r, m / 12 , spot, strike[i], n, N, v022[0][2]) ) ** 2
print(MSErec / 40)
print(MSEnelder / 40)
print(sumtime / 40)
print(sumtimenelder / 40)



N = 1000

tic = ti.perf_counter()
Xparams = [returnmean, vbar,rho, r, matu, spot, striketest, n, N,nappeprice[indexStrike,indexMat]]
x = [0.13107441, 0.22117992, 0.27079042]
res = recuit(x,150,30,Xparams,funct_recuit)
print(res)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#32
#[0.0647841306310231, 0.12467694672221002, 0.18197914351336303, 0.0647841306310231]
#Time : 1100.02699079999

#37
#[0.09465733630441449, 0.34437939284227426, 0.028653993403438903, 0.09465733630441449]
#Time : 300.5942544999998
pv = [0.11502436622966557, 0.04614849629086123, 0.0850012103018249]

tic = ti.perf_counter()
v022 = neldermead(funct_recuit, x,step,0.0001,10, 37,Xparams)
print(v022)
toc = ti.perf_counter()
print("Time :",str(toc-tic))
print(step)


# step 14
# [array([0.15182195, 0.23658533, 0.25168949]), 0.07662085143057329]
#Time : 31.92447140000877

#12
#[array([0.25314815, 0.17253086, 0.21265432]), 0.16194142348826723]
#Time : 667.2252187999984

N = 3000
esa = montecarlo_stochastic(returnmean, vbar,  pv[1], rho, pv[0], r,matu, spot, striketest, n, N, pv[2]) 
print(esa)
    
esa = montecarlo_stochastic(returnmean, vbar,  v022[0][1], rho, v022[0][0], r,matu, spot, striketest, n, N, v022[0][2]) 
print(esa)



       