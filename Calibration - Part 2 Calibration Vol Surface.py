# - * - coding: utf-8 - * -
"""
Created on Wed Dec  2 11:14:28 2020

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 1 méthodes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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


def BlackScholes_continous_vol(spot, strike, vol, r, T):
    """
    Calculate BS vector from a continuous vol vector and strike vector.

    Parameters
    ----------
    spot : TYPE double
        DESCRIPTION.
    strike : TYPE double[]
        DESCRIPTION.
    vol : TYPE double[]
        DESCRIPTION.
    r : TYPE double
        DESCRIPTION.
    T : TYPE double
        DESCRIPTION.

    Returns
    -------
    replicate : TYPE double[]
        DESCRIPTION. BS vector

    """
    replicate = numpy.zeros(len(strike))
    for i in range(len(strike)):
        d1 = (math.log(spot / strike[i]) + (vol[i] ** 2 / 2 + r) * (T))
        d1 = d1 / (vol[i] * math.pow(T, 0.5))
        d2 = d1-vol[i] * math.pow(T, 0.5)
        BS = spot * scipy.stats.norm(0, 1).cdf(d1) 
        BS = BS - scipy.stats.norm(0, 1).cdf(d2) * strike[i] * math.exp(-r * T)
        replicate[i] = BS
    return replicate


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


def BSC(spot,strike,vol,r,T):
    replicate = numpy.zeros(len(strike))
    for i in range(len(strike)):
        d1 = (math.log(spot / strike) + (vol[i] * vol[i] / 2 + r) * (T)) / (vol[i] * math.pow(T,0.5))
        d2 = d1-vol[i] * math.pow(T,0.5)
        replicate[i] = spot * scipy.stats.norm(0,1).cdf(d1) - scipy.stats.norm(0,1).cdf(d2) * strike[i] * math.exp(-r * T)
    return replicate


def BS_vol_vect(tab,strike,vol,r,T):
    replicate = numpy.zeros(len(strike))
    for i in range(len(strike)):
        d1 = (math.log(tab / strike[i]) + (vol[i] ** 2 / 2 + r) * (T)) / (vol[i] * math.pow(T,0.5))
        d2 = d1-vol[i] * math.pow(T,0.5)
        replicate[i] = tab * scipy.stats.norm(0,1).cdf(d1) - scipy.stats.norm(0,1).cdf(d2) * strike[i] * math.exp(-r * T)
    return replicate
 
    
def interpolation_lagrange_absisseTchebychev(X,vol,strike,N):
    val = 0
    product = 1
    yi_interpolation = math.cos(math.pi * (2 * numpy.arange(0,2 * N) + 1) / (4 * N) )
    for i in range(0,N):
        for j in range(0,N):
            if j!= i :
                product = product * (X-strike[j]) / (strike[i]-strike[j])
        val = val + yi_interpolation[i] * product



def vector_vol_interpole(mat,strike,vol,step):
    vol = numpy.zeros(len(strike) * int(1 / step))
    strike_new = numpy.zeros(len(strike) * int(1 / step))
    for i in range(len(strike)-1):
        for j in range(int(1 / step)-1):
            strike_new[j] = strike[i] + j * step
            vol[i * int(1 / step) + j] = mat[0,i] * math.pow(strike_new[j],3) + mat[1,i] * math.pow(strike_new[j],2) + mat[2,i] * math.pow(strike_new[j],1) + mat[3,i]
    return vol



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



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 2 méthodes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def montecarlo1(strike,sigma,simul,steps,T,spot,r):
    S = numpy.zeros((steps, simul))
    S[0] = spot
    opt = 0
    for t in range(1, steps):
        w = numpy.random.standard_normal(size = simul)
        S[t] = S[t-1] * numpy.exp((r - 0.5 * sigma ** 2) * T + (sigma * numpy.sqrt(T) * w))
        opt = numpy.exp(-r * T) * 1 / simul * numpy.sum(numpy.maximum(S[-1] - strike, 0))
    return opt


def montecarlo(strike,sigma,simul,steps,T,spot,r):
    opt = 0
    spott = numpy.repeat(spot,(simul))
    S = numpy.repeat(spot,(simul))
    for t in range(1, steps):
        w = numpy.random.standard_normal(size = simul)
        S = spott * numpy.exp((r - 0.5 * sigma ** 2) * T + (sigma * numpy.sqrt(T) * w))
    opt = numpy.sum(numpy.maximum(S-strike, 0)) * numpy.exp(-r * T) * 1 / simul 
    return opt


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


#Monte Carlo Ornstein Uhlenbeck process
#Ornstein Uhlenbeck
def MontecarloStochastic_Ornstein_Uhlenbeck(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst) :
    """
    

    Parameters
    ----------
    returnmean : TYPE
        DESCRIPTION.
    vbar : TYPE
        DESCRIPTION.
    volvol : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
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
    None.

    """
    
    dt = T / n
    N = int(N)
    n = int(n)
    hurst = abs(hurst)
    S = numpy.repeat(float(spot),(n))
    var = numpy.repeat(float(v0),(n))
    price = 0
    for i in range(1,int(N)):
        W1 = numpy.random.rand(n)
        W2 = brownian_motion(n,n-1,hurst)
        for j in range(1,int(n)):
            var[j] = v0 * numpy.exp(returnmean * (vbar - var[j-1]) * dt * i + volvol * numpy.sqrt(numpy.absolute(var[j-1] * dt * i) ) * W2[j])
            S[j] = S[0] * numpy.exp((r-numpy.absolute(var[j] / 2)) * dt * i + numpy.sqrt((numpy.absolute(var[j] * dt * i))) * W1[j])
        if not math.isnan(numpy.maximum((S[len(S)-1]-strike),0) * numpy.exp(-r * T)) and not math.isinf(numpy.maximum((S[len(S)-1]-strike),0) * numpy.exp(-r * T)):
            price = price + (numpy.maximum((S[len(S)-1]-strike),0)) * numpy.exp(-r * T)
        else : 
            N=N-1
    return(price)  


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


#def nelder_mead(f, x_start,step = 0.1, no_improve_thr = 10e-6,no_improv_break = 10, max_iter = 0,alpha = 1., gamma = 2., rho = -0.5, sigma = 0.5):
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



def newton_raphston2(fct,x,N,i,X) : 
    m = numpy.zeros(len(x))
    m[0] = i#numpy.array([i,0,0,0,0,0])
    x[0] = x[0]-fct(x,X) * (i) / (fct(x,X)-fct(x-m,X))
    n = 1;
    while fct(x,X)!= 0 and abs(fct(x,X))>0.001 and  n<N and fct(x,X)!= "false":
        x[0] = x[0] -fct(x,X) * (i) / (fct(x,X)-fct(x-m,X))
        n = n + 1
    return x[0];



def fe(X):#returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst,market) :
    #X = [volinit,returnmean, vbar, volvol,rho, v0, r, mat[k], spot, strike[y], n, N,hurst]
    return abs(X[12] - MontecarloStochastic_Ornstein_Uhlenbeck(X[1], X[2], X[3], X[4], X[0], X[5], X[6],  X[7],  X[8],  X[9],  X[10], X[11]))


def calibrationVolNappe(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            X = [returnmean, vbar, volvol,rho, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
            x = [v0]
            ttest = newton_raphston2(f,x,30,0.01,X)
            newnappevol[y,k] = ttest
        print(y)
    return newnappevol


def calibrationVolNappeNelderMead(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)): 
            volinit=nappe[y,k]
            X = [returnmean, vbar, volvol,rho, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
            x = [volinit]
            #step,stop_stagne,stop, ittBreak,X_params
            ttest = neldermead(f,x,0.05,0.05,50, 50,X)
            newnappevol[y,k] = ttest[0][0]
        print(y)
    return newnappevol


def calibrationVolVolNappeNelderMead(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            X = [returnmean, vbar,rho,v0, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
            x = [volvol]
            #step,stop_stagne,stop, ittBreak,X_params
            ttest = neldermead(f,x,0.05,0.05,5, 20,X)
            newnappevol[y,k] = ttest[0][0]
        print(y)
    return newnappevol


def calibrationVolvolNappe(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            X = [returnmean, vbar,rho,v0, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
            x = [volvol]
            newnappevol=newton_raphston2(f,x,30,0.01,X)
    return newnappevol


def calibrationHurstnappeNelderMead(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            X = [returnmean, vbar,volvol,rho,v0, r, mat[k], spot, strike[y], n, N,nappe[y,k]]
            x = [hurst]
            #step,stop_stagne,stop, ittBreak,X_params
            ttest = neldermead(f,x,0.05,0.05,5, 20,X)
            newnappevol[y,k] = ttest[0][0]
        print(y)
    return newnappevol


def calibrationHurstNappe(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            X = [returnmean, vbar,volvol,rho,v0, r, mat[k], spot, strike[y], n, N,nappe[y,k]]
            x = [hurst]
            newnappevol[y,k] = newton_raphston2(f,x,30,0.01,X)
    return newnappevol


def calibration3paramsNappeNelderMead(f,nappe,strike,mat,returnmean, vbar, volvol, rho, v0, r, T, spot, n, N,hurst):
    newnappevol = numpy.zeros(shape = [len(strike),len(mat)])
    newnappevolvol = numpy.zeros(shape = [len(strike),len(mat)])
    newnappehurst = numpy.zeros(shape = [len(strike),len(mat)])
    for y in np.arange(0,len(strike)) :
        for k in np.arange(0,len(mat)):
            volinit = v0[y,k]
            X = [returnmean, vbar,rho, r, mat[k], spot, strike[y], n, N,nappe[y,k]]
            x = [volinit,volvol,hurst]
            ttest = neldermead(f,x,0.01,0.0001,10, 60,X)
            newnappevol[y,k] = ttest[0][0]
            newnappehurst[y,k] = ttest[0][2]
            newnappevolvol[y,k] = ttest[0][1]
        print(y)
    test = [newnappevol,newnappevolvol,newnappehurst]
    return test


def MontecarloStochastic_Ornstein_Uhlenbeck(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst) :
        dt = T / n
        N = int(N)
        n = int(n)
        S = numpy.repeat(spot,(N))
        var = numpy.repeat(v0**2,(N))
        for i in range(1,int(n)):
            W1 = numpy.random.rand(N)
            W2 = brownian_motion(n,N-1,hurst)
            var = var + returnmean * (vbar - var) * dt + volvol * W2
            S = S * numpy.exp((r-var / 2) * dt + numpy.sqrt(var * dt) * W1)
        price = max(numpy.exp(-r * T) * (spot- strike),0)
        return numpy.mean(price)    
    
    
def calibrate_Ornstein_Uhlenbeck(X): #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    return X[12] - MontecarloStochastic_Ornstein_Uhlenbeck(X[1], X[2], X[0], X[3], X[4], X[5], X[6],  X[7],  X[8],  X[9],  X[10], X[11])


def calibrate_six_params(X,Y):
    #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    #[v0,returnmean, vbar, volvol,rho,hurst, r,   Xtime[matu], spot, X[stri], n, N,marke_price]
    return abs(Y[6] - montecarlo_stochastic(X[1], X[2], X[0], X[3], X[4], Y[0], Y[1],  Y[2],  Y[3],  Y[4],  Y[5], X[5]))


def calibrate_vol(X,Y): 
    #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    #[returnmean, vbar, volvol,rho, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
    return abs(Y[11] - montecarlo_stochastic( Y[2], X[0], Y[4], Y[5],  Y[6],  Y[7],  Y[8],  Y[9], Y[10]))


# X = [returnmean, vbar,rho,volinit, r, mat[k], spot, strike[y], n, N,hurst,nappe[y,k]]
def calibrate_volvol(X,Y): #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    return abs(Y[11] - montecarlo_stochastic( X[0], Y[3], Y[4], Y[5],  Y[6],  Y[7],  Y[8],  Y[9], Y[10]))


# X = [returnmean, vbar,volvol,rho,volinit, r, mat[k], spot, strike[y], n, N,nappe[y,k]]
def calibrate_hurst(X,Y): 
    # montecarlo_stochastic(volvol, v0, r, T, spot, strike, n, N,hurst)
    return abs(Y[11] - montecarlo_stochastic(Y[2], Y[4], Y[5],  Y[6], Y[7], Y[8], Y[9], Y[10], X[0]))


def calibration_monte_3params(X,Y): #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    monte = montecarlo_stochastic( X[1], X[0], Y[3],  Y[4],  Y[5],  Y[6],  Y[7],Y[8], X[2])
    # print(monte)
    return abs(Y[9] - monte)


def functUB(X,Y): #(returnmean, vbar, volvol, rho, v0, r, T, spot, strike, n, N,hurst)
    #[v0,returnmean, vbar, volvol,rho,hurst, r,   Xtime[matu], spot, X[stri], n, N,marke_price]
    return abs(Y[6] - MontecarloStochastic_Ornstein_Uhlenbeck(X[1], X[2], X[0], X[3], X[4], Y[0], Y[1],  Y[2],  Y[3],  Y[4],  Y[5], X[5]))




strike = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
price = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]
r = 0
T = 1
spot = 100
step = 0.01
vol = vol_imply_vect(price, spot, r, T, strike)
X = [strike[0] + i * step for i in range((len(strike)-1) * int(1 / step))]

# TEST 4 interpolation without lagrange and cvxt
# interpolation sans lagrange cvxt
poly = 4
suite = [0, 2, 3, 4, 5, 6]
poly = cvxt_interpolation(strike, vol, suite)
VOL = polynome_order_three(numpy.array(poly), suite, X, step, 9)
plt.plot(strike, vol, 'o', X, VOL, '-')
plt.title(" derivatives with cvxt")
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 2
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

price9m = [11.79,8.95,8.07,7.03,6.18,6.04,5.76,5.50,5.50,5.39]
price6m = [10.71,8.28,6.91,6.36,5.29,5.07,4.76,4.47,4.35,4.14]
price3m = [8.67,7.14,5.98,4.93,4.09,3.99,3.43,3.01,2.72,2.53]
strike = [95,96,97,98,99,100,101,102,103,104]
price12m = [12.40,9.59,8.28,7.40,6.86,6.58,6.52,6.49,6.47,6.46]

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

maturite = 7 / 12 * 100
striketest = 98.3
print("Index strike ",(striketest-95) / step)
print("Vol :",nappe[33,int(maturite)])
print("Price", BlackScholes(nappe[33,int(maturite)],striketest,100,r,maturite / 100))
print("Price", BlackScholes(0.1352,98.3,100,r,7 / 12))

Xmat = numpy.array([X,] * 100) 
Ymat = numpy.array([Xtime,] * 90).transpose()
Zmat = nappe
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_zlim(0, 0.3)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.title(" Vol surface ")

ax.set_xlabel('Strike')
ax.set_ylabel('Time until mat in year')
ax.set_zlabel('vol')

fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()


nappeprice = numpy.zeros(shape = (90,100))
for i in range(len(VOL12m)):
    for j in range(len(Xtime)):
        nappeprice[i,j] = BlackScholes(nappe[i,j],X[i],spot,r,Xtime[j])
        
print("Price extracted from surface for 7 month option strike at 98.3", nappeprice[33,58])


Xmat = numpy.array([X,] * 100) 
Ymat = numpy.array([Xtime,] * 90).transpose()
Zmat = nappeprice
fig2 = plt.figure()
ax2 = fig2.gca(projection = '3d')
surf = ax2.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_zlim(0, 0.3)
plt.title(" Price surface ")
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.set_xlabel('Strike')
ax2.set_ylabel('Time until mat in year')
ax2.set_zlabel('Price')
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig2.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()


#Calibration surface
rho = 1
v0 = 0.13
hurst = 0.25
returnmean = 1
volvol = 0.2
vbar = 1
maturite = 7 / 12
striketest = 98.3

stri = round((striketest-95) * 10)
matu = round(maturite * 100) 
marke_price = nappeprice[stri,matu]        
print("BS results price :",marke_price)
n = 30
N = 1000


#test Montecarlo with fixed parameters and volvol = 0.1 and vol from the implied vol
tic = ti.perf_counter()
esa = montecarlo_stochastic(0.1,0.1352, r, 7 / 12, spot, 98.3, n, N,0.1) 
print("Monte Carlo test with fixed parameters ",esa)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#test nelder mead calibrationd du strike 98.3 et 7 mois
n = 30
N = 5000
Xparams = [0, 0,0, r, maturite , spot, striketest, n, N,marke_price]
x = [v0,volvol,hurst]
parameters = neldermead(calibration_monte_3params, x,0.1,0.01,40, 70,Xparams)
print("Parameters calibrated Heston [vol,volvol,hurst] ",parameters)
#Parameters calibrated Heston [vol,volvol,hurst]  [array([0.13107441, 0.22117992, 0.27079042]), 0.0065950573488686715]

Xparams = [0, 0,0, r, maturite , spot, striketest, n, N,marke_price]
x = [v0,volvol,hurst]
parameters2_ = neldermead(calibrate_Ornstein_Uhlenbeck, x,0.1,0.01,40, 70,Xparams)




n = 30
N = 5000
price_calibrated = montecarlo_stochastic(parameters[0][1], parameters[0][0], r, Xtime[matu], spot, X[stri], n, N,parameters[0][2]) 
print(price_calibrated) 
#4.995575061993855

price_calibrated = montecarlo_stochastic(0.19873639,0.14231562, r, Xtime[matu], spot, X[stri], n, N,0.25129164) 
print(price_calibrated) 



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Calibration of 3 surfaces for each parameters vol, vol of vol and hurst exponent
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n = 30
N = 3000
v0 = 0.1352
#créer les 3 nappes

#Neldermead c'est 1000 Fois mieux que newton raphston.
voll = [vol3m,vol6m,vol9m,vol12m]    
tic = ti.perf_counter()
nappevol_nelder = calibrationVolNappeNelderMead(calibrate_vol,nappe,strike,time,0, 0, volvol, 0, voll, r, T, spot, n, N,hurst)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#5851sec
tic = ti.perf_counter()
nappevolvol = calibrationVolVolNappeNelderMead(calibrate_volvol,nappeprice,strike,time,0, 0, volvol, 0, v0, r, T, spot, n, N,hurst)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#5323
tic = ti.perf_counter()
nappehurst = calibrationHurstnappeNelderMead(calibrate_hurst,nappeprice,strike,time,0, 0, volvol, 0, v0, r, T, spot, n, N,hurst)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#print 3 nappes: 
Zmat = np.array(nappevol_nelder)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([strike,] * 4) 
Ymat = numpy.array([time,] * 10).transpose()
fig3 = plt.figure()
ax3 = fig3.gca(projection = '3d')
surf3 = ax3.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax3.set_zlim(0,0.9)
ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig3.colorbar(surf3, shrink = 1, aspect = 1)
ax3.set_xlabel('Strike')
ax3.set_ylabel('Time until mat in year')
ax3.set_zlabel('Nappe Vol')
plt.show()

Zmat = np.array(nappevolvol)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([strike,] * 4) 
Ymat = numpy.array([time,] * 10).transpose()
fig4 = plt.figure()
ax4 = fig4.gca(projection = '3d')
surf4 = ax4.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax4.set_zlim(0.15,0.25)
ax4.zaxis.set_major_locator(LinearLocator(10))
ax4.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig4.colorbar(surf4, shrink = 1, aspect = 1)
ax4.set_xlabel('Strike')
ax4.set_ylabel('Time until mat in year')
ax4.set_zlabel('Nappe Vol vol')
plt.show()

Zmat = np.array(nappehurst)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([strike,] * 4) 
Ymat = numpy.array([time,] * 10).transpose()
fig5 = plt.figure()
ax5 = fig5.gca(projection = '3d')
surf5 = ax5.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax5.set_zlim(0.25,0.31)
ax5.zaxis.set_major_locator(LinearLocator(10))
ax5.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig5.colorbar(surf5, shrink = 1, aspect = 1)
ax5.set_xlabel('Strike')
ax5.set_ylabel('Time until mat in year')
ax5.set_zlabel('Nappe Vol vol')
plt.show()


hurst = nappehurst[33,int(maturite * 100)]
volvol = nappevolvol[33,int(maturite * 100)]
V0 = nappevol_nelder[33,int(maturite * 100)]

print("Hurst nappe :",hurst)
print("Volvol nappe :",volvol)
print("Vol nappe :",vol)




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Calibration one surface with 3 parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
N = 3000
tic = ti.perf_counter()
params3 = calibration3paramsNappeNelderMead(calibration_monte_3params,nappeprice,X,Xtime,0, 0, volvol, 0, voll, r, T, spot, n, N,hurst)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

nappeHu = params3[2]
nappevo = params3[0]
nappevovo = params3[1]

Zmat = np.array(nappevo)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([np.arange(95,104,0.5),] * 10) 
Ymat = numpy.array([np.arange(0,1,0.1),] * 18).transpose()
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_zlim(0,0.2)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink = 1, aspect = 1)
plt.show()

Zmat = np.array(nappevovo)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([np.arange(95,104,0.5),] * 10) 
Ymat = numpy.array([np.arange(0,1,0.1),] * 18).transpose()
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_zlim(0.23,0.26)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink = 1, aspect = 1)
plt.show()

Zmat = np.array(nappeHu)
Zmat[numpy.isneginf(Zmat)] = 0
Xmat = numpy.array([np.arange(95,104,0.5),] * 10) 
Ymat = numpy.array([np.arange(0,1,0.1),] * 18).transpose()
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(Xmat, Ymat, Zmat.transpose(), cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_zlim(0.25,0.31)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink = 1, aspect = 1)
plt.show()




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 5 - We calibrate with precision for one specifique point of the surface (strike = 98.3 and mat = 7 months)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n = 30
N = 2000
striketest = 98.3
matu = 7 / 12
posi = 33
posic = 58
#Ici on recalibre de manière indépendante pour avoir une calibration plus précise
hurst = 0.25
volinit = nappe[posi,posic]
XP = [0, 0, volvol,0, r, matu, spot,striketest, n, N,hurst,nappeprice[posi,posic]]
x = [volinit]
ttest = neldermead(calibrate_vol,x,0.01,0.001,50, 100,XP)
v0 = ttest[0][0]
print("Vol 1 params", v0)

volinit = v0
XP = [0, 0,0,volinit, r, matu, spot,striketest, n, N,hurst,nappeprice[posi,posic]]
x = [volvol]
ttest = neldermead(calibrate_volvol,x,0.01,0.001,50,100,XP)
volvol = ttest[0][0]
print("Volvol 1 params", volvol)

XP = [0, 0,volvol,0,volinit, r, matu, spot, striketest, n, N,nappeprice[posi,posic]]
x = [hurst]
ttest = neldermead(calibrate_hurst,x,0.01,0.001,50, 100,XP)
hurst = ttest[0][0]
print("hurst 1 params", hurst)

N = 2000
tic = ti.perf_counter()
esa = montecarlo_stochastic(0, 0, volvol, 0,v0, r, 7 / 12, spot, 98.3, n, N,hurst) 
print("Monte Carlo 1 params",esa)
toc = ti.perf_counter()
print("Time :",str(toc-tic))

#BS
essay = nappe[33,58]
print("BS ",BlackScholes(essay,striketest,spot,r,matu))
print("vol ",essay)


#Avec les 3 parametres
volinit = v0
XP = [0, 0,0, r,matu, spot, striketest, n, N,nappeprice[posi,posic]]
x = [volinit,volvol,hurst]
ttest = neldermead(calibration_monte_3params,x,0.01,0.0001,50, 100,XP)
vol2 = ttest[0][0]
volvol2 = ttest[0][2]
hurst2 = ttest[0][1]

print("3 params", ttest[0])
tic = ti.perf_counter()
esa = montecarlo_stochastic(0, 0, volvol2, 0,vol2, r, 7 / 12, spot, 98.3, n, N,hurst2) 
print("Monte Carlo 3 params ",esa)
toc = ti.perf_counter()
print("Time :",str(toc-tic))



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Calculation with the Ornstein_Uhlenbeck Process 5 parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Avec 5 parametres
returnmean = 1
vbar = 0.1
rho = 0.3
params = [vol2,returnmean, vbar, volvol2,rho,hurst2]
X_params = [ r,matu, spot,striketest, n, N,nappeprice[posi,posic]]
v022 = neldermead(calibrate_six_params, params,step,0.0001,10, 20,X_params)
print(v022)
print(step)
#[array([0.24851852, 0.98518519, 0.91111111, 0.20185185, 0.96296296,0.36851852]), 0.26000935000210745]

tic = ti.perf_counter()
esa = montecarlo_stochastic(v022[0][1], v022[0][2], v022[0][3], v022[0][4],v022[0][0], r,matu, spot, striketest, n, N,v022[0][5]) 
print("Monte Carlo 5 params ",esa)
toc = ti.perf_counter()
print("Time :",str(toc-tic))


       