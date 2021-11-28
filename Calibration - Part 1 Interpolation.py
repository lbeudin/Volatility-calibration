#  - * - coding: utf-8 - * -
"""
Created on Wed Dec  2 11:14:28 2020

@author: Léa
"""
import numpy as numpy
import math
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import matrix 
from scipy.ndimage.interpolation import shift
from scipy import interpolate

import numpy as np
import random

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
    d1 = (math.log(S / K) + (r + math.pow(sigma, 2) / 2)) * (T) / (sigma * math.sqrt(T))
    d2 = d1-sigma * math.pow(T, 0.5)
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



def Spline(K, strike, vol):
    """
    Interpolate volatility vector and return the parameters of the curve.

    Parameters
    ----------
    K : TYPE int
        DESCRIPTION. number of polynome
    strike : TYPE double []
        DESCRIPTION.
    vol : TYPE double[]
        DESCRIPTION.

    Returns
    -------
    params1 : TYPE double
        DESCRIPTION.
    params2 : TYPE double
        DESCRIPTION.
    params3 : TYPE double
        DESCRIPTION.
    params4 : TYPE double
        DESCRIPTION.

    """
    d1 = [(vol[i + 1]-vol[i]) / (strike[i + 1]-strike[i]) for i in range(K)]
    d2 = [(vol[1]-vol[0])] + [(d1[i + 1]-d1[i]) / (strike[i + 1]-strike[i])
                              for i in range(K-1)]
    d2 = d2 + [(vol[len(vol)-1]-vol[len(vol)-2])]
    params1 = [1 / 6 * (d2[i + 1]-d2[i]) / (strike[i + 1]-strike[i])
               for i in range(K)]
    params2 = [1 / 2 * (d2[i]) for i in range(K)]
    params3 = [d1[i]-(strike[i + 1]-strike[i]) * (d2[i + 1] + 2 * d2[i]) / 6
               for i in range(K)]
    params4 = [vol[i] for i in range(K)]
    return params1, params2, params3, params4


def polynome_degree_three(params, strike, vol, step):
    """
    Create a polynonme degree 3 with x=strike y=vol.

    Parameters
    ----------
    params : TYPE double[,]
        DESCRIPTION.
    strike : TYPE double[]
        DESCRIPTION.
    vol : TYPE double[]
        DESCRIPTION.
    step : TYPE double
        DESCRIPTION.

    Returns
    -------
    vol : TYPE double[]
        DESCRIPTION.a new volatility based
        on a polynome relationship with strikes.

    """
    occurence = int(1 / step)
    vol = numpy.zeros(len(strike) * occurence)
    strike_new = numpy.zeros(len(strike) * occurence)
    for i in range(len(strike)-1):
        for j in range(occurence-1):
            strike_new[j] = strike[i] + j * step
            vol[i * occurence + j] = params[0, i] * math.pow(strike_new[j], 3)
            + params[1, i] * math.pow(strike_new[j], 2)
            + params[2, i] * math.pow(strike_new[j], 1)
            + params[3, i]
    return vol


def interpolation_lagrange(vol, strike):
    """
    Interpolate the 3 first point with Lagrange method.

    Parameters
    ----------
    vol : TYPE double[]
        DESCRIPTION.
    strike : TYPE double[]
        DESCRIPTION.

    Returns
    -------
    list double
        DESCRIPTION.three parameters of a polynome order 2

    """
    denom = 1
    b = 0
    a = 0
    c = 1
    B = 0
    C = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if j != i:
                denom = denom * (strike[i]-strike[j])
                b = b-strike[j]
                c = -c * strike[j] / (strike[i]-strike[j])
        a = vol[i] / denom + a
        C = c * vol[i] + C
        B = vol[i] * b / denom + B
        denom = 1
        c = 1
        b = 0
    return ([a, B, C])



def derivation_interpolation(params, strike, vol, suite):
    """
    Interpolate the rest of the curve.

    Parameters
    ----------
    params : TYPE double[]
        DESCRIPTION. params of the interpolation of the first point
    strike : TYPE
        DESCRIPTION.
    vol : TYPE
        DESCRIPTION.
    suite : TYPE int[]
        DESCRIPTION. index of the remaining points to interpolate
        should stops 3 points before end

    Returns
    -------
    list_params : TYPE double[][]
        DESCRIPTION.matrix of the parameters of iterpolation

    """
    list_params = []
    list_params.append(params)
    for i in suite:
        mat = matrix([
            [strike[i + 2] ** 2, strike[i + 2], 1],
            [strike[i] ** 2, strike[i], 1],
            [strike[i] * 2, 1, 0]])
        to = (vol[i + 1]-vol[i]) / (strike[i + 1]-strike[i])
        y = matrix([[vol[i + 2]], [vol[i]], [to]])
        res = numpy.ndarray.tolist((mat.I * y))
        listabc = [item for sublist in res for item in sublist]
        list_params.append(listabc)
    return list_params


def derivation_cvxt_interpolation(params, strike, vol, suite):
    """
    Interpolate the rest of the curve considering convexity.

    Parameters
    ----------
    params : TYPE double[]
        DESCRIPTION. params of the interpolation of the first point
    strike : TYPE
        DESCRIPTION.
    vol : TYPE
        DESCRIPTION.
    suite : TYPE int[]
        DESCRIPTION. index of the remaining points to interpolate
        should stops 3 points before end

    Returns
    -------
    list_params : TYPE double[][]
        DESCRIPTION.matrix of the parameters of iterpolation

    """
    list_params = []
    list_params.append(params)
    for i in suite:
        mat = matrix([
            [strike[i + 2] ** 2, strike[i + 2], 1],
            [strike[i] ** 2, strike[i], 1],
            [2, 0, 0]])

        cvxt = ((vol[i + 2]-vol[i + 1]) / (strike[i + 2] - strike[i + 1])
                - (vol[i+1] - vol[i]) / (strike[i+1] - strike[i]))
        cvxt = cvxt / (strike[i + 1]-strike[i])
        y = matrix([[vol[i + 2]], [vol[i]], [cvxt]])
        res = numpy.ndarray.tolist((mat.I * y))
        listabc = [item for sublist in res for item in sublist]
        list_params.append(listabc)
    return list_params


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


def polynome_multiple_method(params, suite, X, step, index_last_point):
    """
    Calculate the polynome degree 2 with 2 different interpol. method used.

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
    lastsuite = 0
    p = 0
    occurence = int(1 / step)
    Y = numpy.zeros(len(X))
    list_index = suite.copy()
    list_index.append(index_last_point)
    for i in (list_index):
        for k in range(lastsuite, i):
            for j in range(int(1 / step)):
                index = k * occurence + j
                Y[index] = params[p, 0] * X[k * occurence + j] ** 2
                Y[index] = Y[index] + params[p, 1] * X[k * occurence + j]
                Y[index] = Y[index] + params[p, 2]
        p = p + 1
        lastsuite = i
    return Y


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
                variable = params[p, 0] * X[k * int((1 / step)) + j] ** 3
                variable = variable + params[p, 1] * X[k * occurence + j] ** 2
                variable = variable + params[p, 2] * X[k * occurence + j]
                variable = variable + params[p, 3]
                Y[k * int(1 / step) + j] = variable
        p = p + 1
        lastsuite = i
    return Y


def density_neutral_risk(spot, vol, strike, r, T, step):
    """
    Calculate neutral risk density.

    Parameters
    ----------
    price : TYPE double[]
        DESCRIPTION.
    strike : TYPE double[]
        DESCRIPTION.
    r : TYPE double
        DESCRIPTION.
    T : TYPE double
        DESCRIPTION.
    k : TYPE int
        DESCRIPTION. step for the derivative

    Returns
    -------
    densi : TYPE double
        DESCRIPTION. neutral risk density

    """
    densi = numpy.zeros(len(strike)-2*step)
    for i in range(step, len(strike)-step):
        denom = (strike[i+step]-strike[i])
        pricei = BlackScholes(vol[i], strike[i], spot, r, T)
        D1 = (BlackScholes(vol[i], strike[i+step], spot, r, T)-pricei) / denom
        D2 = (pricei-BlackScholes(vol[i], strike[i-step], spot, r, T)) / denom
        densi[i-step] = np.exp(r * T) * (D1-D2) / denom
    return densi



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                         Part 1
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
strike = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
price = [12.40, 9.59, 8.28, 7.40, 6.86, 6.58, 6.52, 6.49, 6.47, 6.46]
r = 0
T = 1
spot = 100
step = 0.01

X = [strike[0] + i * step for i in range((len(strike)-1) * int(1 / step))]

vol = vol_imply_vect(price, spot, r, T, strike)
print(vol)
# We want to obtain a continuous function of the volatilty
# in function of the strikes

# the objectif to obtain :
# Interpolation with a package
ft = interpolate.interp1d(strike, vol, fill_value="extrapolate")
Y = ft(X)
plt.plot(strike, vol, 'o', X, Y, '-')
plt.title("Targeted Interpolation")
plt.show()


# TEST 1 : interpolation with spline
parameter_interpolation = numpy.asarray(Spline(9, strike, vol))
vol_new = polynome_degree_three(parameter_interpolation, strike, vol, step)
plt.title("Spline method")
plt.plot(vol_new)
# we obtain a square signal


# TEST 2 : interpolation Lagrange + derivative interpolation
poly = 4
# index of remaining point to interpolate
suite = [2, 3, 4, 5, 6]
# interpolation of the first 2 points
params = interpolation_lagrange(vol, strike)
poly = derivation_interpolation(params, strike, vol, suite)
VOL = polynome_multiple_method(numpy.array(poly), suite, X, step, 9)

plt.plot(strike, vol, 'o', X, VOL, '-')
plt.title("Lagrange Interpolation + derivatives")
plt.show()
# The interpolation seems better but not as perfect and not continuous yet


# TEST 3 : improvement with convexity
# interpolation avec cvxt
abc = interpolation_lagrange(vol, strike)
poly = derivation_cvxt_interpolation(abc, strike, vol, suite)
VOL = polynome_multiple_method(numpy.array(poly), suite, X, step, 9)
plt.plot(strike, vol, 'o', X, VOL, '-')
plt.title("Lagrange Interpolation + derivatives with cvxt")
plt.show()


# TEST 4 interpolation without lagrange and cvxt
# interpolation sans lagrange cvxt
poly = 4
suite = [0, 2, 3, 4, 5, 6]
poly = cvxt_interpolation(strike, vol, suite)
VOL = polynome_order_three(numpy.array(poly), suite, X, step, 9)
plt.plot(strike, vol, 'o', X, VOL, '-')
plt.title(" derivatives with cvxt")
plt.show()
# We keep this interpolation methd as the curve obtained is continuous.



# Now that we have the interpolation of the volatility
# we calculate the price of the call
continuous_prices = BlackScholes_continous_vol(spot, X, VOL, r, T)
plt.plot(strike, price, 'o', X, continuous_prices, '-')
plt.title("Continious curve Option prices with BS")
plt.show()


# Measure Breeden-Litzenberger
# with the fly approximation
short = -2 * BlackScholes_continous_vol(spot, X, VOL, r, T)
long1 = BlackScholes_continous_vol(spot, shift(X, 1, cval=np.NaN), VOL, r, T)
long2 = BlackScholes_continous_vol(spot, shift(X, -1, cval=np.NaN), VOL, r, T)
fly_densite = np.exp(r * T)*(long1 + long2 + short) / (0.01 ** 2) 
plt.plot(X, fly_densite, '-')
plt.title("Density measure with BS")
plt.show()


# comparison with a gaussian curve.
pts = numpy.zeros(900)
k = 0
for i in numpy.arange(95, 104, 0.01):
    pts[k] = scipy.stats.norm(100, 1).pdf(i)
    k = k + 1
plt.plot(X, pts, '-')
plt.title("Gaussian curve comparison")
plt.show()

#we obtain a dirac combination
densi = density_neutral_risk(spot, VOL, X, r, T, 1)
tt = [strike[1] + i * step for i in range(1,(len(strike)-1) * int(1 / step)-1)]
plt.plot(tt, densi, '-')
plt.title("Density with Breeden-Litzenberger step 1 ")
plt.show()

# on a une interpolation presque parfaite
densi = density_neutral_risk(spot, VOL, X, r, T, 10)
tt = [strike[1] + i * step for i in range(10,(len(strike)-1) * int(1 / step)-10)]
plt.plot(tt, densi, '-')
plt.title("Density with Breeden-Litzenberger step 10 ")
plt.show()


