import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, logistic, gamma

# Helper functions
def cumulative_gamma(x, shape, scale):
    return gamma.cdf(x, a=shape, scale=scale)

def logit(x):
    return np.log(x / (1 - x))

# Function definitions
def generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const):
    # Generate daily incident cases and covariates data
    Z = np.zeros((T, NoCov))
    for t in range(T):
        Z[t, 0] = 5 - (T / 8) + (t / 4) + norm.rvs(loc=0, scale=3)
    Z[:, 1] = logit(np.random.uniform(0.01, 0.99, T)) + 2

    # R[t], the instantaneous reproduction number at time t
    R = np.zeros(T)
    epsilon = norm.rvs(size=T, scale=np.sqrt(-2 * np.log(bias_corr_const)))
    R[0] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(R_0) + Z[0, :] @ OracleBeta + epsilon[0])

    for t in range(1, T):
        R[t] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(R[t-1]) + Z[t, :] @ OracleBeta + epsilon[t])

    # Generate I[t], number of incidences at time t
    I = np.zeros(T)
    I[0] = poisson.rvs(R[0] * I_0 * Omega[0])

    for t in range(1, 25):
        I[t] = poisson.rvs(R[t] * (I_0 * Omega[t] + I[:t] @ Omega[t-1::-1]))

    for t in range(25, T):
        lambda_val = R[t] * (I[t-25:t] @ Omega[::-1])
        if lambda_val <= 100000:
            I[t] = poisson.rvs(lambda_val)
        else:
            multipconstant = int(lambda_val // 100000)
            residueconstant = lambda_val % 100000
            I[t] = poisson.rvs(100000, size=multipconstant).sum() + poisson.rvs(residueconstant)

    # Save Z, I, R as a DataFrame
    df = pd.DataFrame({'Z1': Z[:, 0], 'Z2': Z[:, 1], 'R': R, 'I': I})
    return df

def generate_data_more_var(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta1, bias_corr_const):
    # Initialize matrices
    Ztrial1 = np.empty((T, NoCov + 1))
    Rtrial1 = np.empty(T)
    Itrial1 = np.empty(T)

    # First Column
    for t in range(T):
        Ztrial1[t, 0] = 10 - (T / 8) + (t / 4) + norm.rvs(loc=0, scale=3)

    # Second Column
    Ztrial1[:, 1] = logit(np.random.uniform(0.01, 0.99, T)) + 2

    # Third Column
    for t in range(T):
        Ztrial1[t, 2] = -(T / 18) + (t / 9) + norm.rvs(loc=0, scale=5)

    # epsilon as defined before
    epsilon = norm.rvs(size=T, scale=np.sqrt(-2 * np.log(bias_corr_const)))

    # Compute R
    Rtrial1[0] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(R_0) + np.dot(Ztrial1[0, :], OracleBeta1) + epsilon[0])
    for t in range(1, T):
        Rtrial1[t] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(Rtrial1[t - 1]) + np.dot(Ztrial1[t, :], OracleBeta1) + epsilon[t])

    # Compute I
    Itrial1[0] = poisson.rvs(Rtrial1[0] * I_0 * Omega[0])
    for t in range(1, 25):
        lambda_val = Rtrial1[t] * (I_0 * Omega[t] + np.dot(Itrial1[:t], Omega[t - 1::-1]))
        Itrial1[t] = poisson.rvs(lambda_val)

    for t in range(25, T):
        lambda_val = Rtrial1[t] * np.dot(Itrial1[t - 25:t], Omega[:25])
        if lambda_val <= 100000:
            Itrial1[t] = poisson.rvs(lambda_val)
        else:
            multipconstant = int(lambda_val // 100000)
            residueconstant = lambda_val % 100000
            Itrial1[t] = poisson.rvs(100000, size=multipconstant).sum() + poisson.rvs(residueconstant)

    # Save Z, I, R as a DataFrame
    df1 = pd.DataFrame({'Z1': Ztrial1[:, 0], 'Z2': Ztrial1[:, 1], 'Z3': Ztrial1[:, 2], 'R': Rtrial1, 'I': Itrial1})
    return df1


def generate_data_ar2(T, NoCov, R_0, I_0, Omega, OraclePhi1, OracleBeta, bias_corr_const, Rmin1):
    # Initialize matrices
    Ztrial2 = np.empty((T, NoCov))
    Rtrial2 = np.empty(T)
    Itrial2 = np.empty(T)

    # First Column
    for t in range(T):
        Ztrial2[t, 0] = 7.5 - (T / 8) + (t / 4) + norm.rvs(loc=0, scale=3)

    # Second Column
    Ztrial2[:, 1] = logit(np.random.uniform(0.01, 0.99, T)) + 2

    # Compute R
    epsilon = norm.rvs(size=T, scale=np.sqrt(-2 * np.log(bias_corr_const)))
    Rtrial2[0] = np.exp(OraclePhi1[0] + OraclePhi1[1] * np.log(R_0) + OraclePhi1[2] * np.log(Rmin1) + np.dot(Ztrial2[0, :], OracleBeta) + epsilon[0])
    Rtrial2[1] = np.exp(OraclePhi1[0] + OraclePhi1[1] * np.log(Rtrial2[0]) + OraclePhi1[2] * np.log(R_0) + np.dot(Ztrial2[1, :], OracleBeta) + epsilon[1])

    for t in range(2, T):
        Rtrial2[t] = np.exp(OraclePhi1[0] + OraclePhi1[1] * np.log(Rtrial2[t - 1]) + OraclePhi1[2] * np.log(Rtrial2[t - 2]) + np.dot(Ztrial2[t, :], OracleBeta) + epsilon[t])

    # Compute I
    Itrial2[0] = poisson.rvs(Rtrial2[0] * I_0 * Omega[0])
    for t in range(1, 25):
        lambda_val = Rtrial2[t] * (I_0 * Omega[t] + np.dot(Itrial2[:t], Omega[t - 1::-1]))
        Itrial2[t] = poisson.rvs(lambda_val)
    
    for t in range(25, T):
        lambda_val = Rtrial2[t] * np.dot(Itrial2[t - 25:t], Omega[:25])
        if lambda_val <= 100000:
            Itrial2[t] = poisson.rvs(lambda_val)
        else:
            multipconstant = int(lambda_val // 100000)
            residueconstant = lambda_val % 100000
            Itrial2[t] = poisson.rvs(100000, size=multipconstant).sum() + poisson.rvs(residueconstant)

    # Save Z, I, R as a DataFrame
    df2 = pd.DataFrame({'Z1': Ztrial2[:, 0], 'Z2': Ztrial2[:, 1], 'R': Rtrial2, 'I': Itrial2})
    return df2

