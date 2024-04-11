import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
from scipy.stats import gamma
from scipy.optimize import minimize
import warnings
from create_data import generate_data_true, generate_data_more_var, generate_data_ar2,cumulative_gamma
import os


#set seed for reproducibility
np.random.seed(214)
# Define the common parameters
T = 120
NoCov = 2
R_0 = 3
I_0 = 500
bias_corr_const = np.exp(-0.001 / 2)
Omega = np.array([cumulative_gamma(i, 2.5, 3) - cumulative_gamma(i-1, 2.5, 3)
                  for i in range(1, 26)]) / cumulative_gamma(25, 2.5, 3)
OraclePhi = np.array([0.5, 0.7])
OracleBeta = np.array([-0.02, -0.125]).reshape(NoCov, 1)
OracleBeta1 = np.array([-0.02, -0.125, -0.03]).reshape(NoCov + 1, 1)
OraclePhi1 = np.array([0.5, 0.5, 0.3])
Rmin1 = 2


# Generate data
df0 = generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const)
df = generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const)
df1 = generate_data_more_var(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta1, bias_corr_const)
df2 = generate_data_ar2(T, NoCov, R_0, I_0, Omega, OraclePhi1, OracleBeta, bias_corr_const, Rmin1)
#define tau
tau_0=5
#write out df to .csv file for inspection
df.to_csv("df.csv", index=False)

# Helper Functions
def logistic(x):
    return np.log(x / (1 - x))

def greater_or_na(obj, const, tocat):
    if np.isnan(obj):
        print(f"{obj} {tocat}")
    return np.isnan(obj) or obj > const

def omega_vector(length=25, shape=2.5, scale=3):
    omega = np.array([(gamma.cdf(i, shape, scale=scale) - gamma.cdf(i - 1, shape, scale=scale))
                      / gamma.cdf(length, shape, scale=scale) for i in range(1, length + 1)])
    return omega
Z = df[['Z1', 'Z2']].values
I = df['I'].values
 #print the min and max of I
print(np.min(I),np.max(I))
#def QSOEID(Z, I, NoCov, T, I_0, R_0, bias_corr_const,tau_0):
Omega = omega_vector()
Lambda = np.full(T, np.nan)
Lambda[0] = I_0 * Omega[0]
#range 1-24
for t in range(1, 25):
    Lambda[t] = I_0 * Omega[t] + np.dot(I[:t], Omega[t-1::-1])
#range 25-T-1
for t in range(25, T):
    Lambda[t] = np.dot(I[t-25:t], Omega[::-1])

EstR = np.full(T, np.nan)
EstR[0] = max(1, I[0] / (I_0 * Omega[0]))
for t in range(1, tau_0):
    EstR[t] = max(1, I[t] / (I_0 * Omega[t] + np.dot(I[:t], Omega[t-1::-1])))
print(EstR[:t])

# Calculating barZ, WTilde, YTilde, BetaTilde, ZYTilde, EstPhi, and EstBeta
barZ=Z
for t in range (1,T):
    barZ = np.cumsum(Z, axis=0) / np.arange(1, T + 1)[:, None]
# WTilde calculation
WTilde = [np.zeros((NoCov, NoCov))]  # WTilde as a list of matrices
for t in range(1, tau_0):
    WTilde.append(np.zeros((NoCov, NoCov)))
    for i in range(t):
        WTilde[t] += np.outer(Z[i,] - barZ[t,], Z[i,] - barZ[t,])
    WTilde[t] = np.linalg.inv(WTilde[t] + 0.1 * np.eye(NoCov))

for t in range(2, T):
    WTilde.append(np.zeros((NoCov, NoCov)))
    for i in range(t):
        WTilde[t] += np.outer(Z[i,] - barZ[t,], Z[i,] - barZ[t,])
    WTilde[t] = np.linalg.inv(WTilde[t])

YTilde = np.full((1, T), np.nan)
for t in range(tau_0):
    YTilde[0, t] = np.log(EstR[t])

BetaTilde = np.full((T, NoCov), np.nan)
ZYTilde = np.full((T, NoCov), np.nan)
EstPhi = np.full((T, 2), np.nan)
EstBeta = np.full((T, NoCov), np.nan)
### Intermediate variables, ZYHat
ZYHat=ZYTilde

# Define the profile likelihood function
def ell( phi, k ,bias_corr_const):
    global    R_0, tau_0

    # First part of the calculation
    ZYTilde[(k - 1), ] = (np.log(EstR[ 0]) - phi[1] * np.log(R_0) - phi[0]) * (Z[0, :] - barZ[(k - 1), :])
    EstR[0] = np.exp(phi[0] + phi[1] * np.log(R_0))
    print(ZYTilde[k - 1, :])
    #EstR[0] = np.exp(phi[0] + phi[1] * np.log(R_0))

    # Loop for updating ZYTilde
    for i in range(1, k-1):
        ZYTilde[(k - 1), ] += (np.log(EstR[i]) - phi[1] * np.log(EstR[ i - 1]) - phi[0]) * (Z[i, :] - barZ[(k - 1), :])

        print(ZYTilde[k - 1, :])

    # Updating BetaTilde
    BetaTilde[k-1 , :] = np.dot(ZYTilde[(k - 2), :], WTilde[(k - 2)])
    #print(BetaTilde[k-1, :])
    
    # Loop for updating YTilde
    for i in range(tau_0, k):
        YTilde[:, i] = phi[0] + phi[1] * np.log(EstR[i - 1]) + np.dot(Z[i, :], BetaTilde[k - 1, :])
        print(YTilde[:, i])

    # Calculating the result
    result = 0
    for j in range(tau_0, k):
        result += I[j] * YTilde[:, j] - bias_corr_const * (np.exp(YTilde[:, j])) * Lambda[j]


    return -result

#def ell2(phi, k):
    """ ZYTilde[k - 1, :] = (np.log(EstR[0]) - phi[1] * np.log(R_0) - phi[0]) * (Z[0, :] - barZ[k - 1, :])
    EstR[0] = np.exp(phi[0] + phi[1] * np.log(R_0))
    for i in range(1, k - 1):
        ZYTilde[k - 1, :] += (np.log(EstR[i]) - phi[1] * np.log(EstR[i - 1]) - phi[0]) * (Z[i, :] - barZ[k - 1, :])
    BetaTilde[k - 1, :] = ZYTilde[k - 1, :] @ WTilde[k - 1]
    for i in range(tau_0, k):
        YTilde[i] = phi[0] + phi[1] * np.log(EstR[i - 1]) + Z[i, :] @ BetaTilde[k - 1, :]
    result = 0
    for j in range(tau_0, k):
        result += I[j] * YTilde[j] - bias_corr_const * (np.exp(YTilde[j])) * Lambda[j]
    return -result """

x=ell([0.5, 0.7], 5, bias_corr_const)
print(x)
#print(x)
# Minimize over the minus profile log-likelihood
for t in range(tau_0, T):
    res = minimize(ell, x0=[0.05, 0.7], args=(t, bias_corr_const),
                    bounds=[(-5, 5), ( 0.3,0.95)],method='L-BFGS-B')
    EstPhi[t , :] = res.x
    #print(EstPhi[t, :])
        # Update ZYHat
    #x=ell([0.5, 0.7], 15, bias_corr_const)

    #print(x)
    ZYHat[t - 1, :] = (np.log(EstR[0]) - EstPhi[t, 1] * np.log(R_0) - EstPhi[t, 0]) * (Z[0, :] - barZ[t - 1, :])
    for i in range(1, t - 1):
        ZYHat[t - 1, :] += (np.log(EstR[ i]) - EstPhi[t, 1] * np.log(EstR[i - 1]) - EstPhi[t, 0]) * (Z[i, :] - barZ[t - 1, :])

    # Update EstBeta
    EstBeta[t, :] = np.dot(ZYHat[t - 1, :] , WTilde[t - 1])

    # Update EstR
    EstR[t] = np.exp(EstPhi[t, 0] + EstPhi[t, 1] * np.log(EstR[ t - 1]) + np.dot(Z[t, :], EstBeta[t, :]))
    #print(EstR[:t])
    # Check condition and update if necessary
    if  t > 0 and greater_or_na(abs(EstR[t] - EstR[t - 1]), 5, 'L182?'):
        EstPhi[t, :] = EstPhi[t - 1, :]
        EstBeta[t, :] = EstBeta[t - 1, :]
        EstR[t] = EstR[ t - 1]
print(EstR[:t])
    # Return results (modify as necessary)
    #return EstPhi, EstBeta, EstR

def perform_estimation_and_plot(df, file_suffix):
    Z = df[['Z1', 'Z2']].values
    I = df['I'].values

    EstPhi, EstBeta, EstR = QSOEID(Z, I, NoCov, T, I_0, R_0, bias_corr_const, tau_0)

    R_0_df = df['R'].values

    plt.figure(figsize=(10, 6))

    # Plot R_0 from df
    plt.plot(R_0_df, label='R Truth', color='blue')

    # Plot R_0 from result
    plt.plot(EstR, label='R Estimated', color='red', linestyle='dashed')

    # Adding titles and labels
    plt.title(f'Comparison of R Values {file_suffix}')
    plt.xlabel('Time')
    plt.ylabel('R')
    plt.legend()

    results_folder = "Results_Origin"

    # Save the plot to a file
    plt.savefig(os.path.join(results_folder,f'R_comparison_{file_suffix}.png'))

    # Show the plot
    plt.show()

    # Saving EstPhi, EstBeta, EstR to files
    np.savetxt(f"EstPhi_{file_suffix}.csv", EstPhi, delimiter=",")
    np.savetxt(f"EstBeta_{file_suffix}.csv", EstBeta, delimiter=",")
    np.savetxt(f"EstR_{file_suffix}.csv", EstR, delimiter=",")

# Assuming df1 and df2 are your DataFrames
perform_estimation_and_plot(df, 'df')
perform_estimation_and_plot(df1, 'df1')
perform_estimation_and_plot(df2, 'df2')