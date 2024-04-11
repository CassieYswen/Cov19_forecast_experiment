import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, logistic, gamma
import csv



# Helper functions
def cumulative_gamma(x, shape, scale):
    return gamma.cdf(x, a=shape, scale=scale)

def logit(x):
    return np.log(x / (1 - x))

T = 120
NoCov = 2
R_0 = 3
I_0 = 500
bias_corr_const = np.exp(-0.001 / 2)
Omega = np.array([cumulative_gamma(i, 2.5, 3) - cumulative_gamma(i-1, 2.5, 3)
                  for i in range(1, 26)]) / cumulative_gamma(25, 2.5, 3)
#print the range of Omega
print(np.min(Omega),np.max(Omega))
OraclePhi = np.array([0.5, 0.7])
OracleBeta = np.array([-0.02, -0.125]).reshape(NoCov, 1)
OracleBeta1 = np.array([-0.02, -0.125, -0.03]).reshape(NoCov + 1, 1)
OraclePhi1 = np.array([0.5, 0.5, 0.3])
Rmin1 = 2

# Function definitions
#def generate_data_true(T, NoCov, R_0, I_0, Omega, OraclePhi, OracleBeta, bias_corr_const):
# Generate daily incident cases and covariates data
Z = np.zeros((T, NoCov))
for t in range(T):
    Z[t, 0] = 5-(T/8)+((2*t)/8) + np.random.normal(0, 3)
Z[:, 1] = logit(np.random.uniform(0.01, 0.99, T)) + 2
#print the min and max of Z [:,0] and Z[:,1]
print(np.min(Z[:,0]),np.max(Z[:,0]),np.min(Z[:,1]),np.max(Z[:,1]))
# R[t], the instantaneous reproduction number at time t
R = np.zeros(T)
epsilon = norm.rvs(size=T, scale=np.sqrt(-2 * np.log(bias_corr_const)))
#print the first 10 elements of epsilon
print(epsilon[:10])

R[0] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(R_0) + np.dot(Z[0, :] , OracleBeta) + epsilon[0])

for t in range(1, T):
    R[t] = np.exp(OraclePhi[0] + OraclePhi[1] * np.log(R[t-1]) + np.dot(Z[t, :] , OracleBeta) + epsilon[t])
#print the min and max of R
print(np.min(R),np.max(R))
# Generate I[t], number of incidences at time t
I = np.zeros(T)
lambda_values = []  # List to store lambda values

I[0] = np.random.poisson(R[0] * I_0 * Omega[0])

for t in range(1, 25):
    lambda_val = R[t] * (I_0 * Omega[t] + np.dot(I[:t], Omega[t-1::-1]))
    lambda_values.append(lambda_val)  # Store the lambda value
    I[t] = np.random.poisson(lambda_val)

for t in range(25, T):
    reversed_I_segment = I[t-25:t][::-1]
    lambda_val = R[t]*np.dot(reversed_I_segment, Omega[:25])
    lambda_values.append(lambda_val)  # Store the lambda value
    if np.dot(reversed_I_segment, Omega[:25]) <= 100000:
        I[t] = np.random.poisson( lambda_val)
    else:
        multipconstant = int(( lambda_val) // 100000)
        residueconstant = ( lambda_val) % 100000
        I[t] = np.sum(np.random.poisson(100000, multipconstant)) + np.random.poisson(residueconstant)

#print the min and max if I
#print(np.min(I),np.max(I))
#save I as csv file
np.savetxt("I.csv", I, delimiter=",")     

# Save the lambda values to a CSV file
with open('lambda_values.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['t', 'lambda'])  # Writing headers
    for t, lambda_val in enumerate(lambda_values, 1):
        writer.writerow([t, lambda_val])

# Save Z, I, R as a DataFrame
df = pd.DataFrame({'Z1': Z[:, 0], 'Z2': Z[:, 1], 'R': R, 'I': I})
#return df