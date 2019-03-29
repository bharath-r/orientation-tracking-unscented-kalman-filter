import numpy as np
from quaternion_functions import *
from quaternion_averaging import *
import time 
import matplotlib.pyplot as plt   

#%% Kalman Predict 

def get_sigma_points(P, Q, qt):
    """ Gets sigma points using cholesky decomposition"""
    decomp = np.linalg.cholesky(P+Q)
    n,m = np.shape(P)
    left_vec = decomp*np.sqrt(2*n)
    right_vec = -decomp*np.sqrt(2*n)
    new_vec = np.hstack((left_vec, right_vec))
    nr,nc = np.shape(new_vec)
    
    q = np.matrix(np.zeros([4,6]))
    v = np.matrix(np.zeros([3,6]))
    for i in range(0,nc):
        temp = vec2quat(new_vec[:,i])
        q[:,i] = np.transpose(multiply_quaternions(temp,qt))
        
    return q
    
    
def get_new_covariance(P,error):
    """ Computes new covariance by taking error into account """    
    nr,nc = np.shape(error)
    next_cov = np.zeros([nc,nc])
    for i in range(0,nr):
        temp_cov = np.transpose(error[i])*error[i]
        next_cov += temp_cov
        
    return next_cov/12
    

def kalman_predict(qt, ut, P, Q):
    """ Performs the prediction step of Kalman Filter """    
    qu = vec2quat(ut)
    sigma_points = np.transpose(get_sigma_points(P, Q, qt))
    motion_sig = np.zeros(np.shape(sigma_points))
    for i in range(0,6):
        motion_sig[i] = multiply_quaternions(sigma_points[i], qu)
    next_qt, error = quat_average(motion_sig, qt)
    next_cov = get_new_covariance(P,error)
    return next_qt, next_cov, sigma_points, error


#%% Kalman Update Begins here
    
def get_new_sigma_points(sigma_points, g, R):
    """ Gets new sigma points after Kalman Prediction step """
    new_sigma = np.zeros(np.shape(sigma_points))
    z = np.zeros([np.shape(sigma_points)[0]-1,np.shape(sigma_points)[1]])
    for i in range(0,np.shape(sigma_points)[0]):
        new_sigma[i] = multiply_quaternions(multiply_quaternions(inverse_quaternion(sigma_points[i]),g),sigma_points[i])
        
    z = new_sigma[:,1:]
    z_mean = np.mean(z,0)
    return z, z_mean
    
    
def get_pzz(z, z_mean):
    """ Calculates Pzz from z and z_mean"""
    temp = np.matrix(z - z_mean)
    pzz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
    for i in range(0,np.shape(temp)[0]):
        pzz_temp = np.transpose(temp[i])*temp[i]
        pzz += pzz_temp
        
    return pzz/12.0
    
    
def get_pxz(error, z, z_mean):
    """ Calculates Pxz from error, z and z_mean"""    
    temp = np.matrix(z - z_mean)
    pxz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
    for i in range(0,np.shape(error)[0]):
        pxz_temp = np.transpose(error[i])*temp[i]
        pxz += pxz_temp
        
    return pxz/12.0
    
    
def kalman_gain(pxz,pvv):
    """Calculates Kalman Gain"""    
    K = np.dot(pxz,np.linalg.inv(pvv))
    return K
    
    
def kalman_update(g, R, sigma_points, error,acc_i, predicted_q_i, predicted_cov_i):
    """ Performs Kalman Update Step """
    z, z_mean = get_new_sigma_points(sigma_points, g, R)
    z = np.matrix(z)
    z_mean = np.matrix(z_mean)
    pzz = get_pzz(z, z_mean)
    pvv = pzz + R
    pxz = get_pxz(error, z, z_mean)
    K = kalman_gain(pxz, pvv)
    I = np.transpose(acc_i - z_mean)
    KI = vec2quat(np.transpose(K*I))
    new_q = np.matrix(np.empty([1,4]))
    new_q = multiply_quaternions(KI,predicted_q_i)
    new_pxx = predicted_cov_i - np.dot(np.dot(K,pvv),np.transpose(K))
    return new_q, new_pxx, pxz, pvv, pzz, K, I
    
#%% Testing the functions    
    
ti = time.time() # Keeps track of time
    
#%% Initialization of variables
P = 0.00001*np.identity(3)
Q = 0.00001*np.identity(3)
R = 0.0001*np.identity(3)
q0 = np.matrix([1, 0, 0, 0])
qt = np.matrix([1, 0, 0, 0])
ut = gyro_val[0]
g = np.matrix([0, 0, 0, 1])

#%% Actual update begins here
for i in range(0,np.shape(gyro_val)[0]):
    
    if i==0:
        ut = gyro_val[i]*imu_ts[0]
        predicted_q = q0
    else:
        ut = gyro_val[i]*(imu_ts[i] - imu_ts[i-1])
    
    next_q, next_cov, sigma_points, error = kalman_predict(qt, ut, P, Q) # Computes values from Kalman Predict
    qt_updated, p_updated, pxz, pvv, pzz, K, I = kalman_update(g, R, sigma_points, error, acc_val[i], next_q, next_cov) # Computes values from Kalman Update
    qt = qt_updated # Updated value is the new current value
    P = p_updated
    predicted_q = np.vstack((predicted_q, qt)) # Stores the values
    
#%% Conversion from Quaternion to Euler
phi = np.zeros([np.shape(predicted_q)[0], 1])
theta = np.zeros([np.shape(predicted_q)[0], 1])
psi = np.zeros([np.shape(predicted_q)[0], 1])

for i in range(np.shape(predicted_q)[0]):
    R = quat2rot(predicted_q[i])
    phi[i], theta[i], psi[i] = rot2euler(R)
    
    
elapsed = time.time() - ti
print elapsed

#%% Plots vicon euler angles against computed euler angles
# RED is the predicted values
# Blue is Vicon values
plt.figure(1)
plt.subplot(311)
plt.plot(vicon_phi, 'b', phi, 'r')
plt.ylabel('Roll')
plt.subplot(312)
plt.plot(vicon_theta, 'b', theta, 'r')
plt.ylabel('Pitch')
plt.subplot(313)
plt.plot(vicon_psi, 'b', psi, 'r')
plt.ylabel('Yaw')
plt.show()
