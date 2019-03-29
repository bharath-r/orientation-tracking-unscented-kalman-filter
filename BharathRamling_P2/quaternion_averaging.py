
import numpy as np
from quaternion_functions import *

def quat_average(q,q0):
    
    q = np.matrix(q)
    qt = q0
    nr, nc = np.shape(q)
    qe = np.matrix(np.zeros([nr, 4]))
    ev = np.matrix(np.zeros([nr, 3]))
    
    pi = 22.0/7
    epsilon = 0.0001
    temp = np.zeros([1,4])
    
    for t in range(1000):
        for i in range(0,nr,1):
            q[i] = normalize_quaternion(q[i])
            qe[i] = multiply_quaternions(q[i],inverse_quaternion(qt))
            qs = qe[i,0]
            qv = qe[i,1:4]
            if np.round(norm_quaternion(qv),8) == 0:
                if np.round(norm_quaternion(qe[i]),8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    ev[i] = np.matrix([0, 0, 0])
            if np.round(norm_quaternion(qv),8) != 0:
                if np.round(norm_quaternion(qe[i]),8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    temp[0,0] = np.log(norm_quaternion(qe[i]))
                    temp[0,1:4] = np.dot((qv/norm_quaternion(qv)),math.acos(qs/norm_quaternion(qe[i])))
                    ev[i] = 2*temp[0,1:4]
                    ev[i] = ((-np.pi + (np.mod((norm_quaternion(ev[i]) + np.pi),(2*np.pi))))/norm_quaternion(ev[i]))*ev[i]
        e = np.transpose(np.mean(ev, 0))
        temp2 = np.array(np.zeros([4,1]))
        temp2[0] = 0
        temp2[1:4] = e/2.0
        qt = multiply_quaternions(exp_quaternion(np.transpose(temp2)),qt)

        if norm_quaternion(e) < epsilon:
            return qt, ev