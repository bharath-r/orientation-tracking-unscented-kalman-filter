import numpy as np
import math


def multiply_quaternions(q, r):

    """ Takes in 2 1*4 quaternions, returns a 1*4 quaternion"""
    """Multiplies two quaternions 'q' and 'r'. Returns 't' where t = q*r"""
    t = np.empty(([1,4]))
    t[:,0] = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
    t[:,1] = (r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2])
    t[:,2] = (r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1])
    t[:,3] = (r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0])

    return t


def conjugate_quaternion(q):

    """Returns conjugate of quaternion q"""
    t = np.empty([4, 1])
    t[0] = q[0]
    t[1] = -q[1]
    t[2] = -q[2]
    t[3] = -q[3]

    return t


def divide_quaternions(q, r):

    """Divides two quaternions 'q' and 'r'. Returns quaternion t where t = q/r"""

    t = np.empty([4, 1])
    t[0] = ((r[0] * q[0]) + (r[1] * q[1]) + (r[2] * q[2]) + (r[3] * q[3])) / ((r[0] ** 2) + (r[1] ** 2) + (r[2] ** 2) + (r[3] ** 2))
    t[1] = (r[0] * q[1] - (r[1] * q[0]) - (r[2] * q[3]) + (r[3] * q[2])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[2] = (r[0] * q[2] + r[1] * q[3] - (r[2] * q[0]) - (r[3] * q[1])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[3] = (r[0] * q[3] - (r[1] * q[2]) + r[2] * q[1] - (r[3] * q[0])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)

    return t


def inverse_quaternion(q):

    """Takes in a 1*4 quaternion. Returns a 1*4 quaternion. Returns the inverse of quaternion 'q'. Denoted by q^-1"""
    t = np.empty([4, 1])
    t[0] = q[:,0] / np.power(norm_quaternion(q),2)
    t[1] = -q[:,1] / np.power(norm_quaternion(q),2)
    t[2] = -q[:,2] / np.power(norm_quaternion(q),2)
    t[3] = -q[:,3] / np.power(norm_quaternion(q),2)
    
    t = np.transpose(t)
    
    return t


def norm_quaternion(q):

    """Returns norm of the quaternion."""
    t = np.sqrt(np.sum(np.power(q,2)))
    return t


def normalize_quaternion(q):

    """Returns a normalized quaternion"""
    return q/norm_quaternion(q)

def rotate_vector_by_quaternion(q,v):

    """Returns the vector rotated by a quaternion. V must be a column vector!!"""
    v_rotated = []
    v_rotated = np.matrix([[(1 - 2*(q[2]^2) - 2*(q[3]^2)), 2*(q[1]*q[2] + q[0]*q[3]), 2*((q[1]*q[3]) - (q[0]*q[2]))], 
                           [2*(q[1]*q[2] - q[0]*q[3]), (1 - 2*(q[1]^2) - 2*(q[3]^2)), 2*((q[2]*q[3]) + (q[0]*q[1]))],
                           [2*(q[1]*q[3] + q[0]*q[2]), 2*((q[2]*q[3]) - (q[0]*q[1])), (1 - 2*(q[1]^2) - 2*(q[2]^2))]])*v
#    v_temp = multiply_quaternions(q,conjugate_quaternion(q))
#    v_rotated = multiply_quaternions(v, v_temp)
    return v_rotated


def quat2rot(q):
    
    """Converts a quaternion into a rotation matrix"""
    # Using the second method listed on this link: 
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    q = normalize_quaternion(q)
    qhat = np.zeros([3,3])
    qhat[0,1] = -q[:,3]
    qhat[0,2] = q[:,2]
    qhat[1,2] = -q[:,1]
    qhat[1,0] = q[:,3]
    qhat[2,0] = -q[:,2]
    qhat[2,1] = q[:,1]

    R = np.identity(3) + 2*np.dot(qhat,qhat) + 2*np.array(q[:,0])*qhat
    #R = np.round(R,4)
    return R

    
def rot2euler(R):
    
    """ Gets the euler angles corresponding to the rotation matrix R"""
    phi = -math.asin(R[1,2])
    theta = -math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
    psi = -math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
    
    return phi, theta, psi
    

def rot2quat(R):
    
    """ Converts from rotation matrix R into a quaternion"""
    tr = R[0,0] + R[1,1] + R[2,2];

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2 
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S

    elif ((R[0,0] > R[1,1]) and (R[0,0] > R[2,2])):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    
    elif (R[1,1] > R[2,2]):
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S

    q = [[qw],[qx],[qy],[qz]]
    temp = np.sign(qw)
    q = np.multiply(q,temp)
    return q

    
def vec2quat(r):
    
    """ Converts from Vector into a quaternion"""
    r = r/2.0
    q = np.matrix(np.zeros([4,1]))
    q[0] = math.cos(np.linalg.norm(r))
    if np.linalg.norm(r) == 0:
        temp = np.transpose(np.matrix([0, 0, 0]))
    else:
        temp = np.transpose(np.matrix((r/np.linalg.norm(r))*(math.sin(np.linalg.norm(r)))))
    q[1:4] = temp
    q = np.transpose(q)    
    return q
    
    
def quat2vec(q):
    
    """ Converts from a quaternion into a vector"""
    qs = q[:,0]
    qv = q[:,1:4]
    if np.linalg.norm(qv) == 0:
        v = np.transpose(np.matrix([0,0,0]))
    else:
        v = 2*((qv/np.linalg.norm(qv))*math.acos(qs/np.linalg.norm(q)))
    return v
    
    
def log_quaternion(qe):
    
    """ Calculates the log of the quaternion"""
    qe = np.transpose(qe)
    qs = qe[0]
    qv = qe[1:4]
    log_q = np.zeros(np.shape(qe))

    log_q[0] = np.log(norm_quaternion(qe))
    log_q[1:4] = np.dot(qv/norm_quaternion(qv), math.acos(qs/norm_quaternion(qe)))
    return log_q
    
    
def exp_quaternion(q):
    
    """ Calculates the exp of the quaternion"""
    q = np.transpose(q)
    qs = q[0]
    qv = q[1:4]
    exp_q = np.zeros(np.shape(q))
    
    exp_q[0] = math.cos((norm_quaternion(qv)))
    exp_q[1:4] = np.dot(normalize_quaternion(qv), math.sin(norm_quaternion(qv)))
    return np.transpose(math.exp(qs)*exp_q)