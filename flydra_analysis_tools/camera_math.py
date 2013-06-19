import pylab
import matplotlib.ticker as ticker
import pytz, datetime, time
pacific = pytz.timezone('US/Pacific')
import tables
import numpy as np
import numpy
from scipy import linalg
import matplotlib.pyplot as plt
import time
import scipy
import scipy.optimize.optimize as optimize

orig_determinant = numpy.linalg.det
def determinant( A ):
    return orig_determinant( numpy.asarray( A ) )

def center(P):
    # there is also a copy of this in flydra.reconstruct, but included
    # here so this file doesn't depend on that.
    
    # P is Mhat
    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    C_ = numpy.array( [[ X/T, Y/T, Z/T ]] ).T
    return C_
    
def build_rot_mat(rot_axis, rot_angle):

    rot_axis = rot_axis / scipy.linalg.norm( rot_axis )
    Rn = np.array([ [ rot_axis[0]**2+(1-rot_axis[0]**2)*np.cos(rot_angle),
                        rot_axis[0]*rot_axis[1]*(1-np.cos(rot_angle))-rot_axis[2]*np.sin(rot_angle),
                        rot_axis[0]*rot_axis[2]*(1-np.cos(rot_angle))+rot_axis[1]*np.sin(rot_angle)],
                       [rot_axis[0]*rot_axis[1]*(1-np.cos(rot_angle))+rot_axis[2]*np.sin(rot_angle),
                        rot_axis[1]**2+(1-rot_axis[1]**2)*np.cos(rot_angle),
                        rot_axis[1]*rot_axis[2]*(1-np.cos(rot_angle))-rot_axis[0]*np.sin(rot_angle)],
                       [rot_axis[0]*rot_axis[2]*(1-np.cos(rot_angle))-rot_axis[1]*np.sin(rot_angle),
                        rot_axis[1]*rot_axis[2]*(1-np.cos(rot_angle))+rot_axis[0]*np.sin(rot_angle),
                        rot_axis[2]**2+(1-rot_axis[2]**2)*np.cos(rot_angle)] ])
                        
    return Rn

def build_Bc(X3d,x2d):
    B = []
    c = []

    assert len(X3d)==len(x2d)
    if len(X3d) < 6:
        print 'WARNING: 2 equations and 11 unknowns means we need 6 points!'
    for i in range(len(X3d)):
        X = X3d[i,0]
        Y = X3d[i,1]
        Z = X3d[i,2]
        x = x2d[i,0]
        y = x2d[i,1]

        B.append( [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z] )
        B.append( [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z] )

        c.append( x )
        c.append( y )
    return numpy.array(B), numpy.array(c)

def getMhat(data = None, data_file = None):
    
    if data_file is not None:
        data = np.loadtxt(  data_file,delimiter=',')

    # in terms of alpha
    x2d_alpha = data[:,0:2]
    x2d = np.tan(x2d_alpha)
    X3d = data[:,3:6]

    B,c = build_Bc(X3d,x2d)
    DLT_avec_results = numpy.linalg.lstsq(B,c)
    a_vec,residuals = DLT_avec_results[:2]
    a_vec = a_vec.T
    Mhat = numpy.array(list(a_vec)+[1])
    Mhat.shape=(3,4)
    
    #print Mhat

    cam_id = 'newcam'
    #print cam_id,center(Mhat).T,'residuals:',float(residuals)
    
    return Mhat, residuals[0]
    
def DLT(data3d, data2d, normalize=True):
    # Mhat is, I think, the same as Pmat
    B,c = build_Bc(data3d,data2d)
    DLT_avec_results = numpy.linalg.lstsq(B,c)
    a_vec,residuals = DLT_avec_results[:2]
    a_vec = a_vec.T
    Mhat = numpy.array(list(a_vec)+[1])
    Mhat.shape=(3,4)
    
    if normalize:
        K,R,t = decomp(Mhat)
        K /= K[2,2]
        camera_center = np.zeros([3,4])
        camera_center[:,0:3] = np.eye(3)
        camera_center[:,3] = -1*center(Mhat).T
        print K
        print R
        print t
        Mhat = np.dot( np.dot(K,R), camera_center )
        K,R,t = decomp(Mhat)
        print
        print
        print K
        print R
        print t
    return Mhat, residuals[0]
    
def replace_camera_center(P, camera_center):
    
    K,R,t = decomp(P)
    tmp = np.zeros([3,4])
    tmp[:,0:3] = np.eye(3)
    tmp[:,3] = -1*camera_center
    print K, R, tmp
    Pnew = np.dot( np.dot(K,R), tmp )
    return Pnew   

def decomp(P, SHOW=0):

    # t is NOT the camera center in world coordinates - use function center(Mhat) for that
    # P = [M | -Mt], M = KR
    
    if len( P ) < 2:
        P = numpy.array( [[ 3.53553e2,   3.39645e2,  2.77744e2,  -1.44946e6 ],
                          [-1.03528e2,   2.33212e1,  4.59607e2,  -6.32525e5 ],
                          [ 7.07107e-1, -3.53553e-1, 6.12372e-1, -9.18559e2 ]] )


    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    C_ = numpy.transpose(numpy.array( [[ X/T, Y/T, Z/T ]] ))

    M = P[:,:3]

    # do the work:
    # RQ decomposition: K is upper-triangular matrix and R is
    # orthogonal. Both are components of M such that KR=M

    K,R = scipy.linalg.rq(M) # added to scipy 0.5.3
    Knorm = K/K[2,2]

    # So now R is the rotation matrix (which is orthogonal) describing the
    # camera orientation. K is the intrinsic parameter matrix.

    t = numpy.dot( -R, C_ )

    # reconstruct P via eqn 6.8 (p. 156)
    P_ = numpy.dot( K, numpy.concatenate( (R, t), axis=1 ) )

    if SHOW:
        print 'P (original):'
        print P
        print

        print 'C~ (center):'
        print C_
        print

        print 'K (calibration):'
        print K
        print

        print 'normalized K (calibration):'
        print Knorm
        print

        print 'R (orientation):' # same as rotation matrix
        print R
        print

        print 't (translation in world coordinates):'
        print t
        print

        print 'P (reconstructed):'
        print P_
        print
    
    return K,R,t
    
def getw2c(Mhat):

    K,R,t = decomp(Mhat)
    T = np.concatenate( (R, t), axis=1 )
    
    return T
