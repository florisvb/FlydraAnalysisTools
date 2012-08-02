import floris_math
import adskalman.adskalman as adskalman
import numpy as np

###
def kalman_smoother(data, F, H, Q, R, initx, initv, plot=False):
    os = H.shape[0]
    ss = F.shape[0]
    
    interpolated_data = np.zeros_like(data)
    
    for c in range(os):
        interpolated_data[:,c] = floris_math.interpolate_nan(data[:,c])
        y = interpolated_data
        
    xsmooth,Vsmooth = adskalman.kalman_smoother(y,F,H,Q,R,initx,initv)
    
    if plot:
        plt.plot(xsmooth[:,0])
        plt.plot(y[:,0], '*')

    return xsmooth,Vsmooth
