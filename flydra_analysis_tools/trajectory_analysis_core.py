# commonly used functions in analyzing trajectories
import numpy as np
import scipy.linalg
import floris_math
import kalman_math
import numpyimgproc as nim
import matplotlib.pyplot as plt
from matplotlib import patches
import flydra_analysis_dataset as fad
    

########################################################################################################
# Culling
########################################################################################################

def mark_for_culling_based_on_min_frames(trajec, min_length_frames=10):
    if  trajec.length < min_length_frames:
        trajec.cull = True

def mark_for_culling_based_on_cartesian_position(trajec, ok_range, axis=0):
    if np.max(trajec.positions[:,axis]) > np.max(ok_range) or np.min(trajec.positions[:,axis]) < np.min(ok_range):
        trajec.cull = True
        
def mark_for_culling_based_on_speed(trajec, min_speed, axis=0):
    if np.min(trajec.speed) < min_speed:
        trajec.cull = True
         

########################################################################################################
# Polar
########################################################################################################

def polar_positions(trajec):
    
    r = np.sqrt(trajec.positions[:,0]**2 + trajec.positions[:,1]**2)
    theta = np.arccos(trajec.positions[:,0]/r)

    return r, theta

########################################################################################################
# Distance to point / post
########################################################################################################

def calc_xy_distance_to_point(trajec, xy_point):
    if type(xy_point) is not np.array or type(xy_point) is not np.ndarray:
        xy_point = np.array(xy_point)
    trajec.xy_distance_to_point = np.zeros_like(trajec.speed)
    for i, d in enumerate(trajec.xy_distance_to_point):
        d = scipy.linalg.norm(trajec.positions[i,0:2] - xy_point)
        trajec.xy_distance_to_point[i] = d
    
def calc_z_distance_to_point(trajec, z_point):
    trajec.z_distance_to_point = np.zeros_like(trajec.speed)
    for i, d in enumerate(trajec.z_distance_to_point):
        d = scipy.linalg.norm(trajec.positions[i,2] - z_point)
        trajec.z_distance_to_point[i] = d
        
def calc_xy_distance_to_post(trajec, top_center, radius):
    calc_xy_distance_to_point(trajec, top_center[0:2])
    calc_z_distance_to_point(trajec, top_center[2])
    trajec.xy_distance_to_post = trajec.xy_distance_to_point - radius
        
########################################################################################################
# Heading
########################################################################################################
        
def calc_heading(trajec):
    trajec.heading_norollover = floris_math.remove_angular_rollover(np.arctan2(trajec.velocities[:,1], trajec.velocities[:,0]), 3)
    ## kalman
    
    data = trajec.heading_norollover.reshape([len(trajec.heading_norollover),1])
    ss = 3 # state size
    os = 1 # observation size
    F = np.array([   [1,1,0], # process update
                     [0,1,1],
                     [0,0,1]],
                    dtype=np.float)
    H = np.array([   [1,0,0]], # observation matrix
                    dtype=np.float)
    Q = np.eye(ss) # process noise
    Q[0,0] = .01
    Q[1,1] = .01
    Q[2,2] = .01
    R = 1*np.eye(os) # observation noise
    
    initx = np.array([data[0,0], data[1,0]-data[0,0], 0], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = kalman_math.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)

    trajec.heading_norollover_smooth = xsmooth[:,0]
    trajec.heading_smooth_diff = xsmooth[:,1]*trajec.fps
    
    trajec.heading = floris_math.fix_angular_rollover(trajec.heading_norollover)
    trajec.heading_smooth = floris_math.fix_angular_rollover(trajec.heading_norollover_smooth)
    #trajec.heading_smooth_diff2 = xsmooth[:,2]
    
########################################################################################################
# Saccade detector
########################################################################################################

def get_angle_of_saccade(trajec, sac_range, method='integral', smoothed=True):
    
    if method != 'integral':
        f0 = sac_range[0]
        f1 = sac_range[-1]

        obj_ori_0 = trajec.velocities[f0] / np.linalg.norm(trajec.velocities[f0])   
        obj_ori_1 = trajec.velocities[f1] / np.linalg.norm(trajec.velocities[f1])  

        obj_ori_0_3vec = np.hstack( ( obj_ori_0, 0) ) 
        obj_ori_1_3vec = np.hstack( (obj_ori_1, 0 ) ) 

        sign_of_angle_of_saccade = np.sign( np.sum(np.cross( obj_ori_0, obj_ori_1 ) ) )

        cosangleofsaccade = np.dot(obj_ori_0, obj_ori_1)
        angleofsaccade = np.arccos(cosangleofsaccade)
         
        signed_angleofsaccade = -1*angleofsaccade*sign_of_angle_of_saccade
        
        return floris_math.fix_angular_rollover(signed_angleofsaccade)
    
    else:
        if smoothed is False:
            return floris_math.fix_angular_rollover(np.sum(trajec.heading_smooth_diff[sac_range]/100.)*-1)
        else:
            return floris_math.fix_angular_rollover(np.sum( floris_math.diffa(trajec.heading)[sac_range])*-1)
            
        
def calc_saccades(trajec, threshold_lo=300, threshold_hi=100000000, min_angle=10, plot=False):
    # thresholds in deg/sec
    # min_angle = minimum angle necessary to count as a saccade, deg
    

    fps = 100.
        
    possible_saccade_array = (np.abs(trajec.heading_smooth_diff)*180/np.pi > threshold_lo)*(np.abs(trajec.heading_smooth_diff)*180/np.pi < threshold_hi)
    possible_saccades = nim.find_blobs(possible_saccade_array, [3,100])
    
    if len(possible_saccades) == 1:
        if np.sum(possible_saccades[0]) == 0:
            possible_saccades = []
    
    trajec.all_saccades = []
    trajec.saccades = []
    trajec.sac_ranges = []
    
    
    if len(possible_saccades) > 0:
        for sac in possible_saccades:
            indices = np.where(sac==1)[0].tolist()
            if len(indices) > 0:
                # expand saccade range to make sure we get full turn
                #lo = np.max([indices[0]-5, 0])
                #hi = np.min([indices[-1]+5, len(trajec.speed)-2])
                #new_indices = np.arange(lo, hi).tolist()
                #tmp = np.where( np.abs(trajec.heading_diff_window[new_indices])*180/np.pi > 350 )[0].tolist()
                #indices = np.array(new_indices)[ tmp ].tolist()
                angle_of_saccade = np.abs(get_angle_of_saccade(trajec, indices)*180./np.pi)
                mean_speed = np.mean(trajec.speed[indices])
                if len(indices) > 3 and angle_of_saccade > 10: # and mean_speed > 0.005:
                    trajec.sac_ranges.append(indices)
                    s_rel = np.argmax( np.abs(trajec.heading_smooth_diff[indices]) )
                    s = indices[s_rel]
                    trajec.all_saccades.append(s)
            
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
        ax.plot(trajec.positions[:,0], trajec.positions[:,1], '-', color='black', alpha=1, linewidth=1, zorder=1)
            
        for sac in trajec.sac_ranges:        
            ax.plot(trajec.positions[sac,0], trajec.positions[sac,1], '-', color='red', alpha=1, linewidth=1, zorder=1+1)
        for s in trajec.all_saccades:
            x = trajec.positions[s, 0]
            y = trajec.positions[s, 1]
            saccade = patches.Circle( (x, y), radius=0.001, facecolor='blue', edgecolor='none', alpha=1, zorder=3)
            ax.add_artist(saccade)
                        
        post = patches.Circle( (0, 0), radius=0.009565, facecolor='black', edgecolor='none', alpha=1)
        ax.add_artist(post)
                        
        fig.savefig('saccade_trajectory.pdf', format='pdf')
