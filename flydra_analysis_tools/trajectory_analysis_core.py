# commonly used functions in analyzing trajectories
import numpy as np
import scipy.linalg
import floris_math
import kalman_math
import numpyimgproc as nim
import matplotlib.pyplot as plt
from matplotlib import patches
import flydra_analysis_dataset as fad
    
import copy

import fly_plot_lib.plot as fpl
########################################################################################################
# Culling
########################################################################################################

def mark_for_culling_based_on_min_frames(trajec, min_length_frames=10):
    if  trajec.length < min_length_frames:
        trajec.cull = True

def mark_for_culling_based_on_cartesian_position(trajec, ok_range, axis=0):
    if np.max(trajec.positions[:,axis]) > np.max(ok_range) or np.min(trajec.positions[:,axis]) < np.min(ok_range):
        trajec.cull = True
        
def mark_for_culling_based_on_flight_volume(trajec, envelope, axis=0):
    middle_of_envelope = np.mean(envelope)
    dist_to_middle_of_envelope = np.abs(trajec.positions[:,axis] - middle_of_envelope)
    closest_to_middle = np.min(dist_to_middle_of_envelope)
    if (closest_to_middle > np.max(envelope)) or (closest_to_middle < np.min(envelope)):
        trajec.cull = True
    
        
def mark_for_culling_based_on_speed(trajec, min_speed, axis=0):
    if np.max(trajec.speed) < min_speed:
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

def calc_xy_distance_to_point(trajec, xy_point, normalized_for_speed=True):
    if type(xy_point) is not np.array or type(xy_point) is not np.ndarray:
        xy_point = np.array(xy_point)
    trajec.xy_distance_to_point = np.zeros_like(trajec.speed)
    for i, d in enumerate(trajec.xy_distance_to_point):
        d = scipy.linalg.norm(trajec.positions[i,0:2] - xy_point)
        trajec.xy_distance_to_point[i] = d
    
    if normalized_for_speed:
        trajec.xy_distance_to_point_normalized_by_speed = np.zeros_like(trajec.positions_normalized_by_speed[:,0])
        for i, d in enumerate(trajec.xy_distance_to_point_normalized_by_speed):
            d_normalized_by_speed = scipy.linalg.norm(trajec.positions_normalized_by_speed[i,0:2] - xy_point)
            trajec.xy_distance_to_point_normalized_by_speed[i] = d_normalized_by_speed
    
def calc_z_distance_to_point(trajec, z_point):
    trajec.z_distance_to_point = np.zeros_like(trajec.speed)
    for i, d in enumerate(trajec.z_distance_to_point):
        d = scipy.linalg.norm(trajec.positions[i,2] - z_point)
        trajec.z_distance_to_point[i] = d
        
def calc_distance_to_post(trajec, top_center, radius):
    calc_xy_distance_to_point(trajec, top_center[0:2])
    calc_z_distance_to_point(trajec, top_center[2])
        
    trajec.distance_to_post = np.zeros_like(trajec.xy_distance_to_point)
    for i, d in enumerate(trajec.xy_distance_to_point):
        if trajec.positions[i,2] < top_center[2]:
            trajec.distance_to_post[i] = trajec.xy_distance_to_point[i] - radius
        else:
            trajec.distance_to_post[i] = np.sqrt((trajec.xy_distance_to_point[i] - radius)**2 + (trajec.z_distance_to_point[i]**2))
            
def calc_xy_distance_to_post(trajec, top_center, radius):
    calc_xy_distance_to_point(trajec, top_center[0:2])
    trajec.xy_distance_to_post = trajec.xy_distance_to_point - radius
            
########################################################################################################
# Heading
########################################################################################################

def calc_velocities_normed(trajec):
    trajec.velocities_normed =  trajec.velocities / np.vstack((trajec.speed, trajec.speed, trajec.speed)).T
        
def calc_heading_from_velocities(velocities):
    heading_norollover = floris_math.remove_angular_rollover(np.arctan2(velocities[:,1], velocities[:,0]), 3)
    return heading_norollover
    
def calc_heading(trajec):
    trajec.heading_norollover = calc_heading_from_velocities(trajec.velocities)
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
    
def calc_heading_for_axes(trajec, axis='xy'):
    if axis == 'xy':
        axes = [1,0]
    elif axis == 'xz':
        axes = [2,0]
    elif axis == 'yz':
        axes = [2,1]
    
        
    heading_norollover_for_axes = floris_math.remove_angular_rollover(np.arctan2(trajec.velocities[:,axes[0]], trajec.velocities[:,axes[1]]), 3)
    ## kalman
    
    data = heading_norollover_for_axes.reshape([len(heading_norollover_for_axes),1])
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

    heading_norollover_smooth_for_axes = xsmooth[:,0]
    heading_smooth_diff_for_axes = xsmooth[:,1]*trajec.fps
    heading_for_axes = floris_math.fix_angular_rollover(heading_norollover_for_axes)
    heading_smooth_for_axes = floris_math.fix_angular_rollover(heading_norollover_smooth_for_axes)
    
    if axis == 'xy':
        trajec.heading_norollover_smooth_xy = heading_norollover_smooth_for_axes
        trajec.heading_smooth_diff_xy = heading_smooth_diff_for_axes
        trajec.heading_xy = heading_for_axes
        trajec.heading_smooth_xy = heading_smooth_for_axes
    if axis == 'xz':
        trajec.heading_norollover_smooth_xz = heading_norollover_smooth_for_axes
        trajec.heading_smooth_diff_xz = heading_smooth_diff_for_axes
        trajec.heading_xz = heading_for_axes
        trajec.heading_smooth_xz = heading_smooth_for_axes
    if axis == 'yz':
        trajec.heading_norollover_smooth_yz = heading_norollover_smooth_for_axes
        trajec.heading_smooth_diff_yz = heading_smooth_diff_for_axes
        trajec.heading_yz = heading_for_axes
        trajec.heading_smooth_yz = heading_smooth_for_axes
        
    #trajec.heading_smooth_diff2 = xsmooth[:,2]
    
    
def calc_airvelocity(trajec, windvelocity=[0,0,0]):
    if type(windvelocity) is list:
        windvelocity = np.array(windvelocity)
        
    trajec.airvelocities = copy.copy(trajec.velocities)
    
    for i in range(3):
        trajec.airvelocities[:,i] += windvelocity[i]
        
def calc_airheading(trajec):
    trajec.airheading_norollover = calc_heading_from_velocities(trajec.airvelocities)
    ## kalman
    
    data = trajec.airheading_norollover.reshape([len(trajec.airheading_norollover),1])
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

    trajec.airheading_smooth = floris_math.fix_angular_rollover(xsmooth[:,0])
    
    
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
        if smoothed:
            return floris_math.fix_angular_rollover(np.sum(trajec.heading_smooth_diff[sac_range]/100.))
        else:
            raise ValueError('need to do smoothed')            
        
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
                    trajec.saccades.append(indices)
                    s_rel = np.argmax( np.abs(trajec.heading_smooth_diff[indices]) )
                    s = indices[s_rel]
                    trajec.all_saccades.append(s)
                    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
        ax.plot(trajec.positions[:,0], trajec.positions[:,1], '-', color='black', alpha=1, linewidth=1, zorder=1)
            
        for sac in trajec.saccades:        
            ax.plot(trajec.positions[sac,0], trajec.positions[sac,1], '-', color='red', alpha=1, linewidth=1, zorder=1+1)
        for s in trajec.all_saccades:
            x = trajec.positions[s, 0]
            y = trajec.positions[s, 1]
            saccade = patches.Circle( (x, y), radius=0.001, facecolor='blue', edgecolor='none', alpha=1, zorder=3)
            ax.add_artist(saccade)
                        
        post = patches.Circle( (0, 0), radius=0.009565, facecolor='black', edgecolor='none', alpha=1)
        ax.add_artist(post)
                        
        fig.savefig('saccade_trajectory.pdf', format='pdf')
        
        
def calc_saccades_z(trajec, threshold_lo=200, threshold_hi=100000000, min_angle=10, plot=False):
    # thresholds in deg/sec
    # min_angle = minimum angle necessary to count as a saccade, deg

    fps = 100.
    
    heading_norollover_for_axes = floris_math.remove_angular_rollover(np.arctan2(trajec.speed_xy[:], trajec.velocities[:,2]), 3)
    ## kalman
    
    data = heading_norollover_for_axes.reshape([len(heading_norollover_for_axes),1])
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

    heading_norollover_smooth_for_axes = xsmooth[:,0]
    heading_smooth_diff_for_axes = xsmooth[:,1]*trajec.fps
    heading_for_axes = floris_math.fix_angular_rollover(heading_norollover_for_axes)
    heading_smooth_for_axes = floris_math.fix_angular_rollover(heading_norollover_smooth_for_axes)
    
    trajec.heading_altitude_smooth = heading_smooth_for_axes
    trajec.heading_altitude_smooth_diff = heading_smooth_diff_for_axes
    
    ## saccades
    
    possible_saccade_array = (np.abs(trajec.heading_altitude_smooth_diff)*180/np.pi > threshold_lo)*(np.abs(trajec.heading_altitude_smooth_diff)*180/np.pi < threshold_hi)
    possible_saccades = nim.find_blobs(possible_saccade_array, [3,100])
    
    if len(possible_saccades) == 1:
        if np.sum(possible_saccades[0]) == 0:
            possible_saccades = []
    
    trajec.saccades_z = []
    
    
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
                angle_of_saccade = np.abs(get_angle_of_saccade_z(trajec, indices)*180./np.pi)
                mean_speed = np.mean(trajec.speed[indices])
                if len(indices) > 3 and angle_of_saccade > 10: # and mean_speed > 0.005:
                    trajec.saccades_z.append(indices)

    if plot:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(trajec.positions[:,0], trajec.positions[:,2], color='black')
        for sac in trajec.saccades_z:
            ax.plot(trajec.positions[sac,0], trajec.positions[sac,2], color='red')
            
def get_angle_of_saccade_z(trajec, sac):
    return floris_math.fix_angular_rollover(np.sum(trajec.heading_altitude_smooth_diff[sac]/100.))
    
        
########################################################################################################
# Timestamp stuff
########################################################################################################

def calc_local_timestamps_from_strings(trajec):
    hr = int(trajec.timestamp_local[9:11])
    mi = int(trajec.timestamp_local[11:13])
    se = int(trajec.timestamp_local[13:15])
    trajec.timestamp_local_float = hr + mi/60. + se/3600.    
        
########################################################################################################
# Normalized speed
########################################################################################################
    
def calc_positions_normalized_by_speed(trajec, normspeed=0.2, plot=False):
    
    distance_travelled = np.cumsum(trajec.speed)
    distance_travelled_normalized = np.arange(distance_travelled[0], distance_travelled[-1], normspeed)
    warped_time = np.interp(distance_travelled_normalized, distance_travelled, trajec.time_fly)
    
    trajec.positions_normalized_by_speed = np.zeros([len(warped_time), 3])
    for i in range(3):
        trajec.positions_normalized_by_speed[:,i] = np.interp(warped_time, trajec.time_fly, trajec.positions[:,i])
    
    trajec.time_normalized_by_speed = warped_time
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = trajec.positions[:,0]
        y = trajec.positions[:,1]
        speed = trajec.speed
        fpl.colorline(ax,x,y,speed)    
        
        x_warped = trajec.positions_normalized_by_speed[:,0]
        y_warped = trajec.positions_normalized_by_speed[:,1]
        ax.plot(x_warped, y_warped, '.')
    

########################################################################################################
# Landing vs Not Landing
########################################################################################################
    
def calc_post_behavior(trajec, top_center, radius, landing_threshold=0.003, initial_threshold=0.05, landing_speed=0.1, boomerang_threshold=0.003, takeoff_threshold=0.004):
    try:
        tmp = trajec.distance_to_post
    except:
        calc_distance_to_post(trajec, top_center, radius)
    try:
        tmp = trajec.distance_to_post_min
    except:
        calc_distance_to_post_min(trajec, top_center, radius)
    
    trajec.post_behavior = []
    trajec.post_behavior_frames = []
    trajec.residency_time = None
    
    if np.max(trajec.speed) > 0.01 and np.max(trajec.distance_to_post) > initial_threshold: # double check to make sure fly flew
        
        if trajec.distance_to_post[0] < takeoff_threshold:
            trajec.post_behavior.append('takeoff')
            frame_of_takeoff = np.where(trajec.distance_to_post > takeoff_threshold)[0][0]
            trajec.post_behavior_frames.append(frame_of_takeoff)
            candidate_behavior = 'landing'
        else:
            candidate_behavior = 'landing'
            
        was_flying = False
        if trajec.distance_to_post[0] > 0.01:
            was_flying = True
        
        frame = 0
        while frame < trajec.length-1:
            if len(trajec.post_behavior_frames) > 0:
                most_recent_frame_of_interest = trajec.post_behavior_frames[-1]
            else:
                most_recent_frame_of_interest = 0
                
            if candidate_behavior == 'landing':
                for frame in range(most_recent_frame_of_interest, trajec.length):
                    if trajec.distance_to_post[frame] < landing_threshold and was_flying and trajec.speed[frame] < landing_speed:
                        trajec.post_behavior.append('landing')
                        frame_of_landing = frame
                        trajec.post_behavior_frames.append(frame_of_landing)
                        candidate_behavior = 'takeoff'
                        was_flying = False
                        break
            elif candidate_behavior == 'takeoff':
                for frame in range(most_recent_frame_of_interest, trajec.length):
                    if trajec.distance_to_post[frame] > takeoff_threshold and np.max(trajec.distance_to_post[frame:frame+20]) > 0.02:
                        trajec.post_behavior.append('takeoff')
                        frame_of_takeoff = frame
                        trajec.post_behavior_frames.append(frame_of_takeoff)
                        candidate_behavior = 'landing'
                        break
            
                        
            if trajec.distance_to_post[frame] > 0.01:
                was_flying = True
        
        
        if 'landing' in trajec.post_behavior and 'takeoff' in trajec.post_behavior:
            check_for_takeoff = False
            check_for_landing = True
            for f, behavior in enumerate(trajec.post_behavior):
                if behavior == 'landing':
                    check_for_takeoff = True
                    check_for_landing = False
                    landing_frame = trajec.post_behavior_frames[f]
                if check_for_takeoff:
                    if behavior == 'takeoff':
                        residency_time = (trajec.post_behavior_frames[f] - landing_frame) / trajec.fps
                        trajec.post_behavior.append('boomerang')
                        trajec.residency_time = residency_time 
                        check_for_takeoff = False
        
        if trajec.residency_time is None and len(trajec.post_behavior) > 0:
            if trajec.post_behavior[-1] == 'landing':
                trajec.residency_time = (trajec.length - trajec.post_behavior_frames[-1]) / trajec.fps
        
            
            
        
def calc_distance_to_post_min(trajec, top_center, radius):
    try:
        tmp = trajec.distance_to_post
    except:
        calc_distance_to_post(trajec, top_center, radius)
        
    trajec.distance_to_post_min = np.min(trajec.distance_to_post)
    trajec.distance_to_post_min_index = np.argmin(trajec.distance_to_post)
    
    
    
    
#def calc_post_approaches(trajec):
    
    
    
#########################################################################3

# acceleration kalmanized

def get_acceleration(trajec):
    ## kalman
    velocities = trajec.velocities
    
    data = velocities
    ss = 6 # state size
    os = 3 # observation size
    F = np.array([   [1,0,0,1,0,0], # process update
                     [0,1,0,0,1,0],
                     [0,0,1,0,0,1],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1]],                  
                    dtype=np.float)
    H = np.array([   [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]], # observation matrix
                    dtype=np.float)
    Q = 0.01*np.eye(ss) # process noise
    
    R = 1*np.eye(os) # observation noise
    
    initx = np.array([velocities[0,0], velocities[0,1], velocities[0,2], 0, 0, 0], dtype=np.float)
    initv = 0*np.eye(ss)
    xsmooth,Vsmooth = kalman_math.kalman_smoother(data, F, H, Q, R, initx, initv, plot=False)

    accel_smooth = xsmooth[:,3:]*100.
    
    return accel_smooth
    
    


    
    


