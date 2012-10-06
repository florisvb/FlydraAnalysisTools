# class based wrapper for .h5 files from Flydra
# written by Floris van Breugel

# depends on flydra.analysis and flydra.a2 -- if these are not installed, point to the appropriate path

# once you have this dataset class, the data no longer depends on flydra, and therefore, matplotlib 0.99
    
import numpy as np
import pickle
import sys
import os
import time
import copy

import matplotlib.pyplot as plt

try:
    import flydra.a2.core_analysis as core_analysis
    import flydra.analysis.result_utils as result_utils
except:
    print 'Need to install flydra if you want to load raw data!'
    print 'For unpickling, however, flydra is not necessary'

class Dataset:

    def __init__(self):
        self.trajecs = {}
        self.h5_files_loaded = []
	
    def test(self, filename, kalman_smoothing=False, dynamic_model=None, fps=None, info={}, save_covariance=False):
        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(filename)
        return ca, obj_ids, use_obj_ids, is_mat_file, data_file, extra

    def load_data(self, filename, kalman_smoothing=False, dynamic_model=None, fps=None, info={}, save_covariance=False):
        # use info to pass information to trajectory instances as a dictionary. 
        # eg. info={"post_type": "black"}
        # save_covariance: set to True if you need access to the covariance data. Keep as False if this is not important for analysis (takes up lots of space)
        
        # set up analyzer
        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(filename)
        data_file.flush()

        # data set defaults
        if fps is None:
            fps = result_utils.get_fps(data_file)
        if dynamic_model is None:
            try:
	            dyn_model = extra['dynamic_model_name']
            except:
                print 'cannot find dynamic model'
                print 'using EKF mamarama, units: mm'
                dyn_model = 'EKF mamarama, units: mm'
        if dynamic_model is not None:
            dyn_model = dynamic_model

        # if kalman smoothing is on, then we cannot use the EKF model - remove that from the model name
        print '** Kalman Smoothing is: ', kalman_smoothing, ' **'
        if kalman_smoothing is True:
            dyn_model = dyn_model[4:]
        print 'using dynamic model: ', dyn_model
        print 'framerate: ', fps
        print 'loading data.... '
        
        self.dynamic_model = dyn_model
        self.fps = fps
        
        # load object id's and save as Trajectory instances
        for obj_id in use_obj_ids:
            print 'processing: ', obj_id
            try: 
                print obj_id
                kalman_rows = ca.load_data( obj_id, data_file,
                                 dynamic_model_name = dyn_model,
                                 use_kalman_smoothing= kalman_smoothing,
                                 frames_per_second= fps)
            except:
                print 'object id failed to load (probably no data): ', obj_id
                continue

            # couple object ID dictionary with trajectory objects
            filenamebase = os.path.basename(filename)
            trajecbase = filenamebase.rstrip('5').rstrip('.h').lstrip('DATA').rstrip('.kalmanized')
            trajec_id = trajecbase + '_' + str(obj_id) # filename details + original object id - this is unique
            tmp = Trajectory(trajec_id, kalman_rows, info=info, fps=fps, save_covariance=save_covariance, extra=extra)
            self.trajecs.setdefault(trajec_id, tmp)
            
        self.h5_files_loaded.append(os.path.basename(filename))
            
        return
        
    def del_trajec(self, key):
        del (self.trajecs[key])
    
    def get_trajec(self, n=0):
        key = self.trajecs.keys()[n]
        return self.trajecs[key]
        
    def save(self, filename):
        return save(self, filename)
        

class Trajectory(object):
    def __init__(self, trajec_id, kalman_rows=None, info={}, fps=None, save_covariance=False, extra=None):
        self.key = trajec_id
        
        if kalman_rows is None:
            return
        
        self.info = info
        self.fps = fps
        
        """
        kalman rows =   [0] = obj_id
                        [1] = frame
                        [2] = timestamp
                        [3:6] = positions
                        [6:9] = velocities
                        [9:12] = P00-P02
                        [12:18] = P11,P12,P22,P33,P44,P55
                        [18:21] = rawdir_pos
                        [21:24] = dir_pos
                        
                        dtype=[('obj_id', '<u4'), ('frame', '<i8'), ('timestamp', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('xvel', '<f8'), ('yvel', '<f8'), ('zvel', '<f8'), ('P00', '<f8'), ('P01', '<f8'), ('P02', '<f8'), ('P11', '<f8'), ('P12', '<f8'), ('P22', '<f8'), ('P33', '<f8'), ('P44', '<f8'), ('P55', '<f8'), ('rawdir_x', '<f4'), ('rawdir_y', '<f4'), ('rawdir_z', '<f4'), ('dir_x', '<f4'), ('dir_y', '<f4'), ('dir_z', '<f4')])
                        
        Covariance Matrix (P):

          [ xx  xy  xz
            xy  yy  yz
            xz  yz  zz ]
            
            full covariance for trajectories as velc as well
                        
        """

        self.obj_id = kalman_rows[0][0]
        self.first_frame = int(kalman_rows[0][1])
        self.fps = float(fps)
        self.length = len(kalman_rows)

        print 'local time: ', time.strftime( '%Y%m%d_%H%M%S', time.localtime(extra['time_model'].framestamp2timestamp(kalman_rows[0][1])) )
        print 'epochtime: ', extra['time_model'].framestamp2timestamp(kalman_rows[0][1])

        self.timestamp_local = time.strftime( '%Y%m%d_%H%M%S', time.localtime(extra['time_model'].framestamp2timestamp(kalman_rows[0][1])) )
        self.timestamp_epoch = extra['time_model'].framestamp2timestamp(kalman_rows[0][1])

        self.time_fly = np.linspace(0,self.length/self.fps,self.length, endpoint=True) 
        self.positions = np.zeros([self.length, 3])
        self.velocities = np.zeros([self.length, 3])
        self.speed = np.zeros([self.length])
        self.speed_xy = np.zeros([self.length])
        self.length = len(self.speed)
        self.cull = False
        self.frame_range_roi = (0, self.length)

        for i in range(len(kalman_rows)):
            for j in range(3):
                self.positions[i][j] = kalman_rows[i][j+3]
                self.velocities[i][j] = kalman_rows[i][j+6]
            self.speed[i] = np.sqrt(kalman_rows[i][6]**2+kalman_rows[i][7]**2+kalman_rows[i][8]**2)
            self.speed_xy[i] = np.sqrt(kalman_rows[i][6]**2+kalman_rows[i][7]**2)
            
        if save_covariance:
            self.covariance_position = [np.zeros([3,3]) for i in range(self.length)]
            self.covariance_velocity = [np.zeros([3]) for i in range(self.length)]
            for i in range(self.length):
                self.covariance_position[i][0,0] = kalman_rows[i]['P00']
                self.covariance_position[i][0,1] = kalman_rows[i]['P01']
                self.covariance_position[i][0,2] = kalman_rows[i]['P02']
                self.covariance_position[i][1,1] = kalman_rows[i]['P11']
                self.covariance_position[i][1,2] = kalman_rows[i]['P12']
                self.covariance_position[i][2,2] = kalman_rows[i]['P22']
                self.covariance_velocity[i][0] = kalman_rows[i]['P33']
                self.covariance_velocity[i][1] = kalman_rows[i]['P44']
                self.covariance_velocity[i][2] = kalman_rows[i]['P55']
		
###################################################################################################
# General use dataset functions
###################################################################################################

# recommended function naming scheme:
    # calc_foo(trajec): will save values to trajec.foo (use sparingly on large datasets)
    # get_foo(trajec): will return values
    # use iterate_calc_function(dataset, function) to run functions on all the trajectories in a dataset
    
def save(dataset, filename):
    print 'saving dataset to file: ', filename
    fname = (filename)  
    fd = open( fname, mode='w' )
    pickle.dump(dataset, fd)
    fd.close()
    return 1
    
def load(filename):
    fname = (filename)
    fd = open( fname, mode='r')
    print 'loading data... '
    dataset = pickle.load(fd)
    fd.close()
    return dataset
    
def copy_dataset(dataset):
    new_dataset = Dataset()
    for key, trajec in dataset.trajecs.items():
        new_trajec = Trajectory(copy.copy(trajec.key))
        for key, item in trajec.__dict__.items():
            new_trajec.__setattr__(copy.copy(key), copy.copy(item))
        new_dataset.trajecs.setdefault(key, new_trajec)
    return new_dataset
        
# this function lets you write functions that operate on the trajectory class, but then easily apply that function to all the trajectories within a dataset class. It makes debugging new functions easier/faster if you write to operate on a trajectory. 
def iterate_calc_function(dataset, function, keys=None, *args, **kwargs):
    # keys allows you to provide a list of keys to perform the function on, default is all the keys
    if keys is None:
        keys = dataset.trajecs.keys()
    for key in keys:
        trajec = dataset.trajecs[key]
        function(trajec, *args, **kwargs)
        
def merge_datasets(dataset_list):
    # dataset_list should be a list of datasets
    dataset = Dataset()
    n = 0
    for d in dataset_list:
        dataset.h5_files_loaded.extend(d.h5_files_loaded)
        for k, trajec in d.trajecs.iteritems():
            # check to see if trajec.key is in the dataset so far
            if trajec.key in dataset.trajecs.keys():
                new_trajec_id = str(n) + '_' + k
            else:
                new_trajec_id = k
            trajec.key = new_trajec_id
            
            # backwards compatibility stuff: NOTE: not fully backwards compatible! (laziness)
            if type(trajec) is not Trajectory:
                new_trajec = Trajectory(new_trajec_id)
                new_trajec.positions = trajec.positions
                new_trajec.velocities = trajec.velocities
                new_trajec.speed = trajec.speed
                new_trajec.length = len(trajec.speed)
                new_trajec.fps = trajec.fps
                dataset.trajecs.setdefault(new_trajec_id, new_trajec)
            else:
                dataset.trajecs.setdefault(new_trajec_id, trajec)
        n += 1
    return dataset
    
def make_mini_dataset(dataset, nkeys = 500):
    # helpful if working with large datasets and you want a small one to test out code/plots
    new_dataset = Dataset()
    n = 0
    for k, trajec in dataset.trajecs.iteritems():
        n += 1
        new_trajec_id = str(n) + '_' + k # make sure we use unique identifier
        trajec.key = new_trajec_id
        new_dataset.trajecs.setdefault(new_trajec_id, trajec)
        if n >= nkeys:
            break
    return new_dataset
    
def count_flies(dataset, attr=None, val=None):
    
    print 'n flies: ', len(dataset.trajecs.keys())
    
    def count_for_attribute(a, v):
        n = 0
        for k, trajec in dataset.trajecs.iteritems():
            test_val = trajec.__getattribute__(a)
            if test_val == v:
                n += 1
        print a + ': ' + v + ': ' + str(n)
        
    if attr is not None:
        if type(attr) is list:
            for i, a in enumerate(attr):
                count_for_attribute(a, val[i])
        else:
            count_for_attribute(attr, val)
            
def get_basic_statistics(dataset):
    
    n_flies = 0
    mean_speeds = []
    frame_length = []
    for k, trajec in dataset.trajecs.items():
        n_flies += 1
        mean_speeds.append( np.mean(trajec.speed) )
        frame_length.append( len(trajec.speed) )
        
    mean_speeds = np.array(mean_speeds)
    frame_length = np.array(frame_length)
        
    print 'num flies: ', n_flies
    print 'mean speed: ', np.mean(mean_speeds)
    print 'mean frame length: ', np.mean(frame_length)
            
def load_single_h5(filename, save_as=None, save_dataset=True, return_dataset=True, kalman_smoothing=True, save_covariance=False, info={}):
    # filename should be a .h5 file
    if save_as is None:
        save_as = 'dataset_' + filename.split('/')[-1]
    dataset = Dataset()
    dataset.load_data(filename, kalman_smoothing=True, save_covariance=False, info=info)
    
    if save_dataset:
        print 'saving dataset...'
        save(dataset, save_as)

    if return_dataset:
        return dataset

def load_all_h5s_in_directory(path, print_filenames_only=False, kalmanized=True, savedataset=True, savename='merged_dataset', kalman_smoothing=True, dynamic_model=None, fps=None, info={}, save_covariance=False, tmp_path=''):
    # if you get an error, try appending an '/' at the end of the path
    # only looks at files that end in '.h5', assumes they are indeed .h5 files
    # kalmanized=True will only load files that have a name like kalmanized.h5

    cmd = 'ls ' + path
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
        
    filelist = []
    for i, filename in enumerate(all_filelist):
        if filename[-3:] != '.h5':
            pass
        else:
            if kalmanized:
                if filename[-13:] == 'kalmanized.h5':
                    filelist.append(path + filename)
            else:
                if filename[-13:] == 'kalmanized.h5':
                    pass
                else:
                    filelist.append(path + filename)
    
    print
    print 'loading files: '
    for filename in filelist:
        print filename
    if print_filenames_only:
        return
        
    dataset_list = []
        
    n = 0
    for filename in filelist:
        n += 1
        dataset = Dataset()
        dataset.load_data(filename, kalman_smoothing=kalman_smoothing, dynamic_model=dynamic_model, fps=fps, info=info, save_covariance=save_covariance)
        tmpname = tmp_path + 'dataset_tmp_' + str(n)
        print 'saving tmp file to: ', tmpname
        save(dataset, tmpname)
        dataset_list.append(dataset)
        
    merged_dataset = merge_datasets(dataset_list)
    
    if savedataset:
        save(merged_dataset, savename)
    
    return merged_dataset
        
def get_keys_with_attr(dataset, attributes, values, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
    if type(attributes) is not list:
        attributes = [attributes]
    if type(values) is not list:
        values = [values]
    keys_with_attr = []
    for key in keys:
        trajec = dataset.trajecs[key]
        add_key = True
        for i, attr in enumerate(attributes):
            attr_val_for_trajec = trajec.__getattribute__(attr)
            if type(attr_val_for_trajec) is list:
                if values[i] not in attr_val_for_trajec:
                    add_key = False
            else:
                if trajec.__getattribute__(attr) != values[i]:
                    add_key = False
        if add_key:
            keys_with_attr.append(key)
    return keys_with_attr
    
def print_values_for_attributes_for_keys(dataset, attributes, keys, index=0):
    if type(attributes) is not list:
        attributes = [attributes]
    if type(index) is not list:
        index = [index]
    for key in keys:
        trajec = dataset.trajecs[key]
        print_str = key
        for a, attr in enumerate(attributes):
            attribute = trajec.__getattribute__(attr)
            if type(attribute) in [str, float, int, long, np.float64]:
                print_str += ' -- ' + attr + ': ' + str(attribute)
            else:
                if type(index[a]) is int:
                    print_str += ' -- ' + attr + ': ' + str(attribute[index[a]])
                elif type(index[a]) is str:
                    print_str += ' -- ' + attr + ': ' + str(np.__getattribute__(index[a])(attribute))
                else:
                    print_str += ' -- ' + attr + ': ' + str(attribute)
        print print_str
                
def get_trajec_with_attr(dataset, attr, val, n=0):
    keys = get_keys_with_attr(dataset, attr, val)
    if n > len(keys):
        n = -1
    return dataset.trajecs[keys[n]]
    
def set_attribute_for_trajecs(dataset, attr, val, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
    
    for key in keys:
        trajec = dataset.trajecs[key]
        trajec.__setattr__(attr, val)
        
def make_dataset_with_attribute_filter(dataset, attr, val):
    new_dataset = Dataset()
    keys = get_keys_with_attr(dataset, attr, val)
    for key in keys:
        new_dataset.trajecs.setdefault(key, dataset.trajecs[key])
    return new_dataset        

def plot_simple(dataset, keys, axis=[0,1], view='cartesian'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if view == 'cartesian':
        for key in keys:
            trajec = dataset.trajecs[key]
            ax.plot(trajec.positions[:,axis[0]], trajec.positions[:,axis[1]])
            ax.plot(trajec.positions[0,axis[0]], trajec.positions[0,axis[1]], '.', color='green')
            ax.plot(trajec.positions[-1,axis[0]], trajec.positions[-1,axis[1]], '.', color='red')
            
    elif view == 'radial':
        for key in keys:
            trajec = dataset.trajecs[key]
            ax.plot(trajec.xy_distance_to_post, trajec.positions[:,2])
            ax.plot(trajec.xy_distance_to_post[0], trajec.positions[0,2], '.', color='green')
            ax.plot(trajec.xy_distance_to_post[-1], trajec.positions[-1,2], '.', color='red')

def get_similar_trajecs(dataset, master_trajec, attributes={'positions': 0.01, 'velocities_normed': .01}):
    '''
    Search through dataset and find trajectories that match the master_trajec in all attributes listed
    The value associated with the attribute is the threshold acceptable for being "similar"
    eg. "positions": 0.01 will search through all the positions of two trajectories and if any frames exist where the two positions are within 0.01 (in terms of 3d), then the trajectory is considered similar for that attribute.
    '''
    similar_keys = []
    for key, trajec in dataset.trajecs.items():
        is_similar = True
        for attribute in attributes.keys():
            err = np.abs(trajec.__getattribute__(attribute) - master_trajec.__getattribute__(attribute))
            errsum = np.sum(err, axis=1)
            minerr = np.min(errsum)
            if not minerr < attributes[attribute]:
                is_similar = False
        if is_similar:
            similar_keys.append(key)
    return similar_keys
    
def get_keys_with_similar_attributes(dataset, attributes={'positions': [0, 0, 0], }, attribute_errs={'positions': 0.01}):
    similar_keys = []
    for key, trajec in dataset.trajecs.items():
        is_similar = True
        for attribute, val in attributes.items():
            if type(val) is str:
                if not trajec.__getattribute__(attribute) == val:
                    is_similar = False
            else:
                err = np.abs(trajec.__getattribute__(attribute) - np.asarray(val))
                errsum = np.sum(err, axis=1)
                minerr = np.min(errsum)
                if not minerr < attribute_errs[attribute]:
                    is_similar = False
        if is_similar:
            similar_keys.append(key)
    return similar_keys
    
def save_frame_to_key_dict(dataset):
    frame_to_key = {}
    for key in dataset.trajecs.keys():
        print key
        trajec = dataset.trajecs[key]
        camera_frames = trajec.first_frame + np.arange(0, trajec.length)
        for camera_frame in camera_frames:
            if frame_to_key.has_key(camera_frame): 
                frame_to_key[camera_frame].append(key)
            else:
                frame_to_key.setdefault(camera_frame, [key])
    dataset.frame_to_key = frame_to_key
    return 
        
###################################################################################################
# Example usage
###################################################################################################

def example_load_single_h5_file(filename):
    # filename should be a .h5 file
    info = {'post_type': 'black', 'post_position': np.zeros([3]), 'post_radius': 0.0965}
    dataset = Dataset()
    dataset.load_data(filename, kalman_smoothing=True, save_covariance=False, info=info)
    print 'saving dataset...'
    #save(dataset, 'example_load_single_h5_file_pickled_dataset')

    #for k, trajec in dataset.trajecs.iteritems():
    #    print 'key: ', k, 'trajectory length: ', trajec.length, 'speed at end of trajec: ', trajec.speed[-1]

    return dataset

if __name__ == "__main__":
    pass
    
