from flydra_analysis_tools import flydra_analysis_dataset as fad
import pickle
import scipy.io
import numpy as np

def save_dataset_to_pickle(dataset, filename, min_trajec_length=5, num_trajecs=100):
    data_dict = {}
    
    n = 0
    for key, trajec in dataset.trajecs.items():
        if trajec.length > min_trajec_length:
            if n < num_trajecs:
                t = {}
                t.setdefault('position', trajec.positions)
                #t.setdefault('velocity', trajec.velocities)
                #t.setdefault('speed', trajec.speed)
                #t.setdefault('odor', trajec.odor)
                #t.setdefault('odorstimulus', trajec.odor_stimulus)
                
                data_dict.setdefault(key, t)
                n += 1
                
    f = open(filename, 'w')
    pickle.dump(data_dict, f)
    f.close()
    
    
def save_dataset_to_mat(dataset, filename, min_trajec_length=5, num_trajecs=100, min_speed=0.07):
    data_dict = {}
    
    n = 0
    for key, trajec in dataset.trajecs.items():
        if trajec.length > min_trajec_length:
            if np.max(trajec.speed) > min_speed:
                if n < num_trajecs:
                    t = trajec.positions
                    k = 't' + key
                    data_dict.setdefault(k, t)
                    n += 1
    print n
                
    scipy.io.savemat(filename, data_dict)    
    


def open_file(filename):
    
    f = open(filename)
    dataset = pickle.load(f)
    f.close()
    
    for key, trajectory in dataset.items():
        
        print trajectory.keys()
        print trajectory['position'][0]
        
    return dataset
