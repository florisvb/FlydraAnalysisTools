from flydra_analysis_tools import flydra_analysis_dataset as fad
import pickle

def save_dataset_to_pickle(dataset, filename):
    data_dict = {}
    for key, trajec in dataset.trajecs.items():
        t = {}
        t.setdefault('position', trajec.positions)
        t.setdefault('velocity', trajec.velocities)
        t.setdefault('speed', trajec.speed)
        t.setdefault('odor', trajec.odor)
        t.setdefault('odorstimulus', trajec.odor_stimulus)
        
        data_dict.setdefault(key, t)
    
    f = open(filename, 'w')
    pickle.dump(data_dict, f)
    f.close()
    

def open_file(filename):
    
    f = open(filename)
    dataset = pickle.load(f)
    f.close()
    
    for key, trajectory in dataset.items():
        
        print trajectory.keys()
        print trajectory['position'][0]
        
    return dataset
