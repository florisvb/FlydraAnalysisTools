import os

class Config:
    def __init__(self):
    
        # data paths
        self.data_path = '../data/'
        self.raw_datasets_path = self.data_path + 'raw_datasets/'
        self.culled_datasets_path = self.data_path + 'culled_datasets/'
        self.h5_path = self.data_path + 'h5_files/'
        self.tmp_data_path = self.data_path + 'tmp/'
    
        # datasets
        self.raw_dataset_name = 'raw_dataset.pickle'
        self.culled_dataset_name = 'culled_dataset.pickle'
        
        # odor stuff
        self.odor_control_path = self.data_path + 'odor_data/'
        
        if 1: # default to find a file in the path
            try:
                cmd = 'ls ' + self.odor_control_path
                ls = os.popen(cmd).read()
                all_filelist = ls.split('\n')
                try:
                    all_filelist.remove('')
                except:
                    pass
                if len(all_filelist) == 1:
                    self.odor_control_filename = all_filelist[0]
                else:
                    print 'could not find unique odor control file in path'
            except:
                print 'could not find odor control path -- might need to make it first'
                
        if 0: # hard code the odor control file
            self.odor_control_filename = 'odor_control_signal_20120801_194226'
        
        # h5 files
        self.h5_files = 'all'
        
        # h5 reading parameters
        self.kalman_smoothing = True
        
        # additional data
        self.description = 'description of experiment'
        self.info = {   'flies': 'HCS', 
                        'odor': 'none', 
                        'description': self.description,
                    }
