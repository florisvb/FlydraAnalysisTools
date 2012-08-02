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