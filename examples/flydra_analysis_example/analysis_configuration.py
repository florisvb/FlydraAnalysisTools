class Config:
    def __init__(self):
    
        # datasets
        self.raw_dataset_name = 'raw_datasets/raw_dataset.pickle'
        self.culled_dataset_name = 'culled_datasets/culled_dataset.pickle'
        
        # h5 files
        self.h5_path = 'h5_files/'
        self.h5_files = 'all'
        
        # h5 reading parameters
        self.kalman_smoothing = True
        
        # additional data
        self.info = {}
