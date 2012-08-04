import os

class Config:
    def __init__(self):
    
        # data paths
        self.data_path = 'data/'
        self.raw_datasets_path = os.path.join(self.data_path, 'raw_datasets/')
        self.culled_datasets_path = os.path.join(self.data_path, 'culled_datasets/')
        self.h5_path = os.path.join(self.data_path, 'h5_files/')
        self.tmp_data_path = os.path.join(self.data_path, 'tmp/')
    
        # datasets
        self.raw_dataset_name = 'raw_dataset.pickle'
        self.culled_dataset_name = 'culled_dataset.pickle'
        
        # figures
        self.figure_path = 'figures/'
        self.figures = ['heatmaps', 'spagetti', 'activity']
        
        # h5 files
        self.h5_files = 'all'
        
        # h5 reading parameters
        self.kalmanized = False 
        self.kalman_smoothing = True
        
        # additional data
        self.description = 'example directory of data and figure files'
        self.info = {   'flies': 'PCF', 
                        'description': self.description,
                        'date' : '20120802',
                    }
                    
        print
        print '*********************'
        print 'Configuration info: '
        print self.info
        print '*********************'
        print
