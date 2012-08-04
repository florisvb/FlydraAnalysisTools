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
        
        # odor stuff
        self.odor = False
        self.odor_control_path = os.path.join(self.data_path, 'odor_data/')
        self.odor_control_files = None
        
        # h5 files
        self.h5_files = 'all'
        
        # h5 reading parameters
        self.kalmanized = False 
        self.kalman_smoothing = True
        
        # additional data
        self.description = 'PCF flies 5 min on/off, black post'
        self.info = {   'flies': 'PCF', 
                        'odor': 'ethanol', 
                        'description': self.description,
                        'date' : '20120802',
                        'wind' : 0.4,
                        'rpm_tunnel' : 21,
                    }
                    
        print
        print '*********************'
        print 'Configuration info: '
        print self.info
        print '*********************'
        print
