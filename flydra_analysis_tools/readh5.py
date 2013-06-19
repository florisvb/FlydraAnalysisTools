import sys
import adskalman
adskalman.adskalman = adskalman
 
import numpy
import numpy as np

import tables as PT
import flydra.analysis.result_utils as result_utils
import flydra.a2.core_analysis as core_analysis
from optparse import OptionParser
import flydra_analysis_dataset as fad

class Cam2d:
    def __init__(self, cam_id, h5):
    
        ''' data2_distorted structure:
        /data2d_distorted (Table(3091,)) '2d data'
          description := {
          0: "camn": UInt16Col(shape=(), dflt=0, pos=0),
          1: "frame": UInt64Col(shape=(), dflt=0, pos=1),
          2: "timestamp": Float64Col(shape=(), dflt=0.0, pos=2),
          3: "cam_received_timestamp": Float64Col(shape=(), dflt=0.0, pos=3),
          4: "x": Float32Col(shape=(), dflt=0.0, pos=4),
          5: "y": Float32Col(shape=(), dflt=0.0, pos=5),
          6: "area": Float32Col(shape=(), dflt=0.0, pos=6),
          7: "slope": Float32Col(shape=(), dflt=0.0, pos=7),
          8: "eccentricity": Float32Col(shape=(), dflt=0.0, pos=8),
          9: "frame_pt_idx": UInt8Col(shape=(), dflt=0, pos=9),
          10: "cur_val": UInt8Col(shape=(), dflt=0, pos=10),
          11: "mean_val": Float32Col(shape=(), dflt=0.0, pos=11),
          12: "sumsqf_val": Float32Col(shape=(), dflt=0.0, pos=12)}
          byteorder := 'little'
          chunkshape := (585,)

        '''
        
        self.cam_id = cam_id
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        self.camn = cam_id2camns[self.cam_id]
        self.fps = result_utils.get_fps( h5 )
        self.resolution = getattr(h5.root.images,'Basler_21111538').read().shape

        # pull out only the data that is relevant to this camera
        # note: multiple rows for individual frames if there were multiple 2D objects detected in that frame
        all_data = h5.root.data2d_distorted[:]
        print 'ALL DATA', all_data
        this_idx = numpy.nonzero( all_data['camn']==self.camn )[0]
        self.data = all_data[this_idx]
        
    def get_2d(self, frame):
        unique = True
        rows = []
        for enum, row in enumerate( self.data ):
            if self.data[enum][1] == frame:
                rows.append(enum)
        x = [self.data[row][4] for row in rows]
        y = [self.data[row][5] for row in rows]
        if len(x) > 1:
            unique = False
        return x,y,unique      

class Dataset:

    def __init__(self):
            
        # trajectory related initializations
        self.trajecs = {}
        self.stimulus = None
        self.n_artificial_trajecs = 0
        
        self.datasets = []
        self.cams2d = []
        
    def set_stimulus (self, stimulus):
        self.stimulus = stimulus

        
    def load_data (self,    filename, 
                            calibration_file = None, 
                            objs = None, 
                            obj_filelist = None, 
                            kalman_smoothing = True, 
                            fps = None,     
                            dynamic_model = None, 
                            load_2d=True,):
    
        self.datasets.append(len(self.datasets)+1)
        
        # raw h5 - this is the 2d raw camera data:
        if load_2d is True:
            h5 = PT.openFile( filename, mode='r' )
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
            cam_ids = cam_id2camns.keys()
            cams2d = []
            for cam_id in cam_ids:
                cams2d.append( Cam2d(cam_id, h5) )
            self.cams2d.append(cams2d)
            
        # set up analyzer
        ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file, extra) = ca.initial_file_load(filename)
        
        # data set defaults
        if fps is None:
            fps = result_utils.get_fps(data_file)
        if dynamic_model is None:
            try:
                dyn_model = extra['dynamic_model_name']
            except:
                dyn_model = 'EKF mamarama, units: mm'
        if dynamic_model is not None:
            dyn_model = dynamic_model
        
        # if kalman smoothing is on, then we cannot use the EKF model - remove that from the model name
        print '**Kalman Smoothing is: ', kalman_smoothing, ' **'
        if kalman_smoothing is True:
            dyn_model = dyn_model[4:]
        print 'using dynamic model: ', dyn_model
            
        if objs is None and obj_filelist is None:
            print "running through all object id's, this might take a while..."
            obj_only = use_obj_ids # this is all the unique object id's 
        if obj_filelist is not None:
            tmp = np.loadtxt(obj_filelist,delimiter=',')
            obj_only = np.array(tmp[:,0], dtype='int')
        elif objs is not None:
            obj_only = np.array(objs)
            
        print 'loading data.... '
        for obj_id in obj_only:
            try: 
                kalman_rows =  ca.load_data( obj_id, data_file,
                                     dynamic_model_name = dyn_model,
                                     use_kalman_smoothing= kalman_smoothing,
                                     frames_per_second= fps)
            except:
                print 'object id failed to load (probably no data): ', obj_id
                continue
            
            # couple object ID dictionary with trajectory objects
            traj_id = (str(self.datasets[-1])+'_'+str(obj_id))
            print kalman_rows[0][0]
            print 'wtf'
            self.trajecs.setdefault(traj_id, fad.Trajectory(traj_id, kalman_rows, extra=extra, fps = fps) )

if __name__ == '__main__':

    dataset = Dataset()

