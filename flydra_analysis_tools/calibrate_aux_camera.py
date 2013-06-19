import adskalman
adskalman.adskalman = adskalman

import flydra.a2.core_analysis as core_analysis
import flydra.analysis.result_utils as result_utils
import flydra.reconstruct as reconstruct

import sys
import readh5
import math
import tables as PT
from optparse import OptionParser


import camera_math
import numpy as np
import copy

# open h5 file
# pull out 3D data of trajectories, and 2D data of the camera we wish to calibrate - only use frames where 2D data is unique
#   this approach should mean we are only using 'good' points... unless there are points that were tracked in 3D, and simultaneous the 2D camera saw one stray point
# feed 3D and 2D into DLT, save results as a xml from reconstruct.Reconstructor

def calibrate_from_h5( filename, cam_id, xml_filename):

    dataset = readh5.Dataset()
    dataset.load_data(      filename = filename, 
                            kalman_smoothing = False, 
                            load_2d=True)

    cam_ids = [dataset.cams2d[0][i].cam_id for i in range(len(dataset.cams2d[0]))]
    camcal = cam_ids.index(cam_id)
    resolution = dataset.cams2d[0][camcal].resolution                    
    max_reprojection_error = 4
    
    R = reconstruct.Reconstructor(filename)               
    
    # step through the frames, collect the 3d data and 2d data for the desired camera
    data3d = None
    data2d = None
    rejected2d = []
    print 'stepping through trajectories...'
    for k,v in dataset.trajecs.items():
        #dataset.trajecs[k].calc_covariance_pos()
        trajec = v
        frame_numbers = np.arange(0, trajec.length)
        for enum, frame in enumerate( frame_numbers ):
            # first get 2d, check to make sure it's a unique point:
            x,y,unique = dataset.cams2d[0][camcal].get_2d(frame)
            
            reprojection_error = 0
            for c in range(len(dataset.cams2d[0])):
                if c == 0:
                    continue
                cam = dataset.cams2d[0][c].cam_id    
                reproj_x, reproj_y = R.find2d(cam,dataset.trajecs[k].positions[enum])
                reprojection_error +=  np.abs(x - reproj_x) + np.abs(y - reproj_y)
                print c, np.abs(x - reproj_x)
            reprojection_error = reprojection_error / (len(dataset.cams2d[0])-1)
            
            
            if unique is True and math.isnan(x[0]) is False and math.isnan(y[0]) is False and reprojection_error<max_reprojection_error:
                print 'got data'
                if data3d is None:
                    data3d = np.array([p for p in dataset.trajecs[k].positions[enum]])
                    data2d = np.array([x[0],y[0]])
                else:
                    data3d = np.vstack( (data3d, [p for p in dataset.trajecs[k].positions[enum]]) )
                    data2d = np.vstack( (data2d, [x[0],y[0]]) )
            else:
                rejected2d.append([x[0], y[0]])
                
    print ''
    print 'Calibration Data Stats: '
    print '   num points: ', len(data2d)
    print '   num rejected points: ', len(rejected2d)
    print
    print 'note: if num rejected points is high, it is more likely that there are spurious points in the calibration set'
    
    Pmat, residuals = camera_math.DLT(data3d, data2d)
    
    
    print ''
    print 'Calibration Results: '
    print '   residuals from DLT: ', residuals
    print '   Pmat: '
    print Pmat
    print ''
    print 'old Pmat: '
    print R.Pmat[cam_id]/R.Pmat[cam_id][2,3]
    
    if 0:
        single_camera_calibration = reconstruct.SingleCameraCalibration(    cam_id=cam_id, # non-optional
                                                                            Pmat=Pmat,   # non-optional
                                                                            res=resolution,    # non-optional, resolution
                                                                            helper=None,
                                                                            scale_factor=None, # scale_factor is for conversion to meters (e.g. should be 1e-3 if your units are mm)
                                                                            no_error_on_intrinsic_parameter_problem = False,)
        reconstructor = reconstruct.Reconstructor(single_camera_calibration)
        reconstructor.save_to_xml_filename(xml_filename)
        print 'calibration saved as xml to: ', xml_filename


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--file", type="str", dest="filename", default=None,
                        help="filename for the h5 file, should contain 2d and 3d data")
    parser.add_option("--xml", type="str", dest="xml_filename", default=None,
                        help="filename for the xml of the new camera calibration")
    parser.add_option("--camcal", type="int", dest="camcal", default=None,
                        help="camera number for the camera to calibrate")
    (options, args) = parser.parse_args()
    
    calibrate_from_h5(options.filename, options.camcal, options.xml_filename)
    
