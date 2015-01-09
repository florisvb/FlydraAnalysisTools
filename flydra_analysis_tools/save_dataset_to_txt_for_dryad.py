from flydra_analysis_tools import flydra_analysis_dataset as fad
import pickle
import csv
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def save_datasets_to_txt(datasets, filename, attributes, labels, keys=None, info='Description: ', num_trajecs='all', additional_column_data=[]):
    csvfile = open(filename, 'wb')
    datawrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    datawrite.writerow([info])
    datawrite.writerow(labels)
        
    if keys is None:
        keys = [None for i in range(len(datasets))]
    for i, dataset in enumerate(datasets):
        save_dataset_to_txt(dataset, filename, attributes, labels, keys=keys[i], info='info', num_trajecs=num_trajecs, additional_column_data=[], datawrite=datawrite)
    csvfile.close()
    
def save_dataset_to_txt(dataset, filename, attributes, labels, keys=None, info=None, num_trajecs='all', additional_column_data=[], datawrite=None):
    if num_trajecs == 'all':
        num_trajecs = len(dataset.trajecs.keys())

    if keys is None:
        keys = dataset.trajecs.keys()
        
    if datawrite is None:
        csvfile = open(filename, 'wb')
        datawrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if info is not None:
            datawrite.writerow([info])
        datawrite.writerow(labels)
        closefile = True
    else:
        closefile = False
    
    n = 0
    for key in keys:
        strkey = 's' + key
        trajec = dataset.trajecs[key]
        #print n, key
        if n > num_trajecs:
            break
        
        for i in range(trajec.length):
            row = [strkey]
            for attribute in attributes:
                if attribute == 'timestamp_local_float':
                    row.append( trajec.timestamp_local_float + trajec.time_fly[i] )
                else:
                    try: # if the attribute is there, else write 'NA'
                        if trajec.__getattribute__(attribute) is None:
                            row.append('None')
                        else:
                            if type(trajec.__getattribute__(attribute)) is str: # if it is a string
                                row.append( trajec.__getattribute__(attribute) )
                            else:
                                try: # if the attribute has rows
                                    l = len(trajec.__getattribute__(attribute))
                                    
                                    attribute_val = trajec.__getattribute__(attribute)[i]
                                    try: # if the attribute has columns
                                        iterable = iter(attribute_val)
                                        for val in iterable:
                                            row.append(val)
                                    except:
                                        row.append(attribute_val)
                            
                                except:
                                    row.append( trajec.__getattribute__(attribute) )
                    except:
                        row.append('NA')
                    
            row.extend(additional_column_data)
            #print n, len(row), key
            #print
            datawrite.writerow(row)
        n += 1

    if closefile:
        csvfile.close()
    
def get_datawrite_for_odor_dataset_to_txt(filename, info=None):
    csvfile = open(filename, 'wb')
    datawrite = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if info is not None:
        datawrite.writerow([info])
    labels = ['object_id', 'timestamp_local_time', 'time_fly', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'speed', 'heading_xy', 'odor_stimulus', 'odor_value', 'time_since_odor_relative_to_exit', 'time_since_odor_relative_to_entry', 'wind', 'visual_environment', 'odor_type']
    datawrite.writerow(labels)
    return datawrite, csvfile
    
def append_odor_dataset_to_datawrite(dataset, datawrite, num_trajecs='all', wind=0.4, visual_environment='checkerboard', odor_type='ethanol'):
    attributes = ['timestamp_local_float', 'time_fly', 'positions', 'velocities', 'speed', 'heading_smooth', 'odor_stimulus', 'odor', 'time_since_odor_relative_to_exit', 'time_since_odor_relative_to_entry']
    additional_column_data = [wind, visual_environment, odor_type]
    save_dataset_to_txt(dataset, None, attributes, None, keys=None, info=None, num_trajecs=num_trajecs, additional_column_data=additional_column_data, datawrite=datawrite)
    
def write_odor_dataset_to_txt(dataset, filename, num_trajecs='all', wind=0.4, visual_environment='checkerboard', odor_type='ethanol', info=None):
    datawrite, csvfile = get_datawrite_for_odor_dataset_to_txt(filename, info=info)
    append_odor_dataset_to_datawrite(dataset, datawrite, num_trajecs=num_trajecs, wind=wind, visual_environment=visual_environment, odor_type=odor_type)
    csvfile.close()
    
def save_xyz_dataset_to_txt(dataset, filename, num_trajecs='all', info='Description: ', min_length=100):
    
    labels = ['object_id', 'timestamp_local_time', 'time_fly', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'speed', 'heading_xy']
    attributes = ['timestamp_local_float', 'time_fly', 'positions', 'velocities', 'speed', 'heading_smooth']
    
    keys = []
    for key, trajec in dataset.trajecs.items():
        if trajec.length > min_length:
            keys.append(key)
    
    save_dataset_to_txt(dataset, filename, attributes, labels, keys=keys, info=info, num_trajecs=num_trajecs)
    
    
def plot_headings(dataset, keys):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    headings = []
    for key in keys:
        trajec = dataset.trajecs[key]
        headings.extend(trajec.heading_smooth.tolist())
        
    fpl.histogram(ax, [np.array(headings)], colors=['black'], bins=50)




