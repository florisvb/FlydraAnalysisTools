import os
from distutils import dir_util.copy_tree as copy_tree 



def generate_filestructure(experiment_folder_name, path):
    
    # make base directory    
    if path[-1] != '/':
        path += '/'
    experiment_path = path + experiment_folder_name + '/'
    
    try:
        os.mkdir(experiment_path)
    except:
        os.mkdir(path)
        os.mkdir(experiment_path)
    
    # make subfolders
    figures_directory = experiment_path + 'figures'
    culled_datasets_directory = experiment_path + 'culled_datasets'
    raw_datasets_directory = experiment_path + 'raw_datasets'
    h5_directory = experiment_path + 'h5_files'
    
    subfolders_to_create = [figures_directory, culled_datasets_directory, raw_datasets_directory, h5_directory]
    
    for subfolder in subfolders_to_create:
        os.mkdir(subfolder)
        
    return experiment_path



def generate_list_of_active_h5_files(h5_files, experiment_path):
    if experiment_path[-1] != '/':
        experiment_path += '/'
    
    h5_file_list = []
    for h5_file in h5_files:
        h5_file_name = h5_file.split('/')[-1]
        h5_file_list.append(h5_file_name)
    
    h5_file_list_name = experiment_path + 'h5_file_list'
    fd = open(h5_file_list_name, 'w')
    
    for h5_file in h5_file_list:
        fd.write(h5_file)
        fd.write('\n')
        
    fd.close()



def initialize_experiment_from_h5_files(experiment_path, h5_files):

    generate_list_of_active_h5_files(h5_files, experiment_path)
    
    for h5_file in h5_files:
        h5_file_name_no_path = h5_file.split('/')[-1]
        new_h5_file_name = experiment_path + 'h5_files/' + h5_file_name_no_path
        os.rename(h5_file, new_h5_file_name)
        


def copy_template_path(template_path, experiment_path):
    if not os.path.exists(experiment_path):
        copy_tree(template_path, experiment_path)
    else:
        print 'That path exists, and I will not overwrite it!'
        return 0
        
    



if __name__ == '__main__':
    
    path = '/home/floris/DATA/'
    experiment_folder_name = 'my_test_experiment_1'
    experiment_path = path + experiment_folder_name
    template_path = 'data_analysis_template/'
    
    
    h5_files = ['/home/floris/test123.h5', '/home/floris/test456.h5']
    
    initialize_experiment(experiment_folder_name, path, h5_files)
    
    
