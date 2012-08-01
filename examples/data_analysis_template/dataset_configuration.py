


class Config:
    def __init__(self, filename):
        self.filename = filename

    def read_config(filename):
        
        config = open(filename)
        line = 'something'
        while line is not None:
            line = config.readline()
            if line[0] == '#':
                continue
            else:
                info = line.split('=')
                if 'h5_files' in info[0]:
                    h5_files_as_string = info[1]
                    h5_files = h5_files_as_string.split(',')
                if 'dataset' in info[0]:
                    dataset = info[1]
                    
        self.dataset = dataset
        self.h5_files = h5_files
