import os, sys
import glob


def load_file_protocol(file_path):
    protocol_data = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            protocol_data.append(line.strip().split(','))   # [label, video_id]
            line = f.readline()
        # print('protocol_data:', protocol_data)
        # sys.exit(0)
        
        label_real = '+1'
        label_spoof = '-1'
        for i in range(len(protocol_data)):
            protocol_data[i][0] = 1 if protocol_data[i][0] == label_real else 0
        # print('protocol_data:', protocol_data)
        # sys.exit(0)
        return protocol_data


def find_files(directory, extension, sort=True):
    matching_files = []

    def search_recursively(folder):
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(extension):
                    matching_files.append(os.path.join(root, file))
    search_recursively(directory)
    if sort:
        matching_files.sort()
    return matching_files


def find_neighbor_file(path_file, neighbor_ext):
    path_dir = os.path.dirname(path_file)
    neighbor_path = glob.glob(path_dir + '/*' + neighbor_ext)
    if len(neighbor_path) == 0:
        raise Exception(f'Error, no file \'*{neighbor_ext}\' found in dir \'{path_dir}\'')
    return neighbor_path[0]


def make_samples_list(protocol_data, frames_path_part, rgb_file_ext):
    samples_list = [None] * len(protocol_data)
    for i, (label, video_name) in enumerate(protocol_data):
        # print('label:', label, '    video_name:', video_name)
        rgb_file_pattern = os.path.join(frames_path_part, video_name+'*'+rgb_file_ext)
        rgb_file_path = glob.glob(rgb_file_pattern)
        if len(rgb_file_path) == 0:
            raise Exception(f'Error, no file \'{rgb_file_pattern}\' found in dir \'{frames_path_part}\'')
        rgb_file_path = rgb_file_path[0]
        # print('rgb_file_path:', rgb_file_path)
        
        '''
        pc_file_pattern = os.path.join(frames_path_part, video_name+'*', '*'+pc_file_ext)
        pc_file_path = glob.glob(pc_file_pattern)
        if len(pc_file_path) == 0:
            raise Exception(f'Error, no file \'{pc_file_path}\' found in dir \'{frames_path_part}\'')
        pc_file_path = pc_file_path[0]
        # print('pc_file_path:', pc_file_path)
        '''

        samples_list[i] = (rgb_file_path, label)
        # print('------------------')
        # sys.exit(0)
    return samples_list