import json
import pickle
import os


def read_pickle(pkl_path):
    '''
    Loading a Pickle file.

    Parameters
    ----------
    pkl_path : str
        Absolute path to the .pkl file.
        e.g, /mnt/project/summary.pickle

    '''
    with open(pkl_path, 'rb') as handle:
        pickle_dict = pickle.load(handle)
        
    return pickle_dict


def read_json(json_path):
    '''
    Loading a json file.

    Parameters
    ----------
    json_path : str
        Absolute path to the json file.
        e.g, /mnt/project/summary.json

    '''
    with open(json_path, 'rb') as handle:
        parsed_json = json.load(handle)
        
    return parsed_json


def write_pickle(pkl_path, my_dict):
    '''
    Writing to Pickle file

    Parameters
    ----------
    pkl_path : str
        Absolute path to  .pkl file.
        e.g, /mnt/project/summary.pickle
    my_dict : Dictionary (!?...)

    '''
    with open(pkl_path, 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return None


def write_json(json_path, config):
    '''
    Loading a json file.

    Parameters
    ----------
    json_path : str
        Absolute path to the json file.
        e.g, /mnt/project/summary.json
    config : Dictionary (!?...)

    '''
    
    with open(json_path, 'w') as handle:
        json.dump(config, handle, indent = 4)
        
    return None

def copy_plans_json(src_path,dst_path,n_case):
    """
    Reads a source JSON, updates the number of training cases, and writes it to a new destination.
    The output filename will be 'dataset.json' inside the dst_path directory.
    """
    src_json = read_json(src_path)
    src_json["numTraining"] = n_case
    json_output_path = os.path.join(dst_path, "dataset.json")
    write_json(json_output_path, src_json)
    return None
