__author__ = "Jie Lei"

import os
import json
import cPickle as pickle


def read_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path):
    with open(data_path, "w") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "r") as f:
        return pickle.load(f)


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

