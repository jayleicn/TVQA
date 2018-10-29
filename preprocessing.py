__author__ = "Jie Lei"


import os
import sys
import re
import math
import json
import glob
import copy
import pysrt
import numpy as np

from tqdm import tqdm

from utils import read_json_lines, load_json, save_json


def merge_list_dicts(list_dicts):
    z = list_dicts[0].copy()   # start with x's keys and values
    for i in range(1, len(list_dicts)):
        z.update(list_dicts[i])  # modifies z with y's keys and values & returns None
    return z


def get_vidname2cnt_per_show(base_path):
    """ get jpg file count for each sub dirs in the base_path
    the resulting file is a python dict with {subdir_name: count}
    """
    subdirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    vidname2cnt = {}
    for ele in tqdm(subdirs):
        cur_subdir_path = os.path.join(base_path, ele)
        # cur_files = [name for name in os.listdir(cur_subdir_path) 
        #              if os.path.isfile(os.path.join(cur_subdir_path, name))]
        cur_files = glob.glob(os.path.join(cur_subdir_path, "*jpg"))
        vidname2cnt[ele] = len(cur_files)
    return vidname2cnt


def get_vidname2cnt_all(frame_root_path, vidname2cnt_cache_path):
    if os.path.exists(vidname2cnt_cache_path):
        print("Found frame cnt cache, loading ...")
        return load_json(vidname2cnt_cache_path)
    show_names = ["bbt", "friends", "grey", "met", "castle", "house"]
    vidname2cnt_list = []
    for sn in show_names:
        print("Count frames in %s" % sn)
        cur_base_path = os.path.join(frame_root_path, "%s_frames" % sn)
        vidname2cnt_list.append(get_vidname2cnt_per_show(cur_base_path))
    vidname2cnt = merge_list_dicts(vidname2cnt_list)    
    save_json(vidname2cnt, vidname2cnt_cache_path)    
    return 


def clean_str(string):
    """ Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_srt(srt_dir, srt_cache_path):
    """
    return: A python dict, the keys are the video names, the entries are lists,
            each contains all the text from a .srt file
    sub_times are the start time of the sentences.
    """
    if os.path.exists(srt_cache_path):
        print("Found srt data cache, loading ...")
        return load_json(srt_cache_path)

    print("Loading srt files from %s ..." % srt_dir)
    srt_paths = glob.glob(os.path.join(srt_dir, "*.srt"))
    name2sub_text = {}
    name2sub_time = {}
    for i in tqdm(range(len(srt_paths))):
        subs = pysrt.open(srt_paths[i], encoding="iso-8859-1")
        if len(subs) == 0:
            subs = pysrt.open(srt_paths[i])

        text_list = []
        sub_time_list = []
        for j in range(len(subs)):
            cur_sub = subs[j]
            cur_str = cur_sub.text
            cur_str = "(<UNKNAME>:)" + cur_str if cur_str[0] != "(" else cur_str
            cur_str = cur_str.replace("\n", " ")
            text_list.append(cur_str)
            sub_time_list.append(
                60 * cur_sub.start.minutes + cur_sub.start.seconds + 0.001 * cur_sub.start.milliseconds)

        key_str = os.path.splitext(os.path.basename(srt_paths[i]))[0]
        name2sub_text[key_str] = text_list
        name2sub_time[key_str] = sub_time_list
    srt_data = {"sub_text": name2sub_text, "sub_time": name2sub_time}
    save_json(srt_data, srt_cache_path)
    return srt_data


def convert_ts(ts):
    """ 26.2-34.4  -->  [26.2, 34.4] ,
    also replace any NaN value with [10, 30], a simple replacement, will fix later"""
    new_ts = [float(ele) for ele in ts.split("-")]
    is_nan = False
    if math.isnan(new_ts[0]) or math.isnan(new_ts[1]):
        new_ts = [10, 30]  #
        is_nan = True
    return new_ts, is_nan


def interval2frame(interval, num_frame, fps=3):
    """ downsample to 300 frame max,
    :param interval: e.g. [26.2, 34.4]
    :param num_frame: number of frame for this clip
    :param fps: number of frames used per second
    :return:
    """
    # 0.0356 of the video has more than 300 frames, for those, downsample to 300.
    max_num_frame = 300.
    if num_frame > max_num_frame:
        frame_start_end = [(max_num_frame / num_frame) * fps * ele for ele in interval]
    else:
        frame_start_end = [fps * ele for ele in interval]

    frame_start_end = np.asarray([frame_start_end[0] - fps, frame_start_end[1] + fps])
    frame_start_end = np.floor(np.clip(frame_start_end, 0, 300))
    if frame_start_end[0] == frame_start_end[1]:
        frame_start_end[0] = max(0, frame_start_end[0] - 3)
    frame_start_end = [int(x) for x in frame_start_end]
    return frame_start_end


def tokenize_qa(data_dicts):
    """tokenize the text in QAs"""
    tokenized_data_dicts = []
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4"]
    all_keys = data_dicts[0].keys()
    print("Tokenize QA ...")
    for ele in tqdm(data_dicts):
        tmp_dict = {}
        for k in all_keys:
            if k in text_keys:
                tmp_dict[k] = clean_str(ele[k])
            else:
                tmp_dict[k] = ele[k]
        tokenized_data_dicts.append(tmp_dict)
    return tokenized_data_dicts


def tokenize_srt(srt_data):
    """tokenize the text in srt"""
    tokenized_srt_data = {"sub_text": {}, "sub_time": srt_data["sub_time"]}
    print("Tokenize subtitle ...")
    for k in tqdm(srt_data["sub_text"].keys()):
        tokenized_srt_data["sub_text"][k] = [clean_str(s) for s in srt_data["sub_text"][k]]
    return tokenized_srt_data


def add_srt(raw_data_dicts, srt_data, eos_token="<eos>"):
    """ add entries 'sub_time', 'sub_text' """
    data_dicts = copy.deepcopy(raw_data_dicts)
    eos_token = " %s " % eos_token  # add space around
    print("Adding subtitle ...")
    for i in tqdm(range(len(data_dicts))):
        vid_name = data_dicts[i]["vid_name"]
        data_dicts[i]["sub_text"] = eos_token.join(srt_data["sub_text"][vid_name])
        data_dicts[i]["sub_time"] = srt_data["sub_time"][vid_name]
    return data_dicts


def find_nearest(array, value):
    """closet value in an array to a given value"""
    idx = (np.abs(array-value)).argmin()
    return idx  # array[idx]


def get_located_sub_text(ts, sub_text_list, sub_time, eos_token="<eos>"):
    """return the located subtitle text according to the timestep annotation
    :param ts: (list) e.g. [26.2, 34.4]
    :param sub_text_list: (list) each element is a subtitle sentence
    :param sub_time: (list) each element is a float number indicates the start time of a subtitle sentence
    """
    located_indices = []
    for idx in range(len(sub_time)):
        if ts[0] < sub_time[idx] < ts[1]:
            located_indices.append(idx)

    # deal with 0-length: use three sub sentences most close to START
    if len(located_indices) == 0:
        closest_1 = find_nearest(np.asarray(sub_time), ts[0])
        located_indices.extend([closest_1 - 1, closest_1, closest_1 + 1])

    # rm the indices larger than length of sub_text_list
    located_indices = [located_indices[i] for i in range(len(located_indices))
                       if located_indices[i] <= len(sub_text_list) - 1]

    # TODO is this necessary
    # add the one before the first located ts, no need to do it for the last one
    # if 0 not in located_indices:
    #     located_indices = [located_indices[0] - 1] + located_indices
    eos_token = " %s " % eos_token
    located_sub_text = eos_token.join([sub_text_list[idx] for idx in located_indices])
    return located_sub_text


def add_located(raw_data_dicts, srt_data, frame_cnt):
    """ add entries 'located_frame', 'located_sub_text' """
    data_dicts = copy.deepcopy(raw_data_dicts)
    nan_cnt = 0
    for i in tqdm(range(len(data_dicts))):
        vid_name = data_dicts[i]["vid_name"]
        sub_text_list = srt_data["sub_text"][vid_name]
        sub_time = srt_data["sub_time"][vid_name]
        ts, is_nan = convert_ts(data_dicts[i]["ts"])
        nan_cnt += is_nan
        data_dicts[i]["ts"] = ts
        data_dicts[i]["located_frame"] = interval2frame(ts, frame_cnt[vid_name])
        data_dicts[i]["located_sub_text"] = get_located_sub_text(ts, sub_text_list, sub_time)
    print("There are %d NaN values in ts, which are replaced by [10, 30], will be fixed later" % nan_cnt)
    return data_dicts


def process_qa(qa_path, srt_dir, frame_base_path, srt_cache_path, frame_cnt_cache_path, save_path):
    srt_data = load_srt(srt_dir, srt_cache_path)
    srt_data = tokenize_srt(srt_data)
    qa_data = read_json_lines(qa_path)
    qa_data = tokenize_qa(qa_data)
    qa_srt_data = add_srt(qa_data, srt_data, eos_token="<eos>")
    frame_cnt_dict = get_vidname2cnt_all(frame_base_path, frame_cnt_cache_path)
    qa_srt_located_data = add_located(qa_srt_data, srt_data, frame_cnt_dict)
    save_json(qa_srt_located_data, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="data dir path")
    parser.add_argument("--frm_dir", type=str, help="video frame dir path, the program will use cache if it exists")
    args = parser.parse_args()

    data_dir = args.data_dir
    sub_dir = os.path.join(data_dir, "tvqa_subtitles")
    raw_qa_files = glob.glob(os.path.join(data_dir, "tvqa_qa_release", "*jsonl"))
    sub_cache_path = os.path.join(data_dir, "srt_data_cache.json")
    frm_cnt_cache_path = os.path.join(data_dir, "frm_cnt_cache.json")

    for i, qa_file in enumerate(raw_qa_files):
        print("-"*60)
        print("Process %s" % qa_file)
        processed_qa_path = os.path.join(data_dir, os.path.split(qa_file)[1].replace(".jsonl", "_processed.json"))
        process_qa(qa_file, sub_dir, args.frm_dir, sub_cache_path, frm_cnt_cache_path, processed_qa_path)
