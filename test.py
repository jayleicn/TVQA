__author__ = "Jie Lei"

import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model.tvqa_abc import ABC
from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs
from config import TestOptions
from utils import merge_two_dicts, save_json


def test(opt, dset, model):
    dset.set_mode(opt.mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    qid2preds = {}
    qid2targets = {}
    for valid_idx, batch in tqdm(enumerate(valid_loader)):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        outputs = model(*model_inputs)
        pred_ids = outputs.data.max(1)[1].cpu().numpy().tolist()
        cur_qid2preds = {qid: pred for qid, pred in zip(qids, pred_ids)}
        qid2preds = merge_two_dicts(qid2preds, cur_qid2preds)
        cur_qid2targets = {qid:  target for qid, target in zip(qids, targets)}
        qid2targets = merge_two_dicts(qid2targets, cur_qid2targets)
    return qid2preds, qid2targets


def get_acc_from_qid_dicts(qid2preds, qid2targets):
    qids = qid2preds.keys()
    preds = np.asarray([int(qid2preds[ele]) for ele in qids])
    targets = np.asarray([int(qid2targets[ele]) for ele in qids])
    acc = sum(preds == targets) / float(len(preds))
    return acc


if __name__ == "__main__":
    opt = TestOptions().parse()
    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    model = ABC(opt)

    model.to(opt.device)
    cudnn.benchmark = True
    model_path = os.path.join("results", opt.model_dir, "best_valid.pth")
    model.load_state_dict(torch.load(model_path))

    all_qid2preds, all_qid2targets = test(opt, dset, model)

    if opt.mode == "valid":
        accuracy = get_acc_from_qid_dicts(all_qid2preds, all_qid2targets)
        print("In valid mode, accuracy is %.4f" % accuracy)

    save_path = os.path.join("results", opt.model_dir, "qid2preds_%s.json" % opt.mode)
    save_json(all_qid2preds, save_path)
