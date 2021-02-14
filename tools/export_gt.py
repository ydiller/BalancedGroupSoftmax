import argparse
import os
import json
import mmcv
import pickle
from mmdet.core import lvis_eval
from mmdet.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,  # cfg.data.workers_per_gpu
        dist=False,
        shuffle=False)
    gt_list = []
    for i, data in enumerate(data_loader):  # original script in test_lvis_tnorm.py
        print(i)
        img_id = dataset.img_infos[i]['id']
        gt = dataset.get_ann_info(i)
        gt_dict = dict()
        gt_dict['id'] = img_id
        gt_dict['bboxes'] = gt['bboxes']
        gt_dict['labels'] = gt['labels']
        gt_list.append(gt_dict)

    with open('../ground_truths.p', 'wb') as outfile:
        pickle.dump(gt_list, outfile)

if __name__ == '__main__':
    main()
