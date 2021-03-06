import argparse
import os
import sys
import mmcv
import numpy as np
import pickle
from itertools import repeat
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

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True


def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
    return float(area_A + area_B - interArea)

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def get_iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

def filter_logits_by_gt(bboxes_mat, logits_mat, ground_truths, iou_threshold=0.5):
    """
    Gets bboxes, logits and ground truths per image on dataset.
    Filtering boxes so only bboxes with highest iou remains, eventually bboxes number equals ground truths number.
    Args: bboxes logits: list of dicts. each dict contains image id, bbox, score, logits.
          ground_truths: a dict that contains image_id, bboxes, labels
    """
    results_per_class = []
    gt_bboxes = ground_truths['bboxes']
    gt_labels = ground_truths['labels']
    classes = len(logits_mat[0])
    for c in range(classes):
        filtered_pred_bbox = []
        filtered_pred_logits = []
        current_gt_bboxes = []
        # include only gt boxes fit to current class c
        [current_gt_bboxes.append(g) for i, g in enumerate(gt_bboxes) if gt_labels[i] == c]  # include only gts from class c
        idx_pred_matched = np.zeros(len(bboxes_mat))
        for gt_bbox in current_gt_bboxes:
            iouMax = sys.float_info.min
            # for each box among gt boxes, check which prediction has best iou with it.
            # after the loop save the bbox and logits who fits the gt
            for j, pred in enumerate(bboxes_mat):
                iou = get_iou(gt_bbox, pred)
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            if iouMax >= iou_threshold:
                if idx_pred_matched[jmax] == 0:
                    idx_pred_matched[jmax] = 1  # flag as already seen
                    filtered_pred_bbox.append(bboxes_mat[jmax])
                    filtered_pred_logits.append(logits_mat[jmax])
        results_per_class.append((filtered_pred_bbox, filtered_pred_logits))
    return results_per_class

def logits2output(logits, boxes, img_ids, targets):
    """
    input: lists of logits (num of predictions, num of classes), bboxes (num of predictions, 4) and corresponding img ids.
    (the input are the files saved by 'extract_logits_and_targets' function).
    the function rearranges them to be a list of img ids, each cell is a list of 1230.
    each cell of that list is an array (x, 5) includes bbox coordinates and score for each detection of that class.
    output: a list that suitable for evaluation by lvis.eval()
    """
    # imgs_num = len(set(img_ids))
    outputs = np.empty((5000, 1230, 0)).tolist()
    logits = logits.numpy()
    for i in range(len(logits)):
        print(f"{i}/{len(logits)}")
        img_id = img_ids[i]
        label = np.argmax(logits[i][1:])  # prediction label
        score = np.max(logits[i][1:])
        bbox = boxes[i]
        # target = targets[i]  # gt target
        pred = np.empty(5)
        pred[:4] = bbox
        pred[4] = score
        # pred[5] = label
        outputs[img_id][label].append(pred)

    for i in range(len(outputs)):
        print(f'converting img {i} predictions to ndarray')
        for j in range(len(outputs[i])):
            outputs[i][j] = np.array(outputs[i][j])

    return outputs


def logits2output_unfiltered(logits, boxes, labels):
    """
    input: lists of logits (num of predictions, num of classes), bboxes (num of predictions, 4),
    where each image has 300 predictions.
    the function rearranges them to be a list of img ids, each cell is a list of 1230.
    each cell of that list is an array (x, 5) includes bbox coordinates and score for each detection of that class.
    output: a list that suitable for evaluation by lvis.eval()
    """
    # imgs_num = len(set(img_ids))
    outputs = np.empty((5000, 1230, 0)).tolist()
    # logits = logits.numpy()
    for i in range(logits.shape[0]):
        print(f"image {i}/{logits.shape[0]}")
        for j in range(logits.shape[1]):
            # img_id = i % 300
            # calc labels using argmax
            label = np.argmax(logits[i][j][1:])  # prediction label
            score = np.max(logits[i][j][1:])
            # use original labels columns from the model
            # label = int(labels[i][j])
            # score = logits[i][j][label+1]   # adding 1 because labels where saved without background
            bbox = boxes[i][j]
            pred = np.empty(5)
            pred[:4] = bbox
            pred[4] = score
            # pred[5] = label
            outputs[i][label].append(pred)

    for i in range(len(outputs)):
        print(f'converting img {i} predictions to ndarray')
        for j in range(len(outputs[i])):
            outputs[i][j] = np.array(outputs[i][j])

    return outputs


# def main():
#     args = parse_args()
#     cfg = mmcv.Config.fromfile(args.config)
#     dataset = build_dataset(cfg.data.test)
#     data_loader = build_dataloader(
#         dataset,
#         imgs_per_gpu=1,
#         workers_per_gpu=0,  # cfg.data.workers_per_gpu
#         dist=False,
#         shuffle=False)
    # bboxes_logits = pickle.load(open('../logits.p', 'rb'))
    # gt_list = []
    # for i, data in enumerate(data_loader):  # original script in test_lvis_tnorm.py
    #     print(i)
    #     img_id = dataset.img_infos[i]['id']
    #     gt = dataset.get_ann_info(i)
    #     gt_dict = dict()
    #     gt_dict['id'] = img_id
    #     gt_dict['bboxes'] = gt['bboxes']
    #     gt_dict['labels'] = gt['labels']
    #     gt_list.append(gt_dict)
    #     # filter logits according to equivalent ground truth
    #     results = filter_logits_by_gt(bboxes_logits[i], gt_list[i])
    # with open('../ground_truths.p', 'wb') as outfile:
    #     pickle.dump(gt_list, outfile)
    # bboxes_logits = pickle.load(open('../bboxes_logits.p', 'rb'))
    # print('hi')

#
#
# if __name__ == '__main__':
#     main()
