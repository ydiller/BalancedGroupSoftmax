import torch
import pickle
import numpy as np
from mmdet.ops.nms import nms_wrapper
from collections import Counter

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   dragon_logits=None,
                   filename=None,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # if dragon logits exist, load them instead of original logits
    # if dragon_logits is not None:
    #     multi_scores2 = torch.from_numpy(pickle.load(open(f'{dragon_logits}{filename}.p', 'rb')))
    #     multi_scores2 = multi_scores2.cuda(multi_bboxes.get_device())

    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, indices = nms_op(cls_dets, **nms_cfg_)
        cls_dets = torch.cat([cls_dets, indices[:, None].float()], dim=1)  # adding the indices of the filtered proposals
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    # examine where argmax index changes after dragon fixing
    # diff_max = 0  # count different logit argmax
    # original_argmaxes, dragon_argmaxes = [], []
    # for i in range(multi_scores.shape[0]):
    #     original_logits = multi_scores[i][1:].detach().cpu().numpy()
    #     fixed_logtis = multi_scores2[i][1:].detach().cpu().numpy()
    #     print(np.argmax(original_logits), np.argmax(fixed_logtis))
    #     print(f'max: {np.max(original_logits)} {np.max(fixed_logtis)}')
    #     if np.argmax(original_logits) != np.argmax(fixed_logtis):
    #         diff_max += 1
    #         original_argmaxes.append(np.argmax(original_logits))
    #         dragon_argmaxes.append(np.argmax(fixed_logtis))
    # ordered_classes = list(reversed(list({k: v for k, v in sorted(Counter(dragon_argmaxes).items(), key=lambda item: item[1])}.keys())))
    # ordered_instances = list({k: v for k, v in sorted(Counter(dragon_argmaxes).items(), key=lambda item: item[1], reverse=True)}.values())

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -2].sort(descending=True)  # at original file: bboxes[:, -1]
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)


    #  check for which propoals the argmax result changed due to dragon. and add this proposals to bboxes and labels.
    # for i in range(multi_scores.shape[0]):
    #     original_logits = multi_scores[i][1:].detach().cpu().numpy()
    #     fixed_logtis = multi_scores2[i][1:].detach().cpu().numpy()
    #     if np.argmax(original_logits) != np.argmax(fixed_logtis):
    #         idx = np.argmax(fixed_logtis)
    #         bbox = np.zeros(6)
    #         bbox[:4] = multi_bboxes[i][(idx + 1) * 4:(idx + 2) * 4].cpu().numpy()
    #         bbox[4] = np.max(fixed_logtis)  # get the fixed score
    #         bbox[5] = i
    #         bbox = torch.tensor(bbox).cuda(multi_bboxes.get_device())
    #         bboxes = torch.cat([bboxes, bbox.float()[None, :]])
    #         label = torch.tensor(idx).cuda(multi_bboxes.get_device())[None]
    #         labels = torch.cat([labels, label])

    # go over all proposals and pick the max
    # device = multi_bboxes.get_device()
    # bboxes = multi_bboxes.new_zeros((0, 6)).cuda(device)
    # labels = multi_bboxes.new_zeros((0, ), dtype=torch.long).cuda(device)
    # for i in range(multi_scores.shape[0]):
    #     fixed_logtis = multi_scores2[i][1:].detach().cpu().numpy()
    #     idx = np.argmax(fixed_logtis)
    #     bbox = np.zeros(6)
    #     bbox[:4] = multi_bboxes[i][(idx + 1) * 4:(idx + 2) * 4].cpu().numpy()
    #     bbox[4] = np.max(fixed_logtis)  # get the fixed score
    #     bbox[5] = i
    #     bbox = torch.tensor(bbox).cuda(device)
    #     label = torch.tensor(idx).cuda(device)[None]
    #     bboxes = torch.cat([bboxes, bbox.float()[None, :]])
    #     labels = torch.cat([labels, label])
    #
    # _, inds = bboxes[:, -2].sort(descending=True)
    # inds = inds[:max_num]
    # bboxes = bboxes[inds]
    # labels = labels[inds]


    # go over all proposals and pick the top 300 fixed bboxes
    device = multi_bboxes.get_device()
    bboxes = multi_bboxes.new_zeros((0, 6)).cuda(device)
    labels = multi_bboxes.new_zeros((0, ), dtype=torch.long).cuda(device)
    for i in range(multi_scores.shape[0]):
        fixed_logtis = multi_scores2[i][1:].detach().cpu().numpy()
        idx = np.argmax(fixed_logtis)
        bbox = np.zeros(6)
        bbox[:4] = multi_bboxes[i][(idx + 1) * 4:(idx + 2) * 4].cpu().numpy()
        bbox[4] = np.max(fixed_logtis)  # get the fixed score
        bbox[5] = i
        bbox = torch.tensor(bbox).cuda(device)
        label = torch.tensor(idx).cuda(device)[None]
        bboxes = torch.cat([bboxes, bbox.float()[None, :]])
        labels = torch.cat([labels, label])

    _, inds = bboxes[:, -2].sort(descending=True)
    inds = inds[:max_num]
    bboxes = bboxes[inds]
    labels = labels[inds]


    # check how many bboxes each proposal has in the final bboxes tensor
    # picked_boxes = Counter(bboxes[:, -1].cpu().numpy())
    # ordered_classes = list(reversed(list({k: v for k, v in sorted(Counter(picked_boxes).items(), key=lambda item: item[1])}.keys())))
    # ordered_instances = list({k: v for k, v in sorted(Counter(picked_boxes).items(), key=lambda item: item[1], reverse=True)}.values())

    return bboxes, labels
