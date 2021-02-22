import yaml
import os
from Models.VBB_IMG_v1.VBB_RGB import VBB_RGB
from train.training import restore_gt_numpy, _build_target
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataReader.detectionDataset import DetectionDataset
from dataReader.dataLoader import DataLoader
from torch.autograd import Variable
from utils.cython_bbox import anchor_intersections
from sklearn.metrics import accuracy_score


def _process_batch(data):
    bbox_pred_np, gt_boxes, gt_classes, class_pred_np_b, valid_classes_np_b, iou_pred_np, inp_size, cfg = data

    if cfg['arch'] == 'VBB_7c':
        cell_size = 8
        out_size = inp_size / 8
    else:
        print('Unknown cfg arch')

    cell_w, cell_h = cell_size, cell_size

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape

    # gt
    _classes = np.zeros([hw, num_anchors, cfg['num_classes']], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    gt_classes = np.asarray(gt_classes)

    # scale pred_bbox
    anchors = np.ascontiguousarray(cfg['anchors'], dtype=np.float) * 2

    # for each cell, compare predicted_bbox and gt_bbox
    # bbox_np_b = np.reshape(bbox_np, [-1, 4])
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    gt_boxes_b = np.delete(gt_boxes_b, np.where(gt_classes == -1), axis=0)
    gt_classes = np.delete(gt_classes, np.where(gt_classes == -1))

    # locate the cell of each gt_boxes conda
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * out_size[0] + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    # for each gt boxes, match the best anchor
    # gt_boxes_resize = [(xmin, ymin, xmax, ymax)] unit: cell px
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] /= cell_w
    gt_boxes_resize[:, 1::2] /= cell_h

    anchor_ious = anchor_intersections(anchors, np.ascontiguousarray(gt_boxes_resize, dtype=np.float))
    # eadan: add more anchors (iou>thresh) not just max
    anchor_inds = np.argmax(anchor_ious, axis=0)

    # find pred conf for current class
    valid_ind = np.where(gt_classes >= 0)[0]
    gt_classes = gt_classes[valid_ind]
    anchor_inds = anchor_inds[valid_ind]
    cell_inds = cell_inds[valid_ind]
    confs = class_pred_np_b[cell_inds, anchor_inds, :]

    bg_cells = np.array(list(set(range(0,_class_mask.shape[0])) - set(cell_inds)))
    count = len(cell_inds)
    if count == 0:
        count = 5
    bg_cells_idx = np.random.choice(bg_cells, count,replace=False)
    bg_anchors_idx = np.random.choice(range(0,_class_mask.shape[1]), len(bg_cells_idx), replace=True)



    gt_classes = np.concatenate([gt_classes, np.array([9]*len(bg_cells_idx))])
    confs = np.concatenate([confs, class_pred_np_b[bg_cells_idx, bg_anchors_idx, :]])
    #preds = np.argmax(class_pred_np_b[cell_inds, anchor_inds,:],axis=-1)
    return gt_classes, confs


def save_confs_and_labels(trained_model, cfg):
    # load dataset
    batch_size = 7
    num_workers = 10
    dataset = DetectionDataset(cfg, mode='val')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    # classes = cfg["label_names"]
    # sorted_classes_by_nsamples = {k: v for k, v in
    #                               sorted(dataset.class_hist.items(), key=lambda item: item[1], reverse=True)}
    # load model
    net = VBB_RGB(cfg)
    net.load_net(trained_model)
    net.eval()
    net.cuda()
    print(trained_model)
    print('load model successfully')

    total_confs = None
    total_labels = None

    for step, data in enumerate(dataloader):
        print(step) if step % 10 == 0 else None
        images, labels, valid_classes = data
        im_data = Variable(images.cuda())

        bbox_pred, iou_pred, class_pred = net.forward(im_data, rgb2bgr=False)
        class_pred = net.tmp_class_logits

        network_size_wh = np.array([int(im_data.data.size()[3]), int(im_data.data.size()[2])])

        gt_boxes, gt_classes = restore_gt_numpy(labels)

        bbox_pred_np = bbox_pred.data.cpu().numpy()
        iou_pred_np = iou_pred.data.cpu().numpy()
        class_pred_np = class_pred.data.cpu().numpy()
        valid_classes_np = valid_classes.cpu().numpy()

        data = [(bbox_pred_np[b], gt_boxes[b], gt_classes[b], class_pred_np[b], valid_classes_np[b], iou_pred_np[b],
                 network_size_wh, cfg) for b in range(images.shape[0])]

        values_to_save = list(map(_process_batch, data))
        curr_labels = np.concatenate([values_to_save[i][0] for i in range(images.shape[0])])
        curr_confs = np.concatenate([values_to_save[i][1] for i in range(images.shape[0])])
        if total_confs is None or total_labels is None:
            total_labels = curr_labels
            total_confs = curr_confs
        else:
            total_labels = np.concatenate([total_labels, curr_labels])
            total_confs = np.vstack([total_confs, curr_confs])
    # save confs and labels
    np.savez("/root/tmp_results/model_output/val_confs.npz",total_confs)
    np.savez("/root/tmp_results/model_output/val_labels.npz",total_labels)


if __name__ == "__main__":

    exp = 'VBB_IMG_v1'
    model_path = '/root/Models/' + exp
    exp_yaml = model_path + '/' + exp + '.yaml'

    try:
        cfg = yaml.load(open(exp_yaml, 'r'))
    except Exception:
        print('Error: cannot parse cfg', exp_yaml)
        raise Exception

    gpu_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # trained model
    # epoch = 60
    # trained_model = '/data/VBB_Weights/VBB_IMG_v1/trainedModels/' + exp + '/' \
    #                 + exp + '_' + str(epoch) + '.h5'
    trained_model = '/data/VBB_Weights/VBB_IMG_v1/trainedModels/VBB_IMG_v1/VBB_IMG_v1_60.h5'
    save_confs_and_labels(trained_model, cfg)

