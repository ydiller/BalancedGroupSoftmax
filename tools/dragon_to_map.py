import pickle
import mmcv
import numpy as np
from utils import logits2output, logits2output_unfiltered
from test_lvis import parse_args
from mmdet.core import lvis_eval, results2json
from mmdet.datasets import build_dataloader, build_dataset
import pandas as pd

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    eval_types = args.eval
    # load filtered detections (filterd by iou)
    # logits = pickle.load(open('filtered_logits.p', 'rb'))
    # targets = pickle.load(open('filtered_targets.p', 'rb'))
    # imgs = pickle.load(open('filtered_imgs.p', 'rb'))
    # boxes = pickle.load(open('filtered_boxes.p', 'rb'))
    # outputs = logits2output(logits, boxes, imgs, targets)

    # load unfiltered files (300 detections for each image)
    logits = []
    logits += [pickle.load(open(f'test_logits/smDragon/dragon_logits_mat{i}.p', 'rb')) for i in range(1, 6)]
    logits = np.asarray(logits)
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])  # reshape to (5000, 300, 1231)
    boxes = pickle.load(open('test_logits/bboxes_mat.p', 'rb'))
    labels = pickle.load(open('test_logits/labels_mat.p', 'rb'))
    outputs = logits2output_unfiltered(logits, boxes, labels)
    # otp2 = np.asarray(outputs[:4])  # temp
    # df = pd.DataFrame(otp2)
    # df.to_csv('dragon_otp.csv', index=False)
    # pd.DataFrame(outputs[:50]).to_csv("dragon2map_outputs.csv")
    result_files = results2json(dataset, outputs, args.out)
    lvis_eval(result_files, eval_types, dataset.lvis)


if __name__ == '__main__':
    main()
