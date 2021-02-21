import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib
import torch
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score


def extract_logits_and_targets(bboxes_logits):
    """
    creates matrix for logits (num of bboxes, num of classes) and array for gt (num of boxes, 1)
    """
    logits = torch.empty((0, 1231), dtype=torch.double)
    targets = []
    for i, img in enumerate(bboxes_logits):
        print(f"image number {i}/{len(bboxes_logits)}")
        for j, c_tup in enumerate(img):  # for each tuple in the list of classes
            if len(c_tup[0]) != 0:
                logits = torch.cat((logits, torch.tensor(c_tup[1]).softmax(1)))
                targets += [j for _ in range(len(c_tup[1]))]
    return logits, targets


def plot_gt_instances(oredered_instances):
    x_axis = range(len(oredered_instances))
    plt.plot(x_axis, oredered_instances, linewidth=1.0)
    plt.xlabel("classes sorted by # samples")
    plt.ylabel("ground truth instances")
    plt.xticks([1, 250, 500, 750, 1000], [1, 250, 500, 750, 1000])
    plt.show()


def plot_avg_confidence(classes_order, avg_conf):
    x_axis = range(len(classes_order))
    plt.plot(x_axis, avg_conf, linewidth=1.0)
    # fit poly
    plt.plot(x_axis, np.poly1d(np.polyfit(x_axis, avg_conf, 5))(x_axis), linewidth=1.0)
    plt.xlabel("classes sorted by # samples")
    plt.ylabel("average confidence")
    plt.xticks([1, 250, 500, 750, 1000], [1, 250, 500, 750, 1000])
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
    plt.show()


def plot_avg_acc(classes_order, avg_acc):
    x_axis = range(len(classes_order))
    #plt.figure()
    plt.plot(range(len(classes_order)), avg_acc, linewidth=1.0)
    # plt.plot(x_axis, np.poly1d(np.polyfit(x_axis, avg_acc, 5))(x_axis), linewidth=3.0)
    plt.xlabel("classes sorted by # samples")
    plt.ylabel("accuracy per class")
    plt.xticks([1, 250, 500, 750, 1000], [1, 250, 500, 750, 1000])
    #plt.yticks([0,25,50,75,100],[0.0,0.25,0.5,0.75,1.0])
    plt.show()


def avg_per_class(arr, targets, ordered_classes):
    arr_avg_per_class = []
    for c in ordered_classes:
        arr_avg_per_class = arr_avg_per_class + [np.average(arr[np.where(targets == c)[0], c])]
    return arr_avg_per_class


def calculate_avg_conf_and_acc_per_class(targets, confs, preds, ordered_classes):
    avg_conf_per_class = []
    acc_per_class = []
    for class_id in ordered_classes:
        samples_ind = np.where(targets == class_id)[0]
        if len(samples_ind) == 0:
            continue
        avg_conf_per_class += [np.average(confs[samples_ind, class_id])]
        acc_per_class += [accuracy_score(targets[samples_ind], preds[samples_ind])]
    per_class_acc = sum(acc_per_class) * (1 / len(acc_per_class))
    return avg_conf_per_class, acc_per_class, per_class_acc


def main():
    # load bboxs and logits, process and save
    # bboxes_logits = pickle.load(open('../bboxes_logits.p', 'rb'))
    # logits, targets = extract_logits_and_targets(bboxes_logits)
    # with open('../filtered_logits.p', 'wb') as outfile:
    #     pickle.dump(logits, outfile)
    # with open('../filtered_targets.p', 'wb') as outfile:
    #     pickle.dump(targets, outfile)

    logits = pickle.load(open('../filtered_logits.p', 'rb'))
    targets = pickle.load(open('../filtered_targets.p', 'rb'))
    ordered_classes = list(reversed(list({k: v for k, v in sorted(Counter(targets).items(), key=lambda item: item[1])}.keys())))
    ordered_instances = list({k: v for k, v in sorted(Counter(targets).items(), key=lambda item: item[1], reverse=True)}.values())
    # calculate avg confidence and avg accuracy
    # arr_avg_per_class = avg_per_class(logits, np.array(targets), ordered_classes)
    avg_conf_per_class, acc_per_class, _ = calculate_avg_conf_and_acc_per_class(np.array(targets), logits, np.argmax(logits, axis=1), ordered_classes)

    # plot gt instances in decreasing order
    plot_gt_instances(ordered_instances)
    # plot avg conf per class
    plot_avg_confidence(ordered_classes, np.array(avg_conf_per_class))
    # plot avg acc per class
    plot_avg_acc(ordered_classes, acc_per_class)
    # cm = confusion_matrix(targets, np.argmax(logits, axis=1), labels=ordered_classes)
    # cm_normed = cm.astype(np.float32) / cm.sum(axis=1) * 100
    # avg_acc = np.diag(cm_normed)

if __name__ == '__main__':
    main()
