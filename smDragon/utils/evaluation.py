import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def calculate_avg_cong_and_acc_per_class(total_labels, total_confs, total_preds, classes, sorted_classes_by_nsamples):
    avg_conf_per_class = []
    acc_per_class = []
    class_names = []
    for class_name in sorted_classes_by_nsamples:
        class_id = classes.index(class_name)
        samples_ind = np.where(total_labels == class_id)[0]
        if len(samples_ind) == 0:
            continue
        avg_conf_per_class += [np.average(total_confs[samples_ind])]
        acc_per_class += [accuracy_score(total_labels[samples_ind], total_preds[samples_ind])]
        class_names += [class_name]
    per_class_acc = sum(acc_per_class) * (1 / len(acc_per_class))
    return class_names, avg_conf_per_class, acc_per_class, per_class_acc


def calculate_avg_conf_and_acc_per_class(targets, confs, preds, ordered_classes):
    avg_conf_per_class = []
    acc_per_class = []
    for class_id in ordered_classes:
        samples_ind = np.where(targets == class_id)[0]
        if len(samples_ind) == 0:
            continue
        avg_conf_per_class += [np.average(confs[samples_ind])]
        acc_per_class += [accuracy_score(targets[samples_ind], preds[samples_ind])]
    per_class_acc = sum(acc_per_class) * (1 / len(acc_per_class))
    return avg_conf_per_class, acc_per_class, per_class_acc

def plot_avg_conf(sorted_classes, avg_conf):
    plt.plot(sorted_classes, avg_conf)
    plt.scatter(sorted_classes, avg_conf)
    plt.ylabel("average confidence")
    plt.xlabel("classes sorted by # samples")
    plt.xticks(rotation=70)

def plot_accuracy_per_class(sorted_classes, acc_pc):
    plt.plot(sorted_classes, acc_pc)
    plt.scatter(sorted_classes, acc_pc)
    plt.ylabel("average accuracy")
    plt.xlabel("classes sorted by # samples")
    plt.xticks(rotation=70)