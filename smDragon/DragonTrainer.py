import sys
import os
import json
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from keras import optimizers, callbacks
from keras.models import Model
import tensorflow as tf
from sklearn.metrics import accuracy_score
from smDragon.utils.ml_utils_lago import data_to_pkl
from smDragon.arg_parser import UserArgs, ArgParser


class DragonTrainer(object):

    def __init__(self, model_name, ext):
        self.model_name = model_name
        base_train_dir = UserArgs.base_train_dir
        self.training_dir_wo_ext = os.path.join(
            base_train_dir,
            model_name)
        self.training_dir = os.path.join(
            base_train_dir,
            model_name + ext)
        if UserArgs.test_mode:
            self.training_dir = os.path.join(self.training_dir, "test")
            self.training_dir_wo_ext = os.path.join(self.training_dir_wo_ext, "test")

    def create_training_dir(self):
        # check if directory already exists
        if os.path.exists(self.training_dir):
            print(f"Training dir {self.training_dir} already exists..")
            if os.path.exists(os.path.join(self.training_dir, "best-checkpoint")):
                print("Found pretrained model")
                return False
            else:
                raise Exception(f"Training dir {self.training_dir} already exists.. "
                                f"No pretrained model found...")
        print(f"Current training directory for this run: {self.training_dir}")
        os.makedirs(self.training_dir)
        # save current hyper params to training dir
        ArgParser.save_to_file(UserArgs, self.training_dir, self.model_name)
        return True

    @staticmethod
    def _init_optimizer(optimizer, lr):
        opt_name = optimizer.lower()
        if opt_name == 'adam':
            optimizer = optimizers.Adam(lr=lr)
        elif opt_name == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=lr)
        elif opt_name == 'sgd':
            optimizer = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError('unknown optimizer %s' % opt_name)
        return optimizer

    @staticmethod
    def subset_accuracy(y_gt, y_prediction, subset_indices):
        y_prediction = tf.transpose(tf.gather(tf.transpose(y_prediction), subset_indices))
        arg_p = tf.gather(subset_indices, tf.arg_max(y_prediction, 1))
        y_gt = tf.transpose(tf.gather(tf.transpose(y_gt), subset_indices))
        arg_y = tf.gather(subset_indices, tf.arg_max(y_gt, 1))
        return tf.reduce_mean(tf.to_float(tf.equal(arg_y, arg_p)))

    @staticmethod
    def split_by_distribution(Y_true, Y_pred, classes_idx, n_samples):
        dist_indexes = np.array([], dtype=int)
        for class_idx, n_sample in zip(classes_idx, n_samples):
            all_class_indexes = np.where(Y_true == class_idx)[0]
            curr_n_choice = n_sample
            if len(all_class_indexes) < n_sample:
                curr_n_choice = len(all_class_indexes)
            if curr_n_choice == 0:
                # in case no data indexes where found for current class idx
                dist_indexes = np.append(dist_indexes, np.array([], dtype=int))
            else:
                dist_indexes = np.append(dist_indexes,
                                         np.random.choice(all_class_indexes, curr_n_choice, replace=False))
        Y_pred_dist = Y_pred[dist_indexes]
        Y_true_dist = Y_true[dist_indexes]
        return Y_true_dist, Y_pred_dist

    @staticmethod
    def calc_avgScore(Y_true, Y_pred, train_distribution, eval_n_repeat, eval_seed):
        classes_idx, n_samples = train_distribution
        acc_per_class = []
        weights_per_class = []
        for i, (c, n) in enumerate(zip(classes_idx, n_samples)):
            idx = np.where(Y_true == c)[0]
            if len(idx) != 0:
                acc_per_class = acc_per_class + [sum(Y_true[idx] == Y_pred[idx]) / len(idx)]
                weights_per_class = weights_per_class + [n]
        weights_per_class = (np.array(weights_per_class) / sum(weights_per_class))
        return sum(acc_per_class * weights_per_class)

    @staticmethod
    def avg_sigmoid_output(model_instance, X):
        model = model_instance.get_model()
        if len(list(filter(lambda x: x.name == "Gater_Sigmoid", model.layers))) == 0:
            print("WARN: Gater_Sigmoid does not exist in current model.")
            return {'val_avgSigmoid': -1}
        sigmoid_out_model = Model(inputs=model.inputs,
                                  outputs=model.get_layer(name="Gater_Sigmoid").output)
        return np.average(sigmoid_out_model.predict(X), axis=0)

    # @staticmethod
    # def dragon_evaluation(model_instance, X, Y_true, subset_classes,
    #                       dist_function, eval_n_repeat, eval_max_samples, eval_seed):
    #     # make model prediction on all validation set
    #     preds = model_instance.predict_val_layer(X)
    #     Y_pred = subset_classes[(preds[:, subset_classes]).argmax(axis=1)]
    #     Y_ground_truth = Y_true.argmax(axis=-1)
    #     avg_score = DragonTrainer.calc_avgScore(Y_ground_truth, Y_pred, dist_function,
    #                                             eval_n_repeat, eval_max_samples, eval_seed)
    #     print("Avg Score:", avg_score)
    #     return {'val_avgScore': avg_score}

    # @staticmethod
    # def calc_per_class_acc(Y_true, Y_pred):
    #     # counts_per_class = Counter(Y_true)
    #     counts_per_class = pd.Series(Y_true).value_counts().to_dict()
    #     accuracy = ((Y_pred == Y_true) / np.array(
    #         [counts_per_class[y] for y in Y_true])).sum() / len(counts_per_class)
    #     return accuracy

    @staticmethod
    def calc_per_class_acc(Y_true, Y_pred):
        acc_per_class = []
        for class_id in np.unique(Y_true):
            samples_ind = np.where(Y_true == class_id)[0]
            if len(samples_ind) == 0:
                continue
            acc_per_class += [accuracy_score(Y_true[samples_ind], Y_pred[samples_ind])]
        per_class_acc = sum(acc_per_class) * (1 / len(acc_per_class))
        return per_class_acc

    @staticmethod
    def calc_mean_super_class_acc(Y_true, Y_pred, super_classes, map_to_super_class):
        y_true_super_class = map_to_super_class[Y_true]
        y_preds_super_class = map_to_super_class[Y_pred]
        super_class_acc_ls = []
        for super_class in super_classes:
            inds = np.where(np.asarray(y_true_super_class == super_class))[0]
            if len(inds) == 0:
                super_class_acc_ls += [0]
                continue
            super_class_acc_ls += [accuracy_score(y_true_super_class[inds], y_preds_super_class[inds])]
        print(super_class_acc_ls)
        l = len(list(filter(lambda x: x!=0, super_class_acc_ls)))
        if l == 0:
            return 0
        return sum(super_class_acc_ls)/l
    # @staticmethod
    # def xian_per_class_accuracy(model_instance, X, Y_true, subset_classes):
    #     """ A balanced accuracy metric as in Xian (CVPR 2017). Accuracy is
    #         evaluated individually per class, and then uniformly averaged between
    #         classes.
    #     """
    #     # make model prediction on all validation set
    #     preds = model_instance.predict_val_layer(X)
    #     Y_pred = subset_classes[(preds[:, subset_classes]).argmax(axis=1)]
    #     Y_ground_truth = Y_true.argmax(axis=-1)
    #
    #     accuracy = DragonTrainer.calc_per_class_acc(Y_ground_truth, Y_pred)
    #
    #     print("Per Class Accuracy:", accuracy)
    #     return {'val_perClassAcc': accuracy}

    @staticmethod
    def balance_data_with_sample_weights(Y_labels, add_dummy_class=True):
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(Y_labels),
                                                          Y_labels)
        if add_dummy_class:
            class_weights = np.insert(class_weights, 0, 0)  # add 1 zero so 200 -> 201
        a = []
        b = np.unique(Y_labels)
        for y in Y_labels:
            idx = np.where(b == y)[0]
            if len(idx) != 0:
                a += [class_weights[idx]]
            else:
                a += [0]
        sample_weights = np.array(a).reshape(-1)
        return sample_weights

    @staticmethod
    def harmonic_acc(ms_acc, fs_acc):
        return (2 * (ms_acc * fs_acc)) / (ms_acc + fs_acc)

    @staticmethod
    def training_evaluation(model_instance, X_data, Y_data, classes_subsets, eval_sp_params):
        # gextract classes subsets
        all_classes, ms_classes, fs_classes = classes_subsets
        # Estimate accuracies: regualr accuracy, per class accuracy and dragon avg accuracy
        X, X_many, X_few = X_data
        Y, Y_many, Y_few = Y_data
        # all classes accuracy (generalized accuracy)
        _, _, reg_acc, pc_acc, avg_acc = \
            DragonTrainer.__evaluate(model_instance, X, Y, all_classes, eval_sp_params)
        # ms classes accuracy (generalized many-shot accuracy)
        _, _, ms_reg_acc, ms_pc_acc, ms_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_many, Y_many, all_classes, eval_sp_params)
        # fs classes accuracy (generalized few-shot accuracy)
        _, _, fs_reg_acc, fs_pc_acc, fs_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_few, Y_few, all_classes, eval_sp_params)

        reg_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)
        pc_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)
        avg_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)

        # many among many accuracy
        _, _, ms_ms_reg_acc, ms_ms_pc_acc, ms_ms_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_many, Y_many, ms_classes, eval_sp_params)
        # few among few accuracy
        _, _, fs_fs_reg_acc, fs_fs_pc_acc, fs_fs_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_few, Y_few, fs_classes, eval_sp_params)

        res_df = pd.DataFrame(columns=['reg_acc', 'per_class_acc', 'dragon_acc'])
        res_df.loc["All"] = [reg_acc, pc_acc, avg_acc]
        res_df.loc["MS"] = [ms_reg_acc, ms_pc_acc, ms_avg_acc]
        res_df.loc["FS"] = [fs_reg_acc, fs_pc_acc, fs_avg_acc]
        res_df.loc["Harmonic"] = [reg_harmonic_acc, pc_harmonic_acc, avg_harmonic_acc]
        res_df.loc["MS/MS"] = [ms_ms_reg_acc, ms_ms_pc_acc, ms_ms_avg_acc]
        res_df.loc["FS/FS"] = [fs_fs_reg_acc, fs_fs_pc_acc, fs_fs_avg_acc]

        # Estimate Gater Sigmoid Value
        keras_model = model_instance.get_model()
        if len(list(filter(lambda x: x.name == "Gater_Sigmoid", keras_model.layers))) == 0:
            print("WARN: Gater_Sigmoid does not exist in current model.")
        else:
            all_avg_sigmoid = DragonTrainer.avg_sigmoid_output(model_instance, X)
            ms_avg_sigmoid = DragonTrainer.avg_sigmoid_output(model_instance, X_many)
            fs_avg_sigmoid = DragonTrainer.avg_sigmoid_output(model_instance, X_few)
            res_df.loc['avg_sigmoid'] = [all_avg_sigmoid, ms_avg_sigmoid, fs_avg_sigmoid]
        print(res_df)
        res = {}
        res['val_avgScore'] = avg_acc
        res['val_perClassAcc'] = pc_acc
        res['val_ms_pc_acc'] = ms_pc_acc
        res['val_fs_pc_acc'] = fs_pc_acc
        res['val_har_acc'] = pc_harmonic_acc
        return res

    @staticmethod
    def vaya_evaluate(Y_true, Y_preds):
        total_acc = accuracy_score(Y_true, Y_preds)
        per_class_acc = DragonTrainer.calc_per_class_acc(Y_true, Y_preds)
        print("val_perClassAcc:", per_class_acc)
        return {"val_total_acc": total_acc,
                "val_perClassAcc": per_class_acc}

    def preprate_callbacks_for_vaya(self, model_instance, params):
        training_CB = []

        monitor, mon_mode = 'val_perClassAcc', 'max'
        X_val, Y_val_oh = params

        def my_func():
            predictions = model_instance.predict_val_layer(X_val)
            y_true, y_pred = np.argmax(Y_val_oh, axis=1), predictions.argmax(axis=1)
            return DragonTrainer.vaya_evaluate(y_true, y_pred)
            # total_acc = accuracy_score()
            # per_class_acc = DragonTrainer.calc_per_class_acc(np.argmax(Y_val, axis=1), predictions.argmax(axis=1))
            # mean_super_class = DragonTrainer.calc_mean_super_class_acc(np.argmax(Y_val, axis=1),
            #                                                            predictions.argmax(axis=1),
            #                                                            super_classes,
            #                                                            map_to_super_class)
            # avg_acc = (total_acc + per_class_acc + mean_super_class) / 3
            # return {"val_total_acc": total_acc,
            #         "val_perClassAcc": per_class_acc,
            #         "val_superClassAcc": mean_super_class,
            #         "val_avg": avg_acc}

        training_CB += [callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update(my_func())
        )]

        print(f'monitoring = {monitor}')
        # Save a model checkpoint only when monitor indicates that the best performance so far
        training_CB += [
            callbacks.ModelCheckpoint(monitor=monitor, mode=mon_mode,
                                      save_best_only=True,
                                      filepath=os.path.join(self.training_dir, 'best-checkpoint'),
                                      verbose=UserArgs.verbose)]

        # Set an early stopping callback
        training_CB += [callbacks.EarlyStopping(monitor=monitor, mode=mon_mode,
                                                patience=UserArgs.patience,
                                                verbose=UserArgs.verbose,
                                                min_delta=UserArgs.min_delta)]

        # Log training history to CSV
        training_CB += [callbacks.CSVLogger(os.path.join(self.training_dir, 'training_log.csv'),
                                            separator='|', append=True)]

        # Flush stdout buffer on every epoch
        training_CB += [callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())]
        return training_CB

    def prepare_callbacks_for_training(self, model_instance, eval_params, use_custom_eval=True):
        """
        Prepare Keras Callbacks for model training
        Returns a list of keras callbacks
        """
        training_CB = []

        if eval_params is None:
            monitor, mon_mode = 'val_acc', 'max'
        else:
            X_val, Y_val, val_classes, train_distribution, eval_n_repeat, eval_seed, \
            ms_classes, fs_classes, X_val_many, Y_val_many, X_val_few, Y_val_few = eval_params
            evaluate_specific_params = (train_distribution, eval_n_repeat, eval_seed,
                                        ms_classes, fs_classes)

            # Set the monitor (metric) for validation.
            # This is used for early-stopping during development.
            monitor, mon_mode = None, None

            if use_custom_eval:
                if UserArgs.train_dist == "dragon":
                    monitor, mon_mode = 'val_perClassAcc', 'max'
                else:
                    monitor, mon_mode = 'val_perClassAcc', 'max'

                training_CB += [callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logs.update(
                        DragonTrainer.training_evaluation(model_instance, (X_val, X_val_many, X_val_few),
                                                          (Y_val, Y_val_many, Y_val_few),
                                                          (val_classes, ms_classes, fs_classes),
                                                          evaluate_specific_params))
                )]
            else:
                monitor, mon_mode = 'val_har_acc', 'max'
                training_CB += [callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logs.update(
                        DragonTrainer.training_evaluation(model_instance, (X_val, X_val_many, X_val_few),
                                                          (Y_val, Y_val_many, Y_val_few),
                                                          (val_classes, ms_classes, fs_classes),
                                                          evaluate_specific_params))
                )]
        print(f'monitoring = {monitor}')
        # Save a model checkpoint only when monitor indicates that the best performance so far
        training_CB += [
            callbacks.ModelCheckpoint(monitor=monitor, mode=mon_mode,
                                      save_best_only=True,
                                      filepath=os.path.join(self.training_dir, 'best-checkpoint'),
                                      verbose=UserArgs.verbose)]

        # Set an early stopping callback
        training_CB += [callbacks.EarlyStopping(monitor=monitor, mode=mon_mode,
                                                patience=UserArgs.patience,
                                                verbose=UserArgs.verbose,
                                                min_delta=UserArgs.min_delta)]

        # Log training history to CSV
        training_CB += [callbacks.CSVLogger(os.path.join(self.training_dir, 'training_log.csv'),
                                            separator='|', append=True)]

        # Flush stdout buffer on every epoch
        training_CB += [callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())]
        return training_CB

    @staticmethod
    def __evaluate(model_instance, X, Y, classes_subset, eval_sp_params, plot_cm=None, threshold=None):
        # Inner function to avoid code duplication
        # returns: regular accuracy score, per class accuracy score, dragon avg score
        train_distribution, eval_n_repeat, \
        eval_seed, ms_classes, fs_classes = eval_sp_params

        predictions = model_instance.predict_val_layer(X)
        if threshold is not None:
            predictions[:, ms_classes] = predictions[:, ms_classes] - threshold

        subset_preds = classes_subset[(predictions[:, classes_subset]).argmax(axis=1)]

        # evaluate performance using regular accuracy function
        reg_acc = float(accuracy_score(Y, subset_preds))
        # evaluate performance using per class accuracy
        pc_acc = DragonTrainer.calc_per_class_acc(Y, subset_preds)
        # evaluate performance using average accuracy score function (dragon evaluation)
        dragon_acc = DragonTrainer.calc_avgScore(Y, subset_preds, train_distribution,
                                                 eval_n_repeat, eval_seed)

        return predictions, subset_preds, reg_acc, pc_acc, dragon_acc

    def evaluate_and_save_metrics(self, model_instance,
                                  train_data, val_data, test_data, test_eval_params,
                                  plot_thresh=True,
                                  should_save_predictions=True,
                                  should_save_metrics=True):
        X_train, Y_train, Attributes_train, train_classes = train_data
        X_val, Y_val, Attributes_val, val_classes = val_data
        X_test, Y_test, Attributes_test, test_classes = test_data
        _, _, _, train_distribution, eval_n_repeat, eval_seed, \
        ms_classes, fs_classes, X_test_many, Y_test_many, X_test_few, Y_test_few = test_eval_params

        evaluate_specific_params = (train_distribution, eval_n_repeat, eval_seed,
                                    ms_classes, fs_classes)

        # Evaluate on train data
        train_preds_score, train_preds_argmax, train_reg_acc, train_pc_acc, train_dragon_acc \
            = DragonTrainer.__evaluate(model_instance, X_train, Y_train, train_classes, evaluate_specific_params)

        # Evaluate on val data
        val_preds_score, val_preds_argmax, val_reg_acc, val_pc_acc, val_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_val, Y_val, val_classes, evaluate_specific_params)

        # Evaluate on test data
        test_preds_score, test_preds_argmax, test_reg_acc, test_pc_acc, test_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_test, Y_test, test_classes, evaluate_specific_params)

        ms_test_preds_score, ms_test_preds_argmax, ms_test_reg_acc, ms_test_pc_acc, ms_test_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_test_many, Y_test_many, test_classes, evaluate_specific_params)
        # many among many
        ms_among_ms_test_preds_score, ms_among_ms_test_preds_argmax, ms_among_ms_test_reg_acc, \
        ms_among_ms_test_pc_acc, ms_among_ms_test_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_test_many, Y_test_many, ms_classes, evaluate_specific_params)
        # print("M/M:", ms_among_ms_val_reg_acc, ms_among_ms_val_pc_acc, ms_among_ms_val_avg_acc)
        fs_test_preds_score, fs_test_preds_argmax, fs_test_reg_acc, fs_test_pc_acc, fs_test_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_test_few, Y_test_few, test_classes, evaluate_specific_params)
        # few among few
        fs_among_fs_test_preds_score, fs_among_fs_test_preds_argmax, \
        fs_among_fs_test_reg_acc, fs_among_fs_test_pc_acc, fs_among_fs_test_avg_acc = \
            DragonTrainer.__evaluate(model_instance, X_test_few, Y_test_few, fs_classes, evaluate_specific_params)
        # print("F/F:", fs_among_fs_val_reg_acc, fs_among_fs_val_pc_acc, fs_among_fs_val_avg_acc)

        # Print Results
        res_df = pd.DataFrame(columns=['reg_acc', 'per_class_acc', 'dragon_acc'])
        res_df.loc["Train"] = [train_reg_acc, train_pc_acc, train_dragon_acc]
        res_df.loc["Val"] = [val_reg_acc, val_pc_acc, val_avg_acc]
        res_df.loc["MS_Test"] = [ms_test_reg_acc, ms_test_pc_acc, ms_test_avg_acc]
        res_df.loc["FS_Test"] = [fs_test_reg_acc, fs_test_pc_acc, fs_test_avg_acc]
        res_df.loc["H_Test"] = [DragonTrainer.harmonic_acc(ms_test_reg_acc, fs_test_reg_acc),
                                DragonTrainer.harmonic_acc(ms_test_pc_acc, fs_test_pc_acc),
                                DragonTrainer.harmonic_acc(ms_test_avg_acc, fs_test_avg_acc)]
        res_df.loc["Test"] = [test_reg_acc, test_pc_acc, test_avg_acc]
        res_df.loc["MS/MS"] = [ms_among_ms_test_reg_acc, ms_among_ms_test_pc_acc, ms_among_ms_test_avg_acc]
        res_df.loc["FS/FS"] = [fs_among_fs_test_reg_acc, fs_among_fs_test_pc_acc, fs_among_fs_test_avg_acc]
        pd.options.display.float_format = '{:,.3f}'.format
        print(res_df)
        # print(f"""{res_df.loc["MS_Val"][1]},{res_df.loc["FS_Val"][1]},{res_df.loc["H_Val"][1]}""")

        if should_save_predictions:
            # Save predictions to train dir
            train_pkl_path = os.path.join(self.training_dir, 'predictions_train.pkl')
            data_to_pkl(dict(pred_score_classes=train_preds_score,
                             pred_argmax_classes=train_preds_argmax,
                             gt_classes=Y_train,
                             classes_ids=train_classes), train_pkl_path)
            print(f'Train predictions were written to {train_pkl_path}')

            val_pkl_path = os.path.join(self.training_dir, 'predictions_val.pkl')
            data_to_pkl(dict(pred_score_classes=val_preds_score,
                             pred_argmax_classes=val_preds_argmax,
                             gt_classes=Y_val,
                             classes_ids=val_classes), val_pkl_path)
            print(f'Val predictions were written to {val_pkl_path}')

            test_pkl_path = os.path.join(self.training_dir, 'predictions_test.pkl')
            data_to_pkl(dict(pred_score_classes=test_preds_score,
                             pred_argmax_classes=test_preds_argmax,
                             gt_classes=Y_test,
                             classes_ids=test_classes), test_pkl_path)
            print(f'Test predictions were written to {test_pkl_path}')

        if should_save_metrics:
            # save metrics to train dir
            metrics_path = os.path.join(self.training_dir, 'results.json')
            metric_results = dict(train_accuracy=list(res_df.loc["Train"]),
                                  val_avg_accuracy=list(res_df.loc["Val"]),
                                  ms_val_avg_accuracy=list(res_df.loc["MS_Test"]),
                                  fs_val_avg_accuracy=list(res_df.loc["FS_Test"]),
                                  h_val_avg_accuracy=list(res_df.loc["H_Test"]),
                                  test_avg_accuracy=list(res_df.loc["Test"]),
                                  ms_among_ms_accuracy=list(res_df.loc["MS/MS"]),
                                  fs_among_fs_accuracy=list(res_df.loc["FS/FS"]))
            with open(metrics_path, 'w') as m_f:
                json.dump(metric_results, fp=m_f, indent=4)
            print(f'Results were written to {metrics_path}')
