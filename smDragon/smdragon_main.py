from smDragon.arg_parser import visual_args

visual_args()
from smDragon.utils.evaluation import calculate_avg_cong_and_acc_per_class
from smDragon.arg_parser import UserArgs
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
from keras.utils import to_categorical
from smDragon.sm_dragon import smDragon
from smDragon.DragonTrainer import DragonTrainer
import pickle

if __name__ == "__main__":
    base_path = UserArgs.base_train_dir
    X_train = pickle.load(open(f"{base_path}/filtered_logits.p", 'rb'))
    Y_train = np.array(pickle.load(open(f"{base_path}/filtered_targets.p", 'rb')))

    classes = Counter(Y_train).keys()
    ordered_classes = list(
        reversed(list({k: v for k, v in sorted(Counter(Y_train).items(), key=lambda item: item[1])}.keys())))
    num_classes = 1231
    Y_train_oh = to_categorical(Y_train, num_classes=num_classes)
    # map_to_super_classes = np.array(
    #     ["Vehicle", "Vehicle", "Vehicle", "Vehicle", "Human", "Human", "Human", "Human", "Misc", "Bg"])
    # super_class_names = np.array(["Vehicle", "Human", "Misc","Bg"])
    class_samples_map = Counter(Y_train)
    num_training_samples_per_class = np.zeros(shape=(num_classes,))
    for key in sorted(class_samples_map.keys(), reverse=False):
        num_training_samples_per_class[key] = class_samples_map[key]
    print(num_training_samples_per_class)

    X_test = X_train
    Y_test = Y_train
    Y_test_oh = Y_train_oh

    X_val = X_test
    Y_val = Y_test
    Y_val_oh = Y_test_oh

    print("Train", X_train.shape, Y_train.shape)
    print("Test", X_test.shape, Y_test.shape)
    print("Classes", Counter(Y_test).keys())

    train_preds = np.argmax(X_train, axis=1)
    print("Train:", DragonTrainer.vaya_evaluate(Y_train, train_preds))
    val_preds = np.argmax(X_val, axis=1)
    print("Val:", DragonTrainer.vaya_evaluate(Y_val, val_preds))

    _, _, acc_pc, per_class_acc = calculate_avg_cong_and_acc_per_class(
        Y_val,
        X_val,
        np.argmax(X_val, axis=1),
        list(range(0,1232)),
        ordered_classes)
    print(acc_pc, per_class_acc)

    sm_model = smDragon(X_train.shape[1], 1231)
    sm_model.compile_model()
    sm_model.model.summary()

    sm_model.fit_model(X_train, Y_train_oh, X_val, Y_val_oh)

    sm_model.load_best_model()
    confs = sm_model.predict_val_layer(X_val)

    print("CB-val:",
          DragonTrainer.vaya_evaluate(Y_val, np.argmax(confs, axis=1)))
    _, _, acc_pc, per_class_acc = calculate_avg_cong_and_acc_per_class(
        Y_val,
        confs,
        np.argmax(confs, axis=1),
        list(range(0,1232)),
        ordered_classes)
    print(acc_pc, per_class_acc)
