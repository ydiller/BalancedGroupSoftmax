import os

from keras import regularizers
from keras.engine import Layer
from keras.models import Model
from keras.layers import Input, Activation, Dense, Lambda
from smDragon.DragonTrainer import DragonTrainer
from smDragon.arg_parser import UserArgs


class smDragon(object):
    def __init__(self, input_dim, num_categories):
        print("smDRAGON")
        reg = 0
        input_layer = Input(shape=(input_dim,), name="Input_Layer")
        #soft_input = Activation("softmax")(input_layer)
        class_weights = Dense(num_categories, activation="sigmoid", name="Class_weights",
                              kernel_regularizer=regularizers.l2(reg))(input_layer)
        weighted_expert_preds = Lambda(lambda x: x[0] * x[1], name="Weighted_Preds") \
            ([input_layer, class_weights])
        name = f"smDragon_reg={reg}"
        self.model = Model(inputs=input_layer, outputs=weighted_expert_preds)
        self.model_name = name
        self.dragon_trainer = DragonTrainer(self.model_name, f"-lr={UserArgs.initial_learning_rate}")

    def get_model(self):
        return self.model

    def compile_model(self):
        self.model.compile(optimizer=DragonTrainer._init_optimizer("adam", UserArgs.initial_learning_rate),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def fit_model(self, X_train, Y_train, X_val, Y_val):
        # creating training dir
        should_fit = self.dragon_trainer.create_training_dir()
        if not should_fit:
            return

        sample_weights = None
        # use class weights
        y_ints = [y.argmax() for y in Y_train]
        sample_weights = DragonTrainer.balance_data_with_sample_weights(y_ints, add_dummy_class=False)
        print("Use sample weights:", sample_weights)

        # prepare callbacks
        training_CB = self.dragon_trainer.preprate_callbacks_for_vaya(self, (X_val, Y_val))
        fit_kwargs = dict(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            callbacks=training_CB,
            batch_size=UserArgs.batch_size,
            epochs=UserArgs.max_epochs,
            sample_weight=sample_weights,
            verbose=2)
        # train model
        self.model.fit(**fit_kwargs)

    def load_best_model(self, with_hp_ext=True):
        path = self.dragon_trainer.training_dir
        if not with_hp_ext:
            path = self.dragon_trainer.training_dir_wo_ext
        self.model.load_weights(os.path.join(path, "best-checkpoint"))

    def predict_val_layer(self, X):
        return self.model.predict(X, batch_size=UserArgs.batch_size)

