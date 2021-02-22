from keras import regularizers
from keras.engine import Layer
from keras.models import Model
from keras.layers import Input, Activation, Dense, Lambda
from smDragon.DragonTrainer import DragonTrainer
from smDragon.arg_parser import UserArgs
import numpy as np
import tensorflow as tf
import keras.backend as K
import os

class smDragon(object):
    def __init__(self, input_dim, num_training_samples_per_class, nparams=3):
        print("smDragon - Building Parametric Gater")

        # model 1 input
        input_layer = Input(shape=(input_dim,), name="Input_Layer")
        expert_preds = Activation("softmax")(input_layer)
        # take num_training_samples_per_class as input
        num_samples_inp = Input(tensor=K.variable(np.array(num_training_samples_per_class).reshape(1, -1)),
                                name="Num_Samples_Per_Class")
        norm_num_samples = Lambda(lambda x: x / tf.reshape(tf.reduce_max(x, axis=-1), (-1, 1))
                                  , name="Norm_Num_Samples")(num_samples_inp)

        topk_expert_output = expert_preds
        params_layer = Dense(nparams, name="Poly_Params")(topk_expert_output)

        def poly_weights(x):
            # w0 + w1*x + w2*x^2 + w3*x^3
            poly_params = x[0]
            num_samples = x[1]
            w0 = tf.reshape(poly_params[:, 0], [-1, 1])
            w1 = tf.reshape(poly_params[:, 1], [-1, 1])
            if nparams == 2:
                return w0 + w1 * num_samples
            w2 = tf.reshape(poly_params[:, 2], [-1, 1])
            if nparams == 3:
                return w0 + w1 * num_samples + w2 * tf.pow(num_samples, 2)
            w3 = tf.reshape(poly_params[:, 3], [-1, 1])
            if nparams == 4:
                return w0 + w1 * num_samples + w2 * tf.pow(num_samples, 2) + w3 * tf.pow(num_samples, 3)
            w4 = tf.reshape(poly_params[:, 4], [-1, 1])
            return w0 + w1 * num_samples + w2 * tf.pow(num_samples, 2) + w3 * tf.pow(num_samples, 3) + w4 * tf.pow(
                num_samples, 4)

        # calculate  weights and apply sigmoid
        predictions_weights = Lambda(poly_weights, name="Poly_Weights") \
            ([params_layer, norm_num_samples])
        predictions_weights_sig = Activation("sigmoid", name="Sigmoid_Weights") \
            (predictions_weights)

        weighted_expert_preds = Lambda(lambda x: x[0] * x[1], name="Weighted_Preds") \
            ([expert_preds, predictions_weights_sig])
        out = Lambda(lambda x: x / tf.reshape(tf.reduce_sum(x, axis=-1), (-1, 1))
                     , name="Normalize_Final_Preds")(weighted_expert_preds)

        name = f"smDragon_poly_nparams={nparams}"
        self.model = Model(inputs=[input_layer, num_samples_inp], outputs=out)
        self.model_name = name
        self.dragon_trainer = DragonTrainer(self.model_name, f"-lr={UserArgs.initial_learning_rate}")

    def get_model(self):
        return self.model

    def compile_model(self):
        self.model.compile(optimizer=DragonTrainer._init_optimizer("adam", UserArgs.initial_learning_rate),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def fit_model(self, X_train, Y_train, X_val, Y_val, super_classes, map_to_super_class):
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
        training_CB = self.dragon_trainer.preprate_callbacks_for_vaya(self, (X_val, Y_val,
                                                                             super_classes,
                                                                             map_to_super_class))
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

