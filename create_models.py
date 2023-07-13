import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random
import numpy as np


class CreateModels:

    def __init__(self, args, dir_results):
        # helper functions for tensorflow compiling
        self.real_accuracy_metric = BinaryAccuracy()
        self.fake_accuracy_metric = BinaryAccuracy()
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.cross_entropy_disc = BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)

        classes = 9 if args["leakage_model"] == "HW" else 256

        if args["random_search_generator"] or args["random_search_discriminator"]:

            self.hp_d = {
                "neurons_embed": random.choice([100, 200, 500]),
                "neurons_dropout": random.choice([100, 200, 500]),
                "neurons_bilinear": random.choice([100, 200, 500]),
                "layers_embed": random.choice([1, 2, 3]),
                "layers_dropout": random.choice([1, 2, 3]),
                "layers_bilinear": random.choice([1, 2, 3]),
                "dropout": random.choice([0.5, 0.6, 0.7, 0.8]),
            }
            if args["generator_type"] == "cnn":
                self.hp_g = {
                    "neurons_1": random.choice([100, 200, 300, 400, 500]),
                    "layers": random.choice([1, 2, 3, 4]),
                    "conv_layers": random.choice([1, 2]),
                    "filters_1": random.choice([8, 16, 32]),
                    "kernel_size_1": random.choice([10, 20, 40]),
                    "strides_1": random.choice([5, 10]),
                    "activation": random.choice(["elu", "selu", "relu", "leakyrelu", "linear", "tanh"]),
                }
                for l_i in range(1, self.hp_g["conv_layers"]):
                    self.hp_g[f"filters_{l_i + 1}"] = self.hp_g[f"filters_{l_i}"] * 2
                    self.hp_g[f"kernel_size_{l_i + 1}"] = random.choice([10, 20, 40]),
                    self.hp_g[f"strides_{l_i + 1}"] = random.choice([5, 10]),
                for l_i in range(1, self.hp_g["layers"]):
                    options_neurons = list(range(100, self.hp_g[f"neurons_{l_i}"] + 100, 100))
                    self.hp_g[f"neurons_{l_i + 1}"] = random.choice(options_neurons)
            else:
                self.hp_g = {
                    "neurons_1": random.choice([100, 200, 300, 400, 500]),
                    "layers": random.choice([1, 2, 3, 4]),
                    "activation": random.choice(["elu", "selu", "relu", "leakyrelu", "linear", "tanh"]),
                }
                for l_i in range(1, self.hp_g["layers"]):
                    options_neurons = list(range(100, self.hp_g[f"neurons_{l_i}"] + 100, 100))
                    self.hp_g[f"neurons_{l_i + 1}"] = random.choice(options_neurons)

            # create the discriminator
            self.discriminator = self.define_discriminator_random(args["features"], n_classes=classes)
            # create the generator
            self.generator = self.define_generator_random(args["dataset_target_dim"], args["features"])

            np.savez(f"{dir_results}/hp.npz", hp_d=self.hp_d, hp_g=self.hp_g)
        else:

            self.best_models_random_search(args["dataset_reference"], args["dataset_target"])
            # create the discriminator
            self.discriminator = self.define_discriminator_random(args["features"], n_classes=classes)
            # create the generator
            self.generator = self.define_generator_random(args["dataset_target_dim"], args["features"])

    def best_models_random_search(self, reference, target):
        if reference == "ascad-variable":
            if target == "ASCAD":
                # reference ascad-variable (25000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 3, 'dropout': 0.7,
                    'neurons_bilinear': 200, 'layers_bilinear': 1,
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 1, 'activation': 'linear'
                }
            if target == "dpa_v42":
                # reference ascad-variable (25000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 4, 'activation': 'linear', 'neurons_2': 200, 'neurons_3': 200, 'neurons_4': 100
                }
            if target == "ches_ctf":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "eshard":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 500, 'neurons_3': 100
                }
        if reference == "ASCAD":
            if target == "ascad-variable":
                # reference ASCAD (10000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.6,
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 200, 'neurons_3': 100
                }
            if target == "dpa_v42":
                # reference ASCAD (10000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 2, 'activation': 'linear', 'neurons_2': 100
                }
            if target == "ches_ctf":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "eshard":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 2, 'activation': 'selu', 'neurons_2': 400
                }
        if reference == "dpa_v42":
            if target == "ascad-variable":
                # reference dpa_v42 (15000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'elu'
                }
            if target == "ASCAD":
                # reference dpa_v42 (15000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "ches_ctf":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'linear'
                }
            if target == "eshard":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'selu', 'neurons_2': 300
                }
        if reference == "eshard":
            if target == "ascad-variable":
                # reference eshard (1400) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "ASCAD":
                # reference eshard (1400) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'linear', 'neurons_2': 300
                }
            if target == "dpa_v42":
                # reference eshard (1400) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 1, 'activation': 'linear'
                }

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy_disc(tf.ones_like(real), real)
        fake_loss = self.cross_entropy_disc(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def generator_loss(self, fake):
        return self.cross_entropy(tf.ones_like(fake), fake)

    def define_discriminator(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        y = Dense(200, kernel_initializer=kern_init)(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = Dense(100, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)
        x = Dropout(0.60)(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

    def define_discriminator_random(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        for l_i in range(self.hp_d["layers_embed"]):
            y = Dense(self.hp_d["neurons_embed"], kernel_initializer=kern_init)(y)
            y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = None
        for l_i in range(self.hp_d["layers_dropout"]):
            x = Dense(self.hp_d["neurons_dropout"], kernel_initializer=kern_init)(merge if l_i == 0 else x)
            x = LeakyReLU()(x)
            x = Dropout(self.hp_d["dropout"])(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

        # define the standalone generator model

    def define_generator(self, input_dim: int, output_dim: int, n_classes=256):
        # input_random_data = Input(shape=(self.traces_target_dim,))
        # rnd = Dense(400, activation='elu')(input_random_data)

        in_traces = Input(shape=(input_dim,))
        # x = Lambda(self.add_gaussian_noise)(in_traces)
        x = Dense(400, activation='linear')(in_traces)
        x = Dense(200, activation='linear')(x)
        x = Dense(100, activation='linear')(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model

    # define a random generator model
    def define_generator_random(self, input_dim: int, output_dim: int):

        in_traces = Input(shape=(input_dim,))
        x = None
        for l_i in range(self.hp_g["layers"]):
            x = Dense(self.hp_g[f"neurons_{l_i + 1}"],
                      activation=self.hp_g["activation"] if self.hp_g["activation"] != "leakyrelu" else None)(in_traces if l_i == 0 else x)
            if self.hp_g["activation"] == "leakyrelu":
                x = LeakyReLU()(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model
