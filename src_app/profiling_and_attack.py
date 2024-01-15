from keras.optimizers import Adam
from scipy.stats import entropy
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.callbacks import *
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler


class ProgressCallback(Callback):
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress(epoch + 1, text=f"Epoch: {epoch + 1}/{self.epochs}")


def mlp(output_dim, number_of_samples, mlp_layers, mlp_neurons, mlp_activation, learning_rate=0.001):
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = None
    for l_i in range(mlp_layers):
        if mlp_activation == "leakyrelu":
            x = Dense(mlp_neurons, kernel_initializer="glorot_normal")(input_layer if l_i == 0 else x)
            x = LeakyReLU()(x)
        else:
            x = Dense(mlp_neurons, kernel_initializer="glorot_normal", activation=mlp_activation)(
                input_layer if l_i == 0 else x)

    output_layer = Dense(output_dim, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(learning_rate=learning_rate)
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model


def guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces, key_rank_report_interval=1):
    """
    Function to compute Guessing Entropy
    - this function computes a list of key candidates, ordered by their probability of being the correct key
    - if this function returns final_ge=1, it means that the correct key is actually indicated as the most likely one.
    - if this function returns final_ge=256, it means that the correct key is actually indicated as the least likely one.
    - if this function returns final_ge close to 128, it means that the attack is wrong and the model is simply returing a random key.

    :return
    - final_ge: the guessing entropy of the correct key
    - guessing_entropy: a vector indicating the value 'final_ge' with respect to the number of processed attack measurements
    - number_of_measurements_for_ge_1: the number of processed attack measurements necessary to reach final_ge = 1
    """

    nt = len(predictions)

    key_rank_executions = 40

    # key_ranking_sum = np.zeros(key_rank_attack_traces)
    key_ranking_sum = np.zeros(
        int(key_rank_attack_traces / key_rank_report_interval))

    predictions = np.log(predictions + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index])
                        for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(
            range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(
                    key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    ge = key_ranking_sum / key_rank_executions

    number_of_measurements_for_ge_1 = key_rank_attack_traces
    if ge[int(key_rank_attack_traces / key_rank_report_interval) - 1] < 2:
        for index in range(int(key_rank_attack_traces / key_rank_report_interval) - 1, -1, -1):
            if ge[index] > 2:
                number_of_measurements_for_ge_1 = (index + 1) * key_rank_report_interval
                break

    final_ge = ge[int(key_rank_attack_traces / key_rank_report_interval) - 1]
    print("GE = {}".format(final_ge))
    print("Number of traces to reach GE = 1: {}".format(number_of_measurements_for_ge_1))

    return final_ge, ge, number_of_measurements_for_ge_1


def perceived_information(model, labels, num_classes):
    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    acc = entropy(p_k, base=2)  # we initialize the value with H(K)

    y_pred = np.array(model + 1e-36)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = y_pred[trace_index_with_label_k, k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            acc += p_k[k] * p_k_l

    print(f"PI: {acc}")

    return acc


def extract_features(generator_model, features_dim, dataset, dataset_filepath, progress_bar_features_extraction):
    in_file = h5py.File(dataset_filepath, "r")

    tr_step = 1000

    features_target_profiling_syn = np.zeros((dataset.n_profiling, features_dim))
    features_target_attack_syn = np.zeros((dataset.n_attack, features_dim))

    if dataset.dataset_name == "ascadf" or dataset.dataset_name == "ascadr":
        h5_traces_profiling = "traces"
        h5_traces_attack = "traces"
    else:
        h5_traces_profiling = "Profiling_traces/traces"
        h5_traces_attack = "Attack_traces/traces"

    for tr in range(0, dataset.n_profiling, tr_step):
        traces = np.array(in_file[h5_traces_profiling][tr:tr + tr_step])
        features_target_profiling_syn[tr:tr + tr_step] = np.array(generator_model.predict([traces]))
        progress_bar_features_extraction.progress(tr / dataset.n_profiling,
                                                  text=f"{tr} traces processed in generator's feature extraction.")
    for tr in range(0, dataset.n_attack, tr_step):
        traces = np.array(in_file[h5_traces_attack][tr + dataset.n_attack:tr + tr_step + dataset.n_attack])
        features_target_attack_syn[tr:tr + tr_step] = np.array(generator_model.predict([traces]))
        progress_bar_features_extraction.progress(tr / dataset.n_attack,
                                                  text=f"{tr} traces processed in generator's feature extraction.")

    return features_target_profiling_syn, features_target_attack_syn


def attack(dataset, dataset_filepath, generator_model, features_dim, mlp_layers, mlp_neurons, mlp_activation,
           progress_bar_features_extraction=None, progress_bar_epochs=None):
    """ Generate a batch of synthetic measurements with the trained generator """
    features_target_profiling_syn, features_target_attack_syn = extract_features(generator_model, features_dim, dataset,
                                                                                 dataset_filepath,
                                                                                 progress_bar_features_extraction)

    scaler = StandardScaler()
    features_target_profiling_syn = scaler.fit_transform(features_target_profiling_syn)
    features_target_attack_syn = scaler.transform(features_target_attack_syn)

    progress_callback = ProgressCallback(progress_bar_epochs, 100)

    print(features_target_profiling_syn.shape)
    print(features_target_attack_syn.shape)
    print(features_dim)

    """ Define a neural network (MLP) to be trained with synthetic traces """
    attack_model = mlp(dataset.classes, features_dim, mlp_layers, mlp_neurons, mlp_activation)
    attack_model.fit(
        x=features_target_profiling_syn,
        y=to_categorical(dataset.profiling_labels, num_classes=dataset.classes),
        batch_size=400,
        verbose=2,
        epochs=100,
        shuffle=True,
        validation_data=(
            features_target_attack_syn, to_categorical(dataset.attack_labels, num_classes=dataset.classes)),
        callbacks=[progress_callback])

    """ Predict the trained MLP with target/attack measurements """
    predictions = attack_model.predict(features_target_attack_syn)
    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset.labels_key_hypothesis_attack,
                                         dataset.correct_key_attack, 2000)
    pi = perceived_information(predictions, dataset.attack_labels, dataset.classes)
    return ge, nt, pi, ge_vector
