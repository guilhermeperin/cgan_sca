import argparse
from CGANSCA import *

dataset_root_path = "my_datasets_path"
results_root_path = "my_results_path"
features_root_path = "my_cgan_reference_features_path"


def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    """ root path for datasets """
    parser.add_argument("-dataset_root_path", "--dataset_root_path", default=dataset_root_path)

    """ root path for results """
    parser.add_argument("-results_root_path", "--results_root_path", default=results_root_path)

    """ root path for reference features """
    parser.add_argument("-features_root_path", "--features_root_path", default=features_root_path)

    """ dataset_reference: name of reference dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_reference", "--dataset_reference", default="ascad-variable")

    """ dataset_reference_dim: number of features (samples) in reference dataset """
    parser.add_argument("-dataset_reference_dim", "--dataset_reference_dim", default=1400)

    """ dataset_target: name of target dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_target", "--dataset_target", default="ASCAD")

    """ dataset_target_dim: number of features (samples) in target dataset """
    parser.add_argument("-dataset_target_dim", "--dataset_target_dim", default=700)

    """ features: number of features extracted from reference dataset and output dimension of generator """
    parser.add_argument("-features", "--features", default=100)

    """ n_profiling_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_profiling_reference", "--n_profiling_reference", default=200000)

    """ n_attack_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_attack_reference", "--n_attack_reference", default=5000)

    """ n_profiling_target: number of profiling traces from the target dataset """
    parser.add_argument("-n_profiling_target", "--n_profiling_target", default=50000)

    """ n_validation_target: number of validation traces from the target dataset """
    parser.add_argument("-n_validation_target", "--n_validation_target", default=0)

    """ n_attack_target: number of attack traces from the target dataset """
    parser.add_argument("-n_attack_target", "--n_attack_target", default=10000)

    """ n_attack_ge: number of attack traces for guessing entropy calculation """
    parser.add_argument("-n_attack_ge", "--n_attack_ge", default=2000)

    """ target_byte_reference: key byte index in the reference dataset """
    parser.add_argument("-target_byte_reference", "--target_byte_reference", default=2)

    """ target_byte_target: key byte index in the target dataset """
    parser.add_argument("-target_byte_target", "--target_byte_target", default=2)

    """ leakage_model: leakage model type (ID or HW) """
    parser.add_argument("-leakage_model", "--leakage_model", default="ID")

    """ epochs: number of training epochs for CGAN """
    parser.add_argument("-epochs", "--epochs", default=200)

    """ batch_size: batch size for training the CGAN """
    parser.add_argument("-batch_size", "--batch_size", default=400)

    """ std_gaussian_noise_reference: standard deviation for Gaussian noise artificially added to reference dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_reference", "--std_gaussian_noise_reference", default=0.0)

    """ std_gaussian_noise_target: standard deviation for Gaussian noise artificially added to target dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_target", "--std_gaussian_noise_target", default=0.0)

    """ random_search_generator: when set, generator hyperparameters are generated randomly (default is 0) """
    parser.add_argument("-random_search_generator", "--random_search_generator", default=0)

    """ random_search_discriminator: when set, discriminator hyperparameters are generated randomly (default is 0) """
    parser.add_argument("-random_search_discriminator", "--random_search_discriminator", default=0)

    return parser.parse_args()


if __name__ == "__main__":
    arg_list = get_arguments()

    arguments = {
        "dataset_root_path": arg_list.dataset_root_path,
        "results_root_path": arg_list.results_root_path,
        "features_root_path": arg_list.features_root_path,
        "dataset_reference": arg_list.dataset_reference,
        "dataset_reference_dim": int(arg_list.dataset_reference_dim),
        "dataset_target": arg_list.dataset_target,
        "dataset_target_dim": int(arg_list.dataset_target_dim),
        "n_profiling_reference": int(arg_list.n_profiling_reference),
        "n_attack_reference": int(arg_list.n_attack_reference),
        "n_profiling_target": int(arg_list.n_profiling_target),
        "n_validation_target": int(arg_list.n_validation_target),
        "n_attack_target": int(arg_list.n_attack_target),
        "n_attack_ge": int(arg_list.n_attack_ge),
        "target_byte_reference": int(arg_list.target_byte_reference),
        "target_byte_target": int(arg_list.target_byte_target),
        "features": int(arg_list.features),
        "leakage_model": arg_list.leakage_model,
        "epochs": int(arg_list.epochs),
        "batch_size": int(arg_list.batch_size),
        "std_gaussian_noise_reference": float(arg_list.std_gaussian_noise_reference),
        "std_gaussian_noise_target": float(arg_list.std_gaussian_noise_target),
        "random_search_generator": False if int(arg_list.random_search_generator) == 0 else True,
        "random_search_discriminator": False if int(arg_list.random_search_discriminator) == 0 else True,
    }

    cgan = CGANSCA(args=arguments)
    cgan.train_cgan()
