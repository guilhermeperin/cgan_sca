import os.path

import streamlit as st
import requests
import zipfile
import shutil
from time import time
from keras.metrics import BinaryAccuracy
from keras.losses import BinaryCrossentropy
import tensorflow as tf
import keras as ks
from src_app.load_ascadf import *
from src_app.load_ascadr import *
from src_app.load_eshard import *
from src_app.load_chesctf import *
from src_app.load_dpav42 import *
from src_app.target_labels import *
from src_app.reference_features import *
from src_app.profiling_and_attack import *
from database.SqlAlchemyWrapper import *
from database.create_tables import *

db_class = SqlAlchemyWrapper("analyses")
create_tables(db_class)

st.set_page_config(layout="wide")

st.title("A Novel Conditional GAN Framework for Efficient Profiling Side-channel Analysis")

st.write(
    "This page contains a demonstration of the CGAN-SCA framework as part of the CHES2024 submission paper entitled "
    "'It's a Kind of Magic: A Novel Conditional GAN Framework for Efficient Profiling Side-channel Analysis'.")

with st.container(border=True):
    st.write("Follow these steps to execute the framework:")
    st.write(
        "1. **Download** datasets (*this can take several minutes for large datasets. It only needs to be done once*).")
    st.write("2. **Prepare** datasets. This step prepares the reference and target datasets.")
    st.write("3. **Configure** generator, discriminator and profiling models.")
    st.write("4. **Train** CGAN-SCA and perform profiling attack-based **key recovery**")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Abstract", "Download", "Feature Selection", "Target Dataset", "Model Architectures", "Train & Key Recovery",
     "Results"])

with tab1:
    st.write(
        "Profiling side-channel analysis is an essential technique to assess the security of protected cryptographic "
        "implementations by subjecting them to the worst-case security analysis. This approach assumes the presence of a "
        "highly capable adversary with knowledge of countermeasures and randomness employed by the target device. However, "
        "black-box profiling attacks are commonly employed when aiming to emulate real-world scenarios. These attacks "
        "leverage deep learning as a prominent alternative since deep neural networks can automatically implement high-order "
        "attacks, eliminating the need for secret mask knowledge. Nevertheless, black-box profiling attacks often result in "
        "non-worst-case security evaluations, leading to suboptimal profiling models. In this paper, we propose modifying "
        "the conventional black-box threat model by incorporating a new assumption: the adversary possesses a similar "
        "implementation that can be used as a white-box reference design. We create an adversarial dataset by extracting "
        "features or points of interest from this reference design. These features are then utilized for training a novel "
        "conditional generative adversarial network (CGAN) framework, enabling a generative model to extract features from "
        "high-order leakages of other protected implementation without any assumptions about the masking scheme or secret "
        "masks from the evaluated device. Our framework empowers attackers to perform efficient black-box profiling attack "
        "that achieves (and even surpasses) the performance of the worst-case security assessments.")

with tab2:
    st.write("**Download datasets**")


    def download_file(url, output_file, progress_bar):
        try:
            with requests.get(url, stream=True) as response:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 KB
                written_size = 0

                with open(output_file, 'wb') as file:
                    for data in response.iter_content(chunk_size=block_size):
                        file.write(data)
                        written_size += len(data)
                        text = f"Downloading {output_file} - {round(written_size / total_size, 2) * 100}%"
                        progress_bar.progress(written_size / total_size, text=text)

            st.success(f"Download successful: {output_file}")
        except Exception as e:
            st.error(f"Error downloading file: {e}")


    def unzip_file(zip_file_path, extract_to_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)


    def snr_fast(x, y):
        ns = x.shape[1]
        unique = np.unique(y)
        means = np.zeros((len(unique), ns))
        variances = np.zeros((len(unique), ns))

        for i, u in enumerate(unique):
            new_x = x[np.argwhere(y == int(u))]
            means[i] = np.mean(new_x, axis=0)
            variances[i] = np.var(new_x, axis=0)
        return np.var(means, axis=0) / np.mean(variances, axis=0)


    def get_features(traces, share1, share2, n_poi=100):
        snr_prof_share_1 = snr_fast(np.array(traces, dtype=np.int16), np.asarray(share1))
        snr_prof_share_2 = snr_fast(np.array(traces, dtype=np.int16), np.asarray(share2))
        snr_prof_share_1[np.isnan(snr_prof_share_1)] = 0
        snr_prof_share_2[np.isnan(snr_prof_share_2)] = 0
        ind_snr_masks_poi_sm = np.argsort(snr_prof_share_1)[::-1][:int(n_poi / 2)]
        ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
        ind_snr_masks_poi_r2 = np.argsort(snr_prof_share_2)[::-1][:int(n_poi / 2)]
        ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

        poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

        return traces[:, poi_profiling]


    current_directory = os.path.dirname(os.path.realpath(__file__))
    datasets_path = os.path.join(current_directory, 'datasets')

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    url_ascadf = "https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip"
    ascadf_dataset_zip = "ASCAD_data.zip"
    ascadf_dataset = "ASCAD_nopoi_window_20.h5"

    progress_bar = st.progress(0)

    if st.button("Download ASCADf Dataset", key="downaload_ascadf"):
        if not os.path.exists(os.path.join(datasets_path, ascadf_dataset_zip)):
            download_file(url_ascadf, ascadf_dataset_zip, progress_bar)

            file_path_dataset = os.path.join(current_directory, ascadf_dataset_zip)
            destination_directory = 'datasets'
            shutil.move(file_path_dataset, destination_directory)

        if os.path.exists(os.path.join(datasets_path, ascadf_dataset_zip)):

            if not os.path.exists(os.path.join(datasets_path, "ASCAD_data\\ASCAD_databases\\ATMega8515_raw_traces.h5")):
                zip_file_path = os.path.join(datasets_path, ascadf_dataset_zip)

                if not os.path.exists(datasets_path):
                    os.makedirs(datasets_path)

                unzip_file(zip_file_path, datasets_path)

            if not os.path.exists(os.path.join(datasets_path, "ATMega8515_raw_traces.h5")):
                src_file = os.path.join(datasets_path, "ASCAD_data\\ASCAD_databases\\ATMega8515_raw_traces.h5")
                shutil.move(src_file, datasets_path)

            if not os.path.exists(os.path.join(datasets_path, "ASCAD_nopoi_window_20.h5")):
                progress_bar_ascadf = st.progress(0)
                generate_nopoi_ascadf(os.path.join(datasets_path, "ATMega8515_raw_traces.h5"),
                                      os.path.join(datasets_path, "ASCAD_nopoi_window_20.h5"), 60000, 100000,
                                      progress_bar_ascadf, window=20)
            else:
                st.success(f"ASCADf ({ascadf_dataset_zip}) file already downloaded.")

    url_ascadr = "https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190730-071646/atmega8515-raw-traces.h5"
    ascadr_dataset = "atmega8515-raw-traces.h5"

    progress_bar = st.progress(0)

    if st.button("Download ASCADr Dataset", key="downaload_ascadr"):
        if not os.path.exists(os.path.join(datasets_path, ascadr_dataset)):
            download_file(url_ascadr, ascadr_dataset, progress_bar)

            file_path_dataset = os.path.join(current_directory, ascadr_dataset)
            destination_directory = 'datasets'
            shutil.move(file_path_dataset, destination_directory)

        if not os.path.exists(os.path.join(datasets_path, "ascad-variable_nopoi_window_20.h5")):
            progress_bar_ascadr = st.progress(0)
            generate_nopoi_ascadr(os.path.join(datasets_path, "atmega8515-raw-traces.h5"),
                                  os.path.join(datasets_path, "ascad-variable_nopoi_window_20.h5"), 200000, 250000,
                                  progress_bar_ascadr, window=20)
            st.success(f"ASCADr dataset successfully downloaded and created.")
        else:
            st.success(f"ASCADr ({ascadr_dataset}) file already downloaded.")

    url_eshard = "https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled/-/raw/main/Nucleo_AES_masked_non_shuffled.ets"
    eshard_dataset_ets = "eshard.ets"
    eshard_dataset = "eshard.h5"

    progress_bar = st.progress(0)

    if st.button("Download ESHARD-128 Dataset", key="downaload_eshard"):
        if not os.path.exists(os.path.join(datasets_path, eshard_dataset_ets)):
            download_file(url_eshard, eshard_dataset_ets, progress_bar)

            file_path_dataset = os.path.join(current_directory, eshard_dataset_ets)
            destination_directory = 'datasets'
            shutil.move(file_path_dataset, destination_directory)

        if not os.path.exists(os.path.join(datasets_path, eshard_dataset)):
            convert_eshard_to_h5(datasets_path, os.path.join(datasets_path, eshard_dataset_ets))
            st.success(f"ESHARD-AES128 dataset successfully downloaded and created.")
        else:
            st.success(f"ESHARD-AES128 ({eshard_dataset}) file already downloaded.")

    # url_dpav42 = "http://aisylabdatasets.ewi.tudelft.nl/dpav42/dpa_v42_nopoi_window_20.h5"
    dpav42_dataset = "dpa_v42_nopoi_window_20.h5"

    progress_bar = st.progress(0)

    url_dpav42 = "http://aisylabdatasets.ewi.tudelft.nl/dpav42"
    url_dpav42_1 = "DPA_contestv4_2_k00.zip"
    url_dpav42_2 = "DPA_contestv4_2_k01.zip"
    url_dpav42_3 = "DPA_contestv4_2_k02.zip"
    url_dpav42_4 = "DPA_contestv4_2_k03.zip"
    url_dpav42_5 = "DPA_contestv4_2_k04.zip"
    url_dpav42_6 = "DPA_contestv4_2_k05.zip"
    url_dpav42_7 = "DPA_contestv4_2_k06.zip"
    url_dpav42_8 = "DPA_contestv4_2_k07.zip"
    url_dpav42_9 = "DPA_contestv4_2_k08.zip"
    url_dpav42_10 = "DPA_contestv4_2_k09.zip"
    url_dpav42_11 = "DPA_contestv4_2_k10.zip"
    url_dpav42_12 = "DPA_contestv4_2_k11.zip"
    url_dpav42_13 = "DPA_contestv4_2_k12.zip"
    url_dpav42_14 = "DPA_contestv4_2_k13.zip"
    url_dpav42_15 = "DPA_contestv4_2_k14.zip"
    url_dpav42_16 = "DPA_contestv4_2_k15.zip"
    url_dpav42_17 = "dpav4_2_index.txt"

    if st.button("Download DPAv42 Dataset", key="downaload_dpav42"):
        for dataset_dpav42 in [url_dpav42_1, url_dpav42_2, url_dpav42_3, url_dpav42_4, url_dpav42_5, url_dpav42_6,
                               url_dpav42_7, url_dpav42_8, url_dpav42_9, url_dpav42_10, url_dpav42_11, url_dpav42_12,
                               url_dpav42_13, url_dpav42_14, url_dpav42_15, url_dpav42_16, url_dpav42_17]:
            if not os.path.exists(os.path.join(datasets_path, dataset_dpav42)):
                download_file(f"{url_dpav42}/{dataset_dpav42}", dataset_dpav42, progress_bar)

                file_path_dataset = os.path.join(current_directory, dataset_dpav42)
                destination_directory = 'datasets'
                shutil.move(file_path_dataset, destination_directory)
        if not os.path.exists(os.path.join(datasets_path, dpav42_dataset)):
            progress_bar_dpav42 = st.progress(0)
            generate_nopoi_dpav42(datasets_path, os.path.exists(os.path.join(datasets_path, dpav42_dataset)),
                                  80000, 1704046, progress_bar_dpav42, window=20)
            st.success(f"DPAv42 dataset successfully downloaded and created.")
        else:
            st.success(f"DPAv42 ({dataset_dpav42}) file already downloaded.")

    url_chesctf1 = "https://zenodo.org/record/3733418/files/PinataAcqTask2.1_10k_upload.trs"
    url_chesctf2 = "https://zenodo.org/record/3733418/files/PinataAcqTask2.2_10k_upload.trs"
    url_chesctf3 = "https://zenodo.org/record/3733418/files/PinataAcqTask2.3_10k_upload.trs"
    url_chesctf4 = "https://zenodo.org/record/3733418/files/PinataAcqTask2.4_10k_upload.trs"
    chesctf_dataset1 = "PinataAcqTask2.1_10k_upload.trs"
    chesctf_dataset2 = "PinataAcqTask2.2_10k_upload.trs"
    chesctf_dataset3 = "PinataAcqTask2.3_10k_upload.trs"
    chesctf_dataset4 = "PinataAcqTask2.4_10k_upload.trs"
    chesctf_dataset = "ches_ctf_nopoi_window_20.h5"

    progress_bar = st.progress(0)
    text_download_chesctf = st.empty()
    progress_bar_chesctf = st.empty()

    if st.button("Download CHES CTF 2018 Dataset", key="downaload_chesctf"):
        for dataset_chesctf in [chesctf_dataset1, chesctf_dataset2, chesctf_dataset3, chesctf_dataset4]:
            if not os.path.exists(os.path.join(datasets_path, dataset_chesctf)):
                text_download_chesctf = st.write(f"Downloading {dataset_chesctf} dataset")

                download_file(url_chesctf1, dataset_chesctf, progress_bar)

                file_path_dataset = os.path.join(current_directory, dataset_chesctf)
                destination_directory = 'datasets'
                shutil.move(file_path_dataset, destination_directory)

        if not os.path.exists(os.path.join(datasets_path, chesctf_dataset)):
            generate_chestf_nopoi(datasets_path, progress_bar_chesctf)
            st.success(f"CHES CTF 2018 dataset successfully downloaded and created.")
        else:
            st.success(f"CHES CTF 2018 ({chesctf_dataset}) file already downloaded.")

with tab3:
    st.write("**Generate $f_{ref}$** (a.k.a. reference features)")

    # Initialize session state
    if 'reference_features' not in st.session_state:
        st.session_state.reference_features = None

    if 'target_dataset' not in st.session_state:
        st.session_state.target_dataset = None

    if 'db_attack' not in st.session_state:
        st.session_state.db_attack = None

    if 'reference_dataset_ok' not in st.session_state:
        st.session_state.reference_dataset_ok = True

    with st.container(border=True):

        st.write("**Reference dataset**")

        reference_dataset_name = None

        current_directory = os.path.dirname(os.path.realpath(__file__))
        datasets_path = os.path.join(current_directory, 'datasets')

        lm_col1, tb_col2, nf_col3, ref_dataset_col = st.columns(4, gap="medium")
        with lm_col1:
            reference_leakage_model = st.selectbox('Leakage Model', ('HW', 'ID'), key="leakage_model_reference")
        with tb_col2:
            options = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            target_byte_reference = st.selectbox('Reference Key byte', options, index=options.index(2),
                                                 key="target_byte_reference")
        with nf_col3:
            cgan_features = st.slider('CGAN Features $N_{f}$', min_value=10, max_value=100, value=100)

        with ref_dataset_col:
            reference_dataset_select = st.selectbox('Reference Dataset',
                                                    ('ASCADv1 with fixed keys',
                                                     'ASCADv1 with variable keys',
                                                     'DPAv4.2',
                                                     'ESHARD-AES128'))
            if reference_dataset_select == "ASCADv1 with fixed keys":
                reference_dataset_name = "ascadf"
                n_prof_ref = 50000
                ns_ref = 100000
            if reference_dataset_select == "ASCADv1 with variable keys":
                reference_dataset_name = "ascadr"
                n_prof_ref = 200000
                ns_ref = 250000
            if reference_dataset_select == "DPAv4.2":
                reference_dataset_name = "dpav42"
                n_prof_ref = 70000
                ns_ref = 15000
            if reference_dataset_select == "ESHARD-AES128":
                reference_dataset_name = "eshard"
                n_prof_ref = 90000
                ns_ref = 1400

        if st.button("Prepare Reference Dataset", key="button_prepare_reference_dataset"):
            try:
                st.warning(f"Please wait while reference dataset is being prepared...")
                if reference_dataset_name == "ascadf":
                    if not os.path.exists(
                            f"{datasets_path}/reference_features_{reference_dataset_name}_{reference_leakage_model}.npz"):
                        ReferenceFeatures(reference_dataset_name, datasets_path,
                                          f'{datasets_path}/ATMega8515_raw_traces.h5',
                                          ns_ref, 100, n_prof_ref, target_byte_reference,
                                          reference_leakage_model)
                if reference_dataset_name == "ascadr":
                    if not os.path.exists(
                            f"{datasets_path}/reference_features_{reference_dataset_name}_{reference_leakage_model}.npz"):
                        ReferenceFeatures("ascadr", datasets_path,
                                          f'{datasets_path}/atmega8515-raw-traces.h5',
                                          ns_ref, 100, n_prof_ref, target_byte_reference,
                                          reference_leakage_model, ns_chunk_size=10000)
                if reference_dataset_name == "dpav42":
                    if not os.path.exists(
                            f"{datasets_path}/reference_features_{reference_dataset_name}_{reference_leakage_model}.npz"):
                        ReferenceFeatures("dpav42", datasets_path,
                                          f'{datasets_path}/dpa_v42_nopoi_window_20.h5',
                                          ns_ref, 100, n_prof_ref, target_byte_reference,
                                          reference_leakage_model, ns_chunk_size=5000)
                if reference_dataset_name == "eshard":
                    if not os.path.exists(
                            f"{datasets_path}/reference_features_{reference_dataset_name}_{reference_leakage_model}.npz"):
                        ReferenceFeatures("eshard", datasets_path,
                                          f'{datasets_path}/eshard.h5',
                                          ns_ref, 100, n_prof_ref, target_byte_reference,
                                          reference_leakage_model, ns_chunk_size=ns_ref)

                st.session_state.reference_features = \
                    np.load(
                        f"{datasets_path}/reference_features_{reference_dataset_name}_{reference_leakage_model}.npz")[
                        "reference_features"]
                st.session_state.reference_labels = \
                    np.load(f"{datasets_path}/{reference_dataset_name}_labels_{reference_leakage_model}.npz")[
                        "labels"]
                st.session_state.reference_share1 = \
                    np.load(f"{datasets_path}/{reference_dataset_name}_share1_{reference_leakage_model}.npz")[
                        "share1"]
                st.session_state.reference_share2 = \
                    np.load(f"{datasets_path}/{reference_dataset_name}_share2_{reference_leakage_model}.npz")[
                        "share2"]
                scaler = StandardScaler()
                st.session_state.reference_features = scaler.fit_transform(st.session_state.reference_features)
                st.session_state.reference_features = get_features(st.session_state.reference_features,
                                                                   st.session_state.reference_share1[
                                                                   target_byte_reference, :],
                                                                   st.session_state.reference_share2[
                                                                   target_byte_reference, :],
                                                                   n_poi=cgan_features)

                st.session_state.reference_dataset_ok = True
                st.success(f"{reference_dataset_select} dataset successfully prepared as a reference dataset.")

            except Exception as e:
                st.session_state.reference_dataset_ok = False
                st.error(f"Error preparing {reference_dataset_select} as a reference dataset ({e}).")

with tab4:
    with st.container(border=True):

        st.write("**Target dataset**")

        current_directory = os.path.dirname(os.path.realpath(__file__))
        datasets_path = os.path.join(current_directory, 'datasets')

        ref_col1, ref_col2, ref_col3, ref_col4 = st.columns(4)
        with ref_col1:
            leakage_model = st.selectbox('Leakage Model', ('HW', 'ID'))
        with ref_col2:
            options = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            target_byte_target = st.selectbox('Target Key byte', options, index=options.index(2),
                                              key="target_byte_target")
        with ref_col3:
            st.write(f"Number of features")
            st.write(f"Nf = **{cgan_features}**")
        with ref_col4:
            target_dataset_select = st.selectbox('Target Dataset',
                                                 ('ESHARD-AES128',
                                                  'ASCADv1 with fixed keys',
                                                  'ASCADv1 with variable keys',
                                                  'DPAv4.2',
                                                  'CHES CTF 2018'))
            if target_dataset_select == "ASCADv1 with fixed keys":
                target_dataset_name = "ascadf"
                target_dataset_filepath = os.path.join(datasets_path, ascadf_dataset)
                n_prof_target = 50000
                n_val_target = 5000
                n_attack_target = 5000
                ns_target = 10000
            if target_dataset_select == "ASCADv1 with variable keys":
                target_dataset_name = "ascadr"
                target_dataset_filepath = os.path.join(datasets_path, ascadr_dataset)
                n_prof_target = 200000
                n_val_target = 5000
                n_attack_target = 5000
                ns_target = 25000
            if target_dataset_select == "DPAv4.2":
                target_dataset_name = "dpav42"
                target_dataset_filepath = os.path.join(datasets_path, dpav42_dataset)
                n_prof_target = 70000
                n_val_target = 5000
                n_attack_target = 5000
                ns_target = 15000
            if target_dataset_select == 'ESHARD-AES128':
                target_dataset_name = "eshard"
                target_dataset_filepath = os.path.join(datasets_path, eshard_dataset)
                n_prof_target = 90000
                n_val_target = 5000
                n_attack_target = 5000
                ns_target = 1400
            if target_dataset_select == "CHES CTF 2018":
                target_dataset_name = "chesctf"
                target_dataset_filepath = os.path.join(datasets_path, chesctf_dataset)
                n_prof_target = 30000
                n_val_target = 5000
                n_attack_target = 5000
                ns_target = 15000

        if st.button("Prepare Target Dataset", key="prepare_target_dataset"):
            try:
                st.warning(f"Please wait while target dataset is being prepared...")
                st.session_state.target_dataset_name = target_dataset_name
                st.session_state.target_dataset = TargetLabels(target_dataset_name, n_prof_target, n_val_target,
                                                               n_attack_target, target_byte_target, leakage_model,
                                                               target_dataset_filepath)
                st.success(f"{target_dataset_name} dataset successfully prepared as a target dataset.")
            except Exception as e:
                st.error(f"Error preparing {target_dataset_name} as a target dataset.")

with tab5:
    with st.container(border=True):

        st.write("The architecture configurations below are defined based on a hyperparameter search, as explained in "
                 "the paper submission. When you select reference and target datasets in *Prepare* tab, the "
                 "hyperparemeters are automatically set.")

        best_archs = {
            "ascadf": {
                "ascadr": {
                    "generator": {'layers': 3, 'neurons': [200, 200, 100], 'activation': 'leakyrelu'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2,
                                      'layers_dropout': 1, 'dropout': 0.6}
                },
                "dpav42": {
                    "generator": {'layers': 2, 'neurons': [300, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 2,
                                      'layers_dropout': 2, 'dropout': 0.8}
                },
                "chesctf": {
                    "generator": {'layers': 4, 'neurons': [100, 100, 100, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2,
                                      'layers_dropout': 1, 'dropout': 0.5}
                },
                "eshard": {
                    "generator": {'layers': 2, 'neurons': [500, 400], 'activation': 'selu'},
                    "discriminator": {'neurons_embed': 500, 'neurons_dropout': 200, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.7}
                }
            },
            "ascadr": {
                "ascadf": {
                    "generator": {'layers': 1, 'neurons': [300], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 2,
                                      'layers_dropout': 3, 'dropout': 0.7}
                },
                "dpav42": {
                    "generator": {'layers': 4, 'neurons': [200, 200, 200, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.8}
                },
                "chesctf": {
                    "generator": {'layers': 4, 'neurons': [100, 100, 100, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2,
                                      'layers_dropout': 1, 'dropout': 0.5}
                },
                "eshard": {
                    "generator": {'layers': 3, 'neurons': [500, 500, 100], 'activation': 'leakyrelu'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.7}
                }
            },
            "dpav42": {
                "ascadf": {
                    "generator": {'layers': 4, 'neurons': [500, 100, 100, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.7}
                },
                "ascadr": {
                    "generator": {'layers': 1, 'neurons': [100], 'activation': 'elu'},
                    "discriminator": {'neurons_embed': 500, 'neurons_dropout': 100, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.6}
                },
                "chesctf": {
                    "generator": {'layers': 1, 'neurons': [100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.8}
                },
                "eshard": {
                    "generator": {'layers': 2, 'neurons': [400, 300], 'activation': 'selu'},
                    "discriminator": {'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.6}
                }
            },
            "eshard": {
                "ascadf": {
                    "generator": {'layers': 2, 'neurons': [400, 300], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.6}
                },
                "ascadr": {
                    "generator": {'layers': 4, 'neurons': [100, 100, 100, 100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.7}
                },
                "chesctf": {
                    "generator": {'layers': 1, 'neurons': [100], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1,
                                      'layers_dropout': 1, 'dropout': 0.8}
                },
                "dpav42": {
                    "generator": {'layers': 1, 'neurons': [5], 'activation': 'linear'},
                    "discriminator": {'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 2,
                                      'layers_dropout': 2, 'dropout': 0.8}
                }
            }
        }

        st.write("3. **Define CGAN architecture**")

        col_arch1, col_arch2 = st.columns(2, gap="medium")

        with col_arch1:

            with st.container(border=True):

                st.write("**Generator Architecture:**")

                generator_hp = {}

                hp_gen = best_archs[reference_dataset_name][target_dataset_name]["generator"]

                layers_gen = st.slider("Number of dense layers", min_value=1, max_value=6, value=hp_gen['layers'], )
                neurons_gen = np.zeros(layers_gen)
                for layer in range(layers_gen):
                    if layer < len(hp_gen['neurons']):
                        neurons_gen[layer] = st.slider(f"Number of neurons layer {layer + 1}", min_value=100,
                                                       max_value=500,
                                                       value=hp_gen['neurons'][layer])
                    else:
                        neurons_gen[layer] = st.slider(f"Number of neurons layer {layer + 1}", min_value=100,
                                                       max_value=500, value=100)

                activation = {
                    'leakyrelu': 0,
                    'relu': 1,
                    'selu': 2,
                    'tanh': 3,
                    'linear': 4
                }
                activation_function = st.selectbox('Activation Function',
                                                   ('leakyrelu', 'relu', 'selu', 'tanh', 'linear'),
                                                   index=activation[hp_gen['activation']])

                generator_hp["layers"] = layers_gen
                generator_hp["neurons"] = neurons_gen
                generator_hp["activation"] = activation_function

            discriminator_hp = {}

    with col_arch2:

        with st.container(border=True):
            st.write("**Discriminator Architecture:**")

            hp_disc = best_archs[reference_dataset_name][target_dataset_name]["discriminator"]

            layers_embed_disc = st.slider("Number of embedded layers", min_value=1, max_value=6,
                                          value=hp_disc['layers_embed'])
            for layer in range(layers_embed_disc):
                neurons_embed = st.slider(f"Number of neurons embedded layer {layer + 1}", min_value=100,
                                          max_value=500,
                                          value=hp_disc['neurons_embed'])

            layers_dropout_disc = st.slider("Number of dropout layers", min_value=1, max_value=6,
                                            value=hp_disc['layers_dropout'])
            for layer in range(layers_dropout_disc):
                neurons_dropout = st.slider(f"Number of neurons dropout layer {layer + 1}", min_value=100,
                                            max_value=500,
                                            value=hp_disc['neurons_dropout'])

            dropout = st.slider("Dropout rate", min_value=0.0, max_value=1.0, value=hp_disc['dropout'])

            discriminator_hp['layers_embedding'] = layers_embed_disc
            discriminator_hp['layers_dropout'] = layers_dropout_disc
            discriminator_hp['neurons_embedding'] = neurons_embed
            discriminator_hp['neurons_dropout'] = neurons_dropout
            discriminator_hp['dropout'] = dropout
            discriminator_hp['activation'] = "leakyrelu"

with tab6:
    st.write("4. **Train CGAN architecture**")

    with st.container(border=True):
        st.write("**Select the configurations for the CGAN training**")
        train_col1, train_col2, train_col3 = st.columns(3, gap="medium")
        with train_col1:
            batch_size = st.slider("Batch-size", min_value=100, max_value=1000, value=400)
        with train_col2:
            cgan_epochs = st.slider("Training Epochs", min_value=10, max_value=300, value=100)
        with train_col3:
            epochs_attack = st.slider("Run profiling attack for every E epochs:", min_value=1, max_value=cgan_epochs,
                                      value=50)

    with st.container(border=True):
        st.write("**Select the hyperparameters for the profiling attack model (MLP)**")
        prof_attack_model_col1, prof_attack_model_col2, prof_attack_model_col3 = st.columns(3, gap="medium")
        with prof_attack_model_col1:
            mlp_layers = st.slider("Dense Layers", min_value=1, max_value=4, value=1)
        with prof_attack_model_col2:
            mlp_neurons = st.slider("Neurons", min_value=10, max_value=400, value=100)
        with prof_attack_model_col3:
            mlp_activation = st.selectbox('Activation Function', ('leakyrelu', 'relu', 'selu', 'tanh',))

    max_snr_share_1 = []
    max_snr_share_2 = []

    ge_cgansca = []
    nt_cgansca = []
    pi_cgansca = []


    def discriminator_loss(real, fake):
        real_loss = cross_entropy_disc(tf.ones_like(real), real)
        fake_loss = cross_entropy_disc(tf.zeros_like(fake), fake)
        return real_loss + fake_loss


    def generator_loss(fake):
        return cross_entropy(tf.ones_like(fake), fake)


    def define_discriminator(features_dim, n_classes, hyperparameters):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        for l_i in range(hyperparameters["layers_embedding"]):
            y = Dense(hyperparameters["neurons_embedding"], kernel_initializer='random_normal')(y)
            y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = None
        for l_i in range(hyperparameters["layers_dropout"]):
            x = Dense(hyperparameters["neurons_dropout"], kernel_initializer='random_normal')(merge if l_i == 0 else x)
            x = LeakyReLU()(x)
            x = Dropout(hyperparameters["dropout"])(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model


    # define a random generator model
    def define_generator(input_dim, output_dim, hyperparameters):
        in_traces = Input(shape=(input_dim,))
        x = None
        for l_i in range(hyperparameters["layers"]):
            x = Dense(hyperparameters["neurons"][l_i],
                      activation=hyperparameters["activation"] if hyperparameters[
                                                                      "activation"] != "leakyrelu" else None)(
                in_traces if l_i == 0 else x)
            if hyperparameters["activation"] == "leakyrelu":
                x = LeakyReLU()(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model


    def generate_reference_samples(bs):
        rnd = np.random.randint(0, n_prof_ref - bs)
        features = st.session_state.reference_features[rnd:rnd + bs]
        labels = st.session_state.reference_labels[rnd:rnd + bs]
        return [features, labels]


    def get_target_batch(bs, rnd):
        in_file = h5py.File(target_dataset_filepath, "r")
        traces = np.array(in_file['Profiling_traces/traces'][rnd:rnd + bs])
        scaler = StandardScaler()
        return scaler.fit_transform(traces)


    def generate_target_samples(bs):
        rnd = np.random.randint(0, n_prof_target - bs)
        traces = get_target_batch(bs, rnd)
        labels = st.session_state.target_dataset.profiling_labels[rnd:rnd + bs]
        return [traces, labels]


    def compute_snr_reference_features(ref_snr_plot=None):
        batch_size_reference = 10000

        # prepare traces from target dataset
        rnd_reference = random.randint(0, n_prof_ref - batch_size_reference)
        features_reference = st.session_state.reference_features[rnd_reference:rnd_reference + batch_size_reference]

        snr_reference_features_share_1 = snr_fast(features_reference,
                                                  st.session_state.reference_share1[
                                                  target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])
        snr_reference_features_share_2 = snr_fast(features_reference,
                                                  st.session_state.reference_share2[
                                                  target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])

        with ref_snr_plot:
            df1 = pd.DataFrame({'name': "Share 1 (ref)", 'y': snr_reference_features_share_1})
            df2 = pd.DataFrame({'name': "Share 2 (ref)", 'y': snr_reference_features_share_2})
            df = pd.concat([df1, df2])

            fig = px.line(df, y='y', color='name', markers=True)
            fig.update_layout(xaxis_title='Features')
            fig.update_layout(yaxis_title='SNR')
            st.plotly_chart(fig)
        return snr_reference_features_share_1, snr_reference_features_share_2


    def compute_snr_target_features(epoch, epoch_snr_step, synthetic_traces=True, target_real_snr_plot=None,
                                    target_snr_plot=None, target_max_snr_plot=None):
        bs = 10000

        # prepare traces from target dataset
        rnd = random.randint(0, n_prof_target - bs)

        features_target = get_target_batch(bs, rnd)
        if synthetic_traces:
            features_target = generator.predict([features_target])

        snr_target_features_share_1 = snr_fast(features_target,
                                               st.session_state.target_dataset.share1_profiling[target_byte_target,
                                               rnd:rnd + bs]).tolist()
        snr_target_features_share_2 = snr_fast(features_target,
                                               st.session_state.target_dataset.share2_profiling[target_byte_target,
                                               rnd:rnd + bs]).tolist()

        if synthetic_traces:
            if (epoch + 1) % epoch_snr_step == 0:
                with target_snr_plot:
                    df1 = pd.DataFrame({'name': "Share 1 (target)", 'y': snr_target_features_share_1})
                    df2 = pd.DataFrame({'name': "Share 2 (target)", 'y': snr_target_features_share_2})
                    df = pd.concat([df1, df2])

                    fig = px.line(df, y='y', color='name', markers=True)
                    fig.update_layout(xaxis_title='Features')
                    fig.update_layout(yaxis_title='SNR')
                    st.plotly_chart(fig)

        else:
            with target_real_snr_plot:
                df1 = pd.DataFrame({'name': "Share 1 (target real)", 'y': snr_target_features_share_1})
                df2 = pd.DataFrame({'name': "Share 2 (target real)", 'y': snr_target_features_share_2})
                df = pd.concat([df1, df2])

                fig = px.line(df, y='y', color='name', markers=True)
                fig.update_layout(xaxis_title='Features')
                fig.update_layout(yaxis_title='SNR')
                st.plotly_chart(fig)

        if synthetic_traces:
            max_snr_share_1.append(np.max(snr_target_features_share_1))
            max_snr_share_2.append(np.max(snr_target_features_share_2))
            if (epoch + 1) % epoch_snr_step == 0:
                with target_max_snr_plot:
                    df1 = pd.DataFrame({'name': "Max SNR Share 1", 'y': max_snr_share_1})
                    df2 = pd.DataFrame({'name': "Max SNR Share 2", 'y': max_snr_share_2})
                    df = pd.concat([df1, df2])

                    fig = px.line(df, y='y', color='name', markers=True)
                    fig.update_layout(xaxis_title='CGAN Training Epochs')
                    fig.update_layout(yaxis_title='SNR')
                    st.plotly_chart(fig)
        return snr_target_features_share_1, snr_target_features_share_2


    def attack_eval(ge_plot, progress_bar_features_extraction, progress_bar_epochs,
                    mlp_layers, mlp_neurons, mlp_activation):

        ge, nt, pi, ge_vector = attack(st.session_state.target_dataset, target_dataset_filepath, generator,
                                       cgan_features, mlp_layers, mlp_neurons, mlp_activation,
                                       progress_bar_features_extraction=progress_bar_features_extraction,
                                       progress_bar_epochs=progress_bar_epochs)
        ge_cgansca.append(ge)
        nt_cgansca.append(nt)
        pi_cgansca.append(pi)

        with ge_plot:
            df1 = pd.DataFrame({'name': "Guessing Entropy", 'y': ge_vector})
            df = pd.concat([df1])

            fig = px.line(df, y='y', color='name', markers=True)
            fig.update_layout(xaxis_title='Attack Traces')
            fig.update_layout(yaxis_title='Guessing Entropy')
            st.plotly_chart(fig)

        return ge, nt, pi, ge_vector


    # %%
    @tf.function
    def train_step(traces_batch, label_traces, features, label_features):
        with tf.GradientTape() as disc_tape:
            fake_features = generator(traces_batch)
            real_output = discriminator([label_features, features])
            fake_output = discriminator([label_traces, fake_features])
            disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_features = generator(traces_batch)
            fake_output = discriminator([label_traces, fake_features])
            gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        fake_accuracy_metric.update_state(tf.zeros_like(fake_features), fake_output)
        real_accuracy_metric.update_state(tf.ones_like(features), real_output)

        return gen_loss, disc_loss


    if st.button("Train CGAN & Key Recovery"):

        if st.session_state.reference_features is not None and st.session_state.target_dataset is not None:

            st.session_state.db_attack = db_class.insert_data("attacks",
                                                              {
                                                                  "reference": reference_dataset_name,
                                                                  "target": target_dataset_name,
                                                                  "leakage_model": leakage_model,
                                                                  "reference_key_byte": target_byte_reference,
                                                                  "target_key_byte": target_byte_target,
                                                                  "cgan_features": cgan_features,
                                                              })

            generator_values = {}
            generator_values["layers"] = generator_hp["layers"]
            for n_i, gen_neurons in enumerate(generator_hp["neurons"]):
                generator_values[f"neurons_{n_i + 1}"] = gen_neurons
            generator_values["activation"] = generator_hp["activation"]
            generator_values["attack_id"] = st.session_state.db_attack.id

            db_class.insert_data("generators", generator_values)

            discriminator_hp["attack_id"] = st.session_state.db_attack.id

            db_class.insert_data("discriminators", discriminator_hp)

            db_class.insert_data("trainings", {"batch_size": batch_size, "epochs": cgan_epochs,
                                               "attack_id": st.session_state.db_attack.id})
            db_class.insert_data("profilings",
                                 {"batch_size": 400, "epochs": 100, "layers": mlp_layers, "neurons": mlp_neurons,
                                  "activation": mlp_activation, "attack_id": st.session_state.db_attack.id})

            real_accuracy_metric = BinaryAccuracy()
            fake_accuracy_metric = BinaryAccuracy()
            cross_entropy = BinaryCrossentropy(from_logits=True)
            cross_entropy_disc = BinaryCrossentropy(from_logits=True)
            generator_optimizer = ks.optimizers.Adam(0.0002, beta_1=0.5)
            discriminator_optimizer = ks.optimizers.Adam(0.0025, beta_1=0.5)

            classes = 9 if leakage_model == "HW" else 256

            # create the discriminator
            discriminator = define_discriminator(cgan_features, classes, discriminator_hp)
            # create the generator
            generator = define_generator(ns_target, cgan_features, generator_hp)

            training_set_size = max(n_prof_ref, n_prof_target)

            # determine half the size of one batch, for updating the discriminator
            n_batches = int(training_set_size / batch_size)

            epoch_snr_step = 1

            real_acc = []
            fake_acc = []

            g_loss = []
            d_loss = []

            ge_cgan_epochs = []
            nt_cgan_epochs = []
            pi_cgan_epochs = []
            cgan_training_attack_metric = []

            with st.container(border=True):

                progress_bar_training = st.progress(0)

                acc_col, loss_col = st.columns(2)

                with acc_col:
                    st.write("Discriminator Accuracy:")
                    acc_plot = st.empty()
                with loss_col:
                    st.write("Generator and Discriminator Loss:")
                    loss_plot = st.empty()

            with st.container(border=True):

                snr_col1, snr_col2 = st.columns(2)

                with snr_col1:
                    st.write("SNR target features (generated by CGAN generator):")
                    target_snr_plot = st.empty()
                with snr_col2:
                    st.write("SNR max target features (generated by CGAN generator):")
                    target_max_snr_plot = st.empty()

            with st.container(border=True):

                snr_col1, snr_col2 = st.columns(2)

                with snr_col1:
                    st.write("SNR reference features:")
                    ref_snr_plot = st.empty()
                with snr_col2:
                    st.write("SNR real target features:")
                    target_real_snr_plot = st.empty()

            with st.container(border=True):
                ref_col1, ref_col2 = st.columns(2)
                with ref_col1:
                    st.write("Guessing Entropy:")
                    ge_plot = st.empty()
                with ref_col2:
                    st.write("Key Recovery Information:")
                    progress_bar_features_extraction = st.empty()
                    profiling_attack_progress_bar = st.empty()
                    key_recovery_info_ge = st.empty()
                    key_recovery_info_nt = st.empty()
                    key_recovery_info_pi = st.empty()

                ge_epochs_col1, nt_epochs_col1 = st.columns(2, gap="medium")
                with ge_epochs_col1:
                    st.write("Guessing Entropy vs CGAN Training Epochs:")
                    ge_plot_cgan_epochs = st.empty()
                with nt_epochs_col1:
                    st.write("NT (GE = 1) vs CGAN Training Epochs:")
                    nt_plot_cgan_epochs = st.empty()
                st.write("PI vs CGAN Training Epochs:")
                pi_plot_cgan_epochs = st.empty()

            total_iterations = cgan_epochs * n_batches

            results_columns = {}
            results_columns["accuracy_ref"] = pd.Series(real_acc).to_json()
            results_columns["accuracy_target"] = pd.Series(fake_acc).to_json()
            results_columns["loss_generator"] = pd.Series(g_loss).to_json()
            results_columns["loss_discriminator"] = pd.Series(d_loss).to_json()
            results_columns["snr_ref"] = pd.Series(
                [np.zeros(cgan_features), np.zeros(cgan_features)]).to_json()
            results_columns["snr_target_real"] = pd.Series(
                [np.zeros(cgan_features), np.zeros(cgan_features)]).to_json()
            results_columns["snr_target"] = pd.Series(
                [np.zeros(cgan_features), np.zeros(cgan_features)]).to_json()
            results_columns["snr_share1"] = pd.Series(max_snr_share_1).to_json()
            results_columns["snr_share2"] = pd.Series(max_snr_share_2).to_json()
            results_columns["ge_traces"] = pd.Series(np.zeros(st.session_state.target_dataset.n_attack)).to_json()
            results_columns["ge_epochs"] = pd.Series(ge_cgan_epochs).to_json()
            results_columns["nt_epochs"] = pd.Series(nt_cgan_epochs).to_json()
            results_columns["pi_epochs"] = pd.Series(pi_cgan_epochs).to_json()
            results_columns["attack_epochs_interval"] = epochs_attack
            results_columns["attack_id"] = st.session_state.db_attack.id

            db_class.insert_data("results", results_columns)

            snr_reference_features_share_1, snr_reference_features_share_2 = [], []
            snr_target_real_share_1, snr_target_real_share_2 = [], []
            snr_target_syn_share_1, snr_target_syn_share_2 = [], []
            ge_vector = np.zeros(st.session_state.target_dataset.n_attack)

            for e in range(cgan_epochs):
                for b in range(n_batches):

                    start = time()

                    [features_reference, labels_reference] = generate_reference_samples(batch_size)
                    [traces_target, labels_target] = generate_target_samples(batch_size)

                    gen_loss, disc_loss = train_step(traces_target, labels_target, features_reference, labels_reference)

                    if (b + 1) % 100 == 0:
                        real_acc.append(real_accuracy_metric.result().numpy())
                        fake_acc.append(fake_accuracy_metric.result().numpy())
                        g_loss.append(float(gen_loss))
                        d_loss.append(float(disc_loss))

                        with acc_plot:
                            df1 = pd.DataFrame({'name': "Accuracy Reference", 'y': real_acc})
                            df2 = pd.DataFrame({'name': "Accuracy Target", 'y': fake_acc})
                            df = pd.concat([df1, df2])

                            fig = px.line(df, y='y', color='name', markers=True)
                            fig.update_layout(yaxis_title='Accuracy')
                            fig.update_layout(xaxis_title='CGAN Training Batches / 100')
                            st.plotly_chart(fig)

                        with loss_plot:
                            df1 = pd.DataFrame({'name': "Loss", 'y': d_loss})
                            df2 = pd.DataFrame({'name': "Generator Loss", 'y': g_loss})
                            df = pd.concat([df1, df2])

                            fig = px.line(df, y='y', color='name', markers=True)
                            fig.update_layout(yaxis_title='Loss')
                            fig.update_layout(xaxis_title='CGAN Training Batches / 100')
                            st.plotly_chart(fig)

                    batch_time = time() - start

                    progress_count = ((e * n_batches) + b) / total_iterations
                    estimated_time = batch_time * (total_iterations - ((e * n_batches) + b))
                    datetime_obj = datetime.fromtimestamp(estimated_time)
                    formatted_time = datetime_obj.strftime('%H:%M:%S')
                    text = f"{round(progress_count * 100, 2)}% of CGAN training completed (estimated time to finish: {formatted_time} | batches to finish: {total_iterations - ((e * n_batches) + b)})"
                    progress_bar_training.progress(progress_count, text=text)

                if e == 0:
                    snr_reference_features_share_1, snr_reference_features_share_2 = compute_snr_reference_features(
                        ref_snr_plot=ref_snr_plot)
                    snr_target_real_share_1, snr_target_real_share_2 = compute_snr_target_features(e,
                                                                                                   epoch_snr_step,
                                                                                                   synthetic_traces=False,
                                                                                                   target_real_snr_plot=target_real_snr_plot)
                else:
                    snr_target_syn_share_1, snr_target_syn_share_2 = compute_snr_target_features(e, epoch_snr_step,
                                                                                                 target_real_snr_plot=target_real_snr_plot,
                                                                                                 target_snr_plot=target_snr_plot,
                                                                                                 target_max_snr_plot=target_max_snr_plot)
                if (e + 1) % epochs_attack == 0:
                    st.progress(0)

                    ge, nt, pi, ge_vector = attack_eval(ge_plot, progress_bar_features_extraction,
                                                        profiling_attack_progress_bar,
                                                        mlp_layers, mlp_neurons, mlp_activation)

                    ge_cgan_epochs.append(ge)
                    nt_cgan_epochs.append(nt)
                    pi_cgan_epochs.append(pi)

                    cgan_training_attack_metric.append(e + 1)

                    with key_recovery_info_ge:
                        st.write(f"Guessing entropy: {ge}")
                    with key_recovery_info_nt:
                        st.write(f"Number of attack traces for GE = 1: {nt}")
                    with key_recovery_info_pi:
                        st.write(f"Perceived Informarion: {pi}")

                    with ge_plot_cgan_epochs:
                        df1 = pd.DataFrame(
                            {'x': cgan_training_attack_metric, 'y': ge_cgan_epochs, 'name': "Guessing Entropy"})
                        df = pd.concat([df1])

                        fig = px.line(df, x='x', y='y', color='name', markers=True)
                        fig.update_layout(xaxis_title='CGAN Training Epochs')
                        fig.update_layout(yaxis_title='Guessing Entropy')

                        st.plotly_chart(fig)
                    with nt_plot_cgan_epochs:
                        df1 = pd.DataFrame(
                            {'x': cgan_training_attack_metric, 'y': nt_cgan_epochs,
                             'name': "Number of Attack Traces for GE = 1"})
                        df = pd.concat([df1])

                        fig = px.line(df, x='x', y='y', color='name', markers=True)
                        fig.update_layout(xaxis_title='CGAN Training Epochs')
                        fig.update_layout(yaxis_title='Number of Attack Traces for GE = 1')
                        st.plotly_chart(fig)
                    with pi_plot_cgan_epochs:
                        df1 = pd.DataFrame(
                            {'x': cgan_training_attack_metric, 'y': pi_cgan_epochs,
                             'name': "Perceived Information"})
                        df = pd.concat([df1])

                        fig = px.line(df, x='x', y='y', color='name', markers=True)
                        fig.update_layout(xaxis_title='CGAN Training Epochs')
                        fig.update_layout(yaxis_title='Perceived Information')
                        st.plotly_chart(fig)

                results_columns = {}
                results_columns["accuracy_ref"] = pd.Series(real_acc).to_json()
                results_columns["accuracy_target"] = pd.Series(fake_acc).to_json()
                results_columns["loss_generator"] = pd.Series(g_loss).to_json()
                results_columns["loss_discriminator"] = pd.Series(d_loss).to_json()
                results_columns["snr_ref"] = pd.Series(
                    [snr_reference_features_share_1, snr_reference_features_share_1]).to_json()
                results_columns["snr_target_real"] = pd.Series(
                    [snr_target_real_share_1, snr_target_real_share_2]).to_json()
                results_columns["snr_target"] = pd.Series(
                    [snr_target_syn_share_1, snr_target_syn_share_2]).to_json()
                results_columns["snr_share1"] = pd.Series(max_snr_share_1).to_json()
                results_columns["snr_share2"] = pd.Series(max_snr_share_2).to_json()
                results_columns["ge_traces"] = pd.Series(ge_vector).to_json()
                results_columns["ge_epochs"] = pd.Series(ge_cgan_epochs).to_json()
                results_columns["nt_epochs"] = pd.Series(nt_cgan_epochs).to_json()
                results_columns["pi_epochs"] = pd.Series(pi_cgan_epochs).to_json()
                results_columns["attack_epochs_interval"] = epochs_attack
                results_columns["attack_id"] = st.session_state.db_attack.id

                db_class.update_data("results", [("attack_id", st.session_state.db_attack.id)], results_columns)
        else:
            st.error(f"Reference and target datasets not selected.")

with tab7:
    st.write("**All your results**")
    with st.container(border=True):
        db_attacks = db_class.select_data("attacks")
        select_attack_options = []
        for db_attack in db_attacks:
            select_attack_options.append(f"{db_attack.reference} vs {db_attack.target} ({db_attack.datetime})")
        st.selectbox('Analyses', select_attack_options)
