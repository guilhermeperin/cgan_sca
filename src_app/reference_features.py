import gc

import h5py
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from src.aes import *
from src.stats import *


class ReferenceFeatures:

    def __init__(self, dataset_name, dataset_path, filepath, n_samples, n_features, n_traces, target_key_byte,
                 laekage_model, ns_chunk_size=10000):

        if os.path.exists(f"reference_features_{dataset_name}.npz"):
            return

        self.datasets_path = dataset_path

        self.dataset_name = dataset_name
        self.filepath = filepath
        self.n_features = n_features
        self.n_traces = n_traces
        self.target_key_byte = target_key_byte
        self.leakage_model = laekage_model

        # self.plaintexts = np.zeros((n_traces, 16))
        # self.keys = np.zeros((n_traces, 16))
        # self.masks = np.zeros((n_traces, 18))
        self.share1 = None
        self.share2 = None
        self.labels = None

        self.create_metadata()

        progress_bar = st.progress(0)

        n_chunks = int(n_samples / ns_chunk_size)

        snr_chunks_share1 = np.zeros((n_chunks, ns_chunk_size))
        snr_chunks_share2 = np.zeros((n_chunks, ns_chunk_size))

        chunk_process_status = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            snr_chunk = st.empty()
        with col2:
            snr_all_chunks = st.empty()

        snr_shares = st.empty()

        for ns_i, ns in enumerate(range(0, n_samples, ns_chunk_size)):
            with chunk_process_status:
                st.write(f"Processing trace samples from {ns} to {ns + ns_chunk_size} (total samples: {n_samples})")

            snr_chunk_share1, snr_chunk_share2 = self.get_features_per_chunk(ns, ns_chunk_size, progress_bar)

            snr_chunks_share1[ns_i] = snr_chunk_share1
            snr_chunks_share2[ns_i] = snr_chunk_share2

            with snr_chunk:
                df1 = pd.DataFrame({'name': "Share 1", 'y': snr_chunks_share1.flatten()})
                df2 = pd.DataFrame({'name': "Share 2", 'y': snr_chunks_share2.flatten()})
                df = pd.concat([df1, df2])

                fig = px.line(df, y='y', color='name', markers=True)
                fig.update_layout(xaxis_title='Features')
                fig.update_layout(yaxis_title='SNR')
                st.plotly_chart(fig)

            with snr_all_chunks:
                df1 = pd.DataFrame({'name': "Share 1", 'y': snr_chunk_share1})
                df2 = pd.DataFrame({'name': "Share 2", 'y': snr_chunk_share2})
                df = pd.concat([df1, df2])

                fig = px.line(df, y='y', color='name', markers=True)
                fig.update_layout(xaxis_title='Features')
                fig.update_layout(yaxis_title='SNR')
                st.plotly_chart(fig)

        np.savez(os.path.join(self.datasets_path, f'{self.dataset_name}_snr_chunks_share1_{self.leakage_model}.npz'),
                 snr_chunks_share1=snr_chunks_share1.flatten())
        np.savez(os.path.join(self.datasets_path, f'{self.dataset_name}_snr_chunks_share2_{self.leakage_model}.npz'),
                 snr_chunks_share1=snr_chunks_share1.flatten())

        reference_features = self.get_reference_features(filepath, snr_chunks_share1.flatten(),
                                                         snr_chunks_share2.flatten())

        snr_share1 = snr_fast(np.array(reference_features, dtype=np.int16),
                              np.asarray(self.share1[self.target_key_byte, :]))
        snr_share2 = snr_fast(np.array(reference_features, dtype=np.int16),
                              np.asarray(self.share2[self.target_key_byte, :]))

        with snr_shares:
            df1 = pd.DataFrame({'name': "Share 1", 'y': snr_share1})
            df2 = pd.DataFrame({'name': "Share 2", 'y': snr_share2})
            df = pd.concat([df1, df2])

            fig = px.line(df, y='y', color='name', markers=True)
            st.plotly_chart(fig)

        np.savez(os.path.join(self.datasets_path, f'reference_features_{self.dataset_name}_{self.leakage_model}.npz'),
                 reference_features=reference_features)

    def create_metadata(self):
        in_file = h5py.File(self.filepath, "r")

        if self.dataset_name == "ascadf" or self.dataset_name == "ascadr":
            h5_metadata = "metadata"
        else:
            h5_metadata = "Profiling_traces/metadata"

        raw_plaintexts = in_file[h5_metadata]['plaintext'][:self.n_traces]
        raw_keys = in_file[h5_metadata]['key'][:self.n_traces]
        raw_masks = in_file[h5_metadata]['masks'][:self.n_traces]

        # progress_bar = st.progress(0)
        # st.write("Retrieving metadata (plaintexts, keys, masks)")
        # for i in range(self.n_traces):
        #     self.plaintexts[i] = raw_plaintexts[i]
        #     self.keys[i] = raw_keys[i]
        #     self.masks[i] = raw_masks[i]
        #     progress_bar.progress(i / self.n_traces, text=f"{round((i / self.n_traces) * 100, 2)}%")

        # self.share1, self.share2 = create_intermediates(self.plaintexts, self.masks, self.keys, self.n_traces,
        #                                                 leakage_model=self.leakage_model)
        # self.labels = aes_labelize(self.plaintexts, self.keys, self.target_key_byte, leakage_model=self.leakage_model)
        self.share1, self.share2 = create_intermediates(raw_plaintexts, raw_masks, raw_keys, self.n_traces,
                                                        leakage_model=self.leakage_model, dataset=self.dataset_name)
        self.labels = aes_labelize(raw_plaintexts, raw_keys, self.target_key_byte, leakage_model=self.leakage_model)

        np.savez(os.path.join(self.datasets_path, f'{self.dataset_name}_share1_{self.leakage_model}.npz'),
                 share1=self.share1)
        np.savez(os.path.join(self.datasets_path, f'{self.dataset_name}_share2_{self.leakage_model}.npz'),
                 share2=self.share2)
        np.savez(os.path.join(self.datasets_path, f'{self.dataset_name}_labels_{self.leakage_model}.npz'),
                 labels=self.labels)

    def get_features_per_chunk(self, ns_start, ns_chunk_size, progress_bar):

        in_file = h5py.File(f'{self.filepath}', "r")

        samples = np.zeros((self.n_traces, ns_chunk_size), dtype=np.int8)

        if self.dataset_name == "ascadf" or self.dataset_name == "ascadr":
            h5_traces = "traces"
        else:
            h5_traces = "Profiling_traces/traces"

        for i in range(self.n_traces):
            samples[i] = in_file[h5_traces][i, ns_start:ns_start + ns_chunk_size]
            progress_bar.progress(i / self.n_traces, text=f"{round((i / self.n_traces) * 100, 2)}%")

        snr_chunk_share1 = snr_fast(np.array(samples, dtype=np.int16),
                                    np.asarray(self.share1[self.target_key_byte, :]))
        snr_chunk_share2 = snr_fast(np.array(samples, dtype=np.int16),
                                    np.asarray(self.share2[self.target_key_byte, :]))

        del samples
        gc.collect()

        return snr_chunk_share1, snr_chunk_share2

    def get_reference_features(self, raw_dataset_file, snr_share1, snr_share2):
        snr_share1[np.isnan(snr_share1)] = 0
        snr_share2[np.isnan(snr_share2)] = 0
        ind_snr_masks_poi_sm = np.argsort(snr_share1)[::-1][:int(self.n_features / 2)]
        ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
        ind_snr_masks_poi_r2 = np.argsort(snr_share2)[::-1][:int(self.n_features / 2)]
        ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

        poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

        prof_samples = np.zeros((self.n_traces, self.n_features), dtype=np.int8)

        if self.dataset_name == "ascadf" or self.dataset_name == "ascadr":
            h5_traces = "traces"
        else:
            h5_traces = "Profiling_traces/traces"

        in_file = h5py.File(raw_dataset_file, "r")
        for i in range(self.n_traces):
            trace = np.array(in_file[h5_traces][i])
            prof_samples[i] = trace[poi_profiling]

        return prof_samples
