import numpy as np
import bz2
import h5py
from tqdm import tqdm
from numba import njit


@njit
def winres(trace, window=20, overlap=0.5):
    trace_winres = []
    step = int(window * overlap)
    max = len(trace)
    for i in range(0, max, step):
        trace_winres.append(np.mean(trace[i:i + window]))
    return np.array(trace_winres)


def generate_nopoi_dpav42(datasets_path, filepath_destination, n_traces, n_samples, progress_bar, window=20):
    filepath_data = f"{datasets_path}/dpav4_2_index.txt"

    file_data = open(filepath_data, "r")
    file_lines = file_data.readlines()

    fs = 250000
    ns = 150000

    mask_vector = [3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252]

    out_file = h5py.File(filepath_destination, 'w')

    n_traces_profiling = 70000
    n_traces_attack = 10000

    samples = np.zeros((n_traces_profiling, 15000))
    plaintexts = np.zeros((n_traces_profiling, 16))
    ciphertexts = np.zeros((n_traces_profiling, 16))
    masks = np.zeros((n_traces_profiling, 16))
    keys = np.zeros((n_traces_profiling, 16))

    for file_index in range(14):

        for p, i in enumerate(range(5000 * file_index, 5000 * file_index + 5000)):
            line = file_lines[i]

            bz2_file_name = "DPACV42_{}".format(str(i).zfill(6))
            filepath = f"{datasets_path}/DPA_contestv4_2/k{str(file_index).zfill(2)}/{bz2_file_name}.trc.bz2"
            data = bz2.BZ2File(filepath).read()  # get the decompressed data

            samples[i] = winres(np.array(np.frombuffer(data[357:len(data) - 357], dtype='int8')[fs: fs + ns]))

            keys[i] = np.frombuffer(bytearray.fromhex(line[0:32]), np.uint8)
            plaintexts[i] = np.frombuffer(bytearray.fromhex(line[33:65]), np.uint8)
            ciphertexts[i] = np.frombuffer(bytearray.fromhex(line[66:98]), np.uint8)

            offset3 = [int(s, 16) for s in line[133:149]]
            for b in range(16):
                masks[i][b] = int(mask_vector[int(offset3[b] + 1) % 16])

            progress_bar.progress(p / 5000, text=f"{p}/5000 traces processed in {filepath}")

    trace_group = out_file.create_group("Profiling_traces")
    trace_group.create_dataset(name="traces", data=samples, dtype=samples.dtype)
    metadata_type_profiling = np.dtype([("plaintext", plaintexts.dtype, (len(plaintexts[0]),)),
                                        ("ciphertext", ciphertexts.dtype, (len(ciphertexts[0]),)),
                                        ("masks", masks.dtype, (len(masks[0]),)),
                                        ("key", keys.dtype, (len(keys[0]),))
                                        ])
    trace_index = [n for n in range(0, n_traces_profiling)]
    profiling_metadata = np.array([(plaintexts[n], ciphertexts[n], masks[n], keys[n]) for n, k in
                                   zip(trace_index, range(0, len(samples)))], dtype=metadata_type_profiling)
    trace_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    samples = np.zeros((n_traces_attack, 15000))
    plaintexts = np.zeros((n_traces_attack, 16))
    ciphertexts = np.zeros((n_traces_attack, 16))
    masks = np.zeros((n_traces_attack, 16))
    keys = np.zeros((n_traces_attack, 16))

    for file_index in range(14, 16):

        for p, i in enumerate(range(5000 * file_index, 5000 * file_index + 5000)):
            line = file_lines[i]

            bz2_file_name = "DPACV42_{}".format(str(i).zfill(6))
            filepath = f"{datasets_path}/DPA_contestv4_2/k{str(file_index).zfill(2)}/{bz2_file_name}.trc.bz2"
            data = bz2.BZ2File(filepath).read()  # get the decompressed data

            samples[i - 70000] = winres(np.array(np.frombuffer(data[357:len(data) - 357], dtype='int8')[fs: fs + ns]))

            keys[i - 70000] = np.frombuffer(bytearray.fromhex(line[0:32]), np.uint8)
            plaintexts[i - 70000] = np.frombuffer(bytearray.fromhex(line[33:65]), np.uint8)
            ciphertexts[i - 70000] = np.frombuffer(bytearray.fromhex(line[66:98]), np.uint8)

            offset3 = [int(s, 16) for s in line[133:149]]
            for b in range(16):
                masks[i - 70000][b] = int(mask_vector[int(offset3[b] + 1) % 16])

            progress_bar.progress(p / 5000, text=f"{p}/5000 traces processed in {filepath}")

    trace_group = out_file.create_group("Attack_traces")
    trace_group.create_dataset(name="traces", data=samples, dtype=samples.dtype)
    metadata_type_attack = np.dtype([("plaintext", plaintexts.dtype, (len(plaintexts[0]),)),
                                     ("ciphertext", ciphertexts.dtype, (len(ciphertexts[0]),)),
                                     ("masks", masks.dtype, (len(masks[0]),)),
                                     ("key", keys.dtype, (len(keys[0]),))
                                     ])
    trace_index = [n for n in range(0, n_traces_attack)]
    attack_metadata = np.array([(plaintexts[n], ciphertexts[n], masks[n], keys[n]) for n, k in
                                zip(trace_index, range(0, len(samples)))], dtype=metadata_type_attack)
    trace_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)
    out_file.flush()
    out_file.close()
