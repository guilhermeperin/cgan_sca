import numpy as np
from numba import njit
import trsfile
import random
import h5py


@njit
def winres(trace, window=20, overlap=0.5):
    trace_winres = []
    step = int(window * overlap)
    max = len(trace)
    for i in range(0, max, step):
        trace_winres.append(np.mean(trace[i:i + window]))
    return np.array(trace_winres)


def load_trs_trace(progress_bar_chesctf, path, filename, number_of_traces, number_of_samples, data_length,
                   number_of_samples_resampled=None, window=20, desync=False):
    samples = np.zeros((number_of_traces,
                        number_of_samples_resampled if number_of_samples_resampled is not None else number_of_samples),
                       dtype=np.float16)
    plaintexts = np.zeros((number_of_traces, 16), dtype=np.uint8)
    ciphertexts = np.zeros((number_of_traces, 16), dtype=np.uint8)
    keys = np.zeros((number_of_traces, 16), dtype=np.uint8)

    """ The second file contains traces shifted by 800 samples in acquisition phase """
    if filename == f"{path}/PinataAcqTask2.2_10k_upload.trs":
        sample_offset = 800
    else:
        sample_offset = 0

    trace_file = trsfile.open(filename)
    for i, trace in enumerate(trace_file[:number_of_traces]):

        if number_of_samples_resampled is not None:

            if desync:
                trace_tmp = abs(trace[sample_offset:sample_offset + number_of_samples])
                trace_tmp_shifted = np.zeros(number_of_samples)
                shift = random.randint(-50, 50)
                if shift > 0:
                    trace_tmp_shifted[0:number_of_samples - shift] = trace_tmp[shift:number_of_samples]
                    trace_tmp_shifted[number_of_samples - shift:number_of_samples] = trace_tmp[0:shift]
                else:
                    trace_tmp_shifted[0:abs(shift)] = trace_tmp[number_of_samples - abs(shift):number_of_samples]
                    trace_tmp_shifted[abs(shift):number_of_samples] = trace_tmp[0:number_of_samples - abs(shift)]
                trace_tmp = trace_tmp_shifted
            else:
                trace_tmp = abs(trace[sample_offset:sample_offset + number_of_samples])

            samples[i] = winres(trace_tmp, window=window)
        else:
            samples[i] = trace[sample_offset:sample_offset + number_of_samples]

        plaintexts[i] = np.frombuffer(trace.data, dtype=np.uint8, count=data_length)[:16]
        ciphertexts[i] = np.frombuffer(trace.data, dtype=np.uint8, count=data_length)[16:32]
        keys[i] = np.frombuffer(trace.data, dtype=np.uint8, count=data_length)[32:]
        text = f"{i} traces processed (plaintext: {plaintexts[i]} - ciphertext: {ciphertexts[i]} - key: {keys[i]})"
        progress_bar_chesctf.progress(i / number_of_traces, text=text)

    return samples, plaintexts, ciphertexts, keys


def generate_chestf_nopoi(path, progress_bar_chesctf, window=20):
    n_traces_file = 10000
    n_profiling = 30000
    n_attack = 10000
    number_of_samples = int(150000 / window) * 2
    profiling_samples = np.zeros((n_profiling, number_of_samples), dtype=np.float16)
    profiling_plaintexts = np.zeros((n_profiling, 16))
    profiling_ciphertexts = np.zeros((n_profiling, 16))
    profiling_keys = np.zeros((n_profiling, 16))

    samples, plaintexts, ciphertexts, keys = load_trs_trace(
        progress_bar_chesctf,
        path, f"{path}/PinataAcqTask2.1_10k_upload.trs", n_traces_file,
        150000, 48, number_of_samples_resampled=number_of_samples, window=window)
    profiling_samples[:n_traces_file] = samples
    profiling_plaintexts[:n_traces_file] = plaintexts
    profiling_ciphertexts[:n_traces_file] = ciphertexts
    profiling_keys[:n_traces_file] = keys

    samples, plaintexts, ciphertexts, keys = load_trs_trace(
        progress_bar_chesctf,
        path, f"{path}/PinataAcqTask2.2_10k_upload.trs", n_traces_file,
        150000, 48, number_of_samples_resampled=number_of_samples, window=window)
    profiling_samples[n_traces_file:n_traces_file * 2] = samples
    profiling_plaintexts[n_traces_file:n_traces_file * 2] = plaintexts
    profiling_ciphertexts[n_traces_file:n_traces_file * 2] = ciphertexts
    profiling_keys[n_traces_file:n_traces_file * 2] = keys

    samples, plaintexts, ciphertexts, keys = load_trs_trace(
        progress_bar_chesctf,
        path, f"{path}/PinataAcqTask2.3_10k_upload.trs", n_traces_file,
        150000, 48, number_of_samples_resampled=number_of_samples, window=window)
    profiling_samples[n_traces_file * 2:n_traces_file * 3] = samples
    profiling_plaintexts[n_traces_file * 2:n_traces_file * 3] = plaintexts
    profiling_ciphertexts[n_traces_file * 2:n_traces_file * 3] = ciphertexts
    profiling_keys[n_traces_file * 2:n_traces_file * 3] = keys

    attack_samples, attack_plaintexts, attack_ciphertexts, attack_keys = load_trs_trace(
        progress_bar_chesctf,
        path, f"{path}/PinataAcqTask2.4_10k_upload.trs", n_traces_file,
        150000, 48, number_of_samples_resampled=number_of_samples, window=window)

    out_file = h5py.File(f'{path}/ches_ctf_nopoi_window_{window}.h5', 'w')

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts[0].dtype, (16,)),
                                        ("ciphertext", profiling_ciphertexts[0].dtype, (16,)),
                                        ("key", profiling_keys[0].dtype, (16,))])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintexts[0].dtype, (16,)),
                                     ("ciphertext", attack_ciphertexts[0].dtype, (16,)),
                                     ("key", attack_keys[0].dtype, (16,))])

    profiling_metadata = np.array(
        [(profiling_plaintexts[n], profiling_ciphertexts[n], profiling_keys[n]) for n in range(n_profiling)],
        dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintexts[n], attack_ciphertexts[n], attack_keys[n]) for n in range(n_attack)],
                               dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()
