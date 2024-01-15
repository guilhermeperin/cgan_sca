import numpy as np
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


def generate_nopoi_ascadr(filepath, filepath_destination, n_traces, n_samples, progress_bar, window=20):
    n_profiling = 200000
    n_attack = 10000
    n_attack_total = 100000

    profiling_index = [n for n in range(0, n_profiling + n_attack_total) if n % 3 != 2]
    attack_index = [n for n in range(2, n_profiling + n_attack_total, 3)]

    in_file = h5py.File(f'{filepath}', "r")
    metadata = in_file["metadata"]
    raw_plaintexts = metadata['plaintext']
    raw_keys = metadata['key']
    raw_masks = metadata['masks']

    ns = int(n_samples / window) * 2

    profiling_samples = np.zeros((n_profiling, ns), dtype=np.int8)
    attack_samples = np.zeros((n_attack, ns), dtype=np.int8)

    print(f"Retrieving profiling traces from index {profiling_index[:20]}")
    for i, j in tqdm(enumerate(profiling_index)):
        profiling_samples[i] = winres(in_file["traces"][j], window=window)
        progress_bar.progress(i / n_profiling, text=f"{i + 1}/{n_profiling} profiling traces processed")

    print(f"Retrieving attack traces from index {attack_index[:20]}")
    for i, j in tqdm(enumerate(attack_index)):
        if i == 10000:
            break
        attack_samples[i] = winres(in_file["traces"][j], window=window)
        progress_bar.progress(i / n_profiling, text=f"{i + 1}/{n_attack} attack traces processed")

    profiling_plaintext = np.zeros((n_profiling, 16))
    profiling_key = np.zeros((n_profiling, 16))
    profiling_masks = np.zeros((n_profiling, 18))

    attack_plaintext = np.zeros((n_attack, 16))
    attack_key = np.zeros((n_attack, 16))
    attack_masks = np.zeros((n_attack, 18))

    print(f"Retrieving profiling metadata from index {profiling_index[:20]}")
    for i, j in tqdm(enumerate(profiling_index)):
        profiling_plaintext[i] = raw_plaintexts[j]
        profiling_key[i] = raw_keys[j]
        profiling_masks[i] = raw_masks[j]
        progress_bar.progress(i / n_profiling, text=f"{i + 1}/{n_profiling} profiling traces processed")

    print(f"Retrieving attack metadata from index {attack_index[:20]}")
    for i, j in tqdm(enumerate(attack_index)):
        if i == 10000:
            break
        attack_plaintext[i] = raw_plaintexts[j]
        attack_key[i] = raw_keys[j]
        attack_masks[i] = raw_masks[j]
        progress_bar.progress(i / n_profiling, text=f"{i + 1}/{n_attack} attack traces processed")

    out_file = h5py.File(f'{filepath_destination}', 'w')

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                        ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                        ])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                     ("key", attack_key.dtype, (len(attack_key[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                     ])

    profiling_metadata = np.array([(profiling_plaintext[n], profiling_key[n], profiling_masks[n]) for n in
                                   profiling_index], dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintext[n], attack_key[n], attack_masks[n]) for n in
                                attack_index], dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()
