import numpy as np
import h5py
from numba import njit
from tqdm import tqdm


@njit
def winres(trace, window=20, overlap=0.5):
    trace_winres = []
    step = int(window * overlap)
    max = len(trace)
    for i in range(0, max, step):
        trace_winres.append(np.mean(trace[i:i + window]))
    return np.array(trace_winres)


def generate_nopoi_ascadf(filepath, filepath_destination, n_traces, n_samples, progress_bar, window=20):
    n_profiling = 50000
    n_attack = 10000

    in_file = h5py.File(f'{filepath}', "r")
    metadata = in_file["metadata"]

    ns = int(n_samples / window) * 2
    profiling_samples = np.zeros((n_traces, ns))
    for trace_index in tqdm(range(n_profiling + n_attack)):
        profiling_samples[trace_index] = winres(in_file["traces"][trace_index], window=window)
        progress_bar.progress(trace_index/n_traces, text=f"{trace_index + 1}/{n_traces} processed")

    attack_samples = profiling_samples[n_profiling:n_profiling + n_attack]
    profiling_samples = profiling_samples[0:n_profiling]

    raw_plaintexts = metadata['plaintext']
    raw_keys = metadata['key']
    raw_masks = metadata['masks']

    profiling_plaintext = raw_plaintexts[0:n_profiling]
    profiling_key = raw_keys[0:n_profiling]
    profiling_masks = raw_masks[0:n_profiling]

    attack_plaintext = raw_plaintexts[n_profiling:n_profiling + n_attack]
    attack_key = raw_keys[n_profiling:n_profiling + n_attack]
    attack_masks = raw_masks[n_profiling:n_profiling + n_attack]

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
