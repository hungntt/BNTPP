import pickle
import numpy as np
from process_batch_seq import process_seq
import torch


def convert_helpdesk(max_length, fold):
    timestamps_list = []
    types_list = []
    lengths_list = []
    timeintervals_list = []
    task = 'helpdesk'

    file_path = f'data/{task}/whole_dataset_fold{fold}.pkl'
    file = torch.load(file_path)
    dim_process = file['num_marks']
    print('dim_process: {} for task: {}'.format(dim_process, task))
    seqs = file['sequences']
    one_seq_num = 0
    for seq in seqs:
        # Add dummy event at the end of marks and timestamps
        seq['arrival_times'].append(seq['t_end'])
        seq['marks'].append(dim_process)
        t_start = seq['t_start']
        timestamps = np.array(seq['arrival_times']) - t_start
        timestamps = timestamps[:max_length]
        types = np.array(seq['marks'])[:max_length]
        timeintervals = np.ediff1d(np.concatenate([[0], timestamps]))
        # Add dummy event at the end
        lengths = len(timestamps)
        if lengths == 1:
            one_seq_num += 1
            continue
        timestamps_list.append(np.asarray(timestamps))
        types_list.append(np.asarray(types))
        lengths_list.append(np.asarray(lengths))
        timeintervals_list.append(np.asarray(timeintervals))

    divisor = np.mean([item for timestamps in timeintervals_list for item in timestamps])
    timeintervals_list = [np.asarray(timeintervals) / divisor for timeintervals in timeintervals_list]
    print('one_seq_num: {}'.format(one_seq_num))
    dataset_dir = f'data/{task}/'
    save_path = f'data/{task}/whole_fold{fold}_manifold_format.pkl'
    with open(save_path, "wb") as f:
        save_data_ = {'timestamps': np.asarray(timestamps_list, dtype=object),
                      'types': np.asarray(types_list, dtype=object),
                      'lengths': np.asarray(lengths_list),
                      'timeintervals': np.asarray(timeintervals_list, dtype=object),
                      }
        pickle.dump(save_data_, f)
    return dataset_dir, dim_process


if __name__ == '__main__':
    for fold in range(5):
        max_length = 15
        sub_dataset = ['train', 'val', 'test']
        num_samples = [2931, 733, 916]
        save_path, process_dim = convert_helpdesk(max_length, fold)
        process_seq(save_path, sub_dataset, num_samples, process_dim=process_dim, fold=fold)
