import pickle
import numpy as np
from process_batch_seq import process_seq
import torch

def convert_mooc(max_length):
    lengths_list = []
    timeintervals_list = []
    task = 'mooc'
    file_path = 'data/' + task + '/' + 'whole_dataset' +'.pkl'

    with open(file_path, 'rb') as f:
        file = pickle.load(f, encoding='latin1')

    dim_process = file['dim_process']
    print('dim_process: {} for task: {}'.format(dim_process, task))
    
    file['arrival_times'] = [np.array(e)[:max_length] for e in file['arrival_times']]
    file['marks'] = [np.array(e)[:max_length] for e in file['marks']]

    seqs = file['arrival_times']
    one_seq_num = 0
    for seq in seqs:
        timeintervals = np.ediff1d(np.concatenate([[0], seq]))
        
        lengths = len(seq)
        if lengths == 1:
            one_seq_num += 1
            continue

        lengths_list.append(np.asarray(lengths))
        timeintervals_list.append(np.asarray(timeintervals))

    print('one_seq_num: {}'.format(one_seq_num))
    dataset_dir = 'data/' + task + '/'
    save_path = 'data/' + task + '/' + 'whole' +'_manifold_format.pkl'
    
    with open(save_path, "wb") as f:
        save_data_ = {'timestamps': np.asarray(file['arrival_times'], dtype=object),
                     'types': np.asarray(file['marks'], dtype=object),
                     'lengths': np.asarray(lengths_list),
                     'timeintervals': np.asarray(timeintervals_list, dtype=object)
                      }
        pickle.dump(save_data_,f)

    print('max_length: ', np.max(np.asarray(lengths_list)), 
        'min_length:', np.min(np.asarray(lengths_list)), 
        'mean_length:', np.mean(np.asarray(lengths_list)), 
        'num_seqence:', len(np.asarray(lengths_list)))
    return dataset_dir, dim_process

if __name__ == '__main__':
    max_length = 256
    sub_dataset = ['train', 'val', 'test']
    num_samples = [5047, 700, 1300]
    save_path, process_dim = convert_mooc(max_length)
    process_seq(save_path, sub_dataset, num_samples, process_dim=process_dim)
