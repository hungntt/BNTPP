# Read data csv file
import time

import pandas as pd
import torch

data_name = 'BPI_Challenge_2012'
helpdesk = pd.read_csv(f'data/{data_name}/{data_name}.csv')
for fold in range(5):
    train = pd.read_csv(f'data/{data_name}/train_fold{fold}_variation0_{data_name}.csv')
    test = pd.read_csv(f'data/{data_name}/test_fold{fold}_variation0_{data_name}.csv')
    val = pd.read_csv(f'data/{data_name}/val_fold{fold}_variation0_{data_name}.csv')
    num_unique_activities = len(helpdesk['ActivityID'].unique())
    data_pkl = dict()
    data_pkl['sequences'] = list()
    for dataset, dataset_name in zip([train, val, test], ['train', 'val', 'test']):
        num_cases = 0
        data_in_one_case = dataset.groupby('CaseID')
        for name, indices in data_in_one_case.groups.items():
            # Append all 'ActivityID' in one case to a list
            marks = list(dataset.loc[indices, 'ActivityID'])
            arrival_times = list()
            for timestamp in dataset.loc[indices, 'CompleteTimestamp']:
                # Convert timestamp (in type of 2012-02-10 13:48:04) to seconds
                t = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                # Convert datetime object to timestamp
                timestamp = time.mktime(t)
                # Append all timestamps in one case to a list
                arrival_times.append(timestamp)
            t_start = arrival_times[0]
            t_end = arrival_times[-1]
            # Append the sequence to the list
            data_pkl['sequences'].append({
                'arrival_times': arrival_times,
                'marks': marks,
                't_start': t_start,
                't_end': t_end,
            })
            num_cases += 1
        print(f'fold {fold} {dataset_name}: {num_cases}')

    data_pkl['num_marks'] = num_unique_activities
    print(f'Number of unique activities: {num_unique_activities}')
    # Print the length of sequences
    # Save the sequences
    torch.save(data_pkl, f'data/{data_name}/whole_dataset_fold{fold}.pkl')
    print('--------------------------------')
