import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd
import numpy as np
import torch


def load_synth_data(file_path, rows=None, input_len=512, output_len=512, random_seed=42):
    table = feather.read_table(file_path)
    df = table.to_pandas()
    target_lists = df['target'].tolist()
    numpy_array = np.array(target_lists)

    tensor = torch.tensor(numpy_array, dtype=torch.float32)
    if rows is not None and rows < tensor.size(0):
        torch.manual_seed(random_seed)
        tensor = tensor[torch.randperm(tensor.size(0))[:rows]]
    
    input_tensor = tensor[:, :input_len]
    output_tensor = tensor[:, input_len:input_len + output_len]
    return input_tensor, output_tensor

def load_single_synth_data(file_path, rows=None, input_len=512, output_len=512, random_seed=42):
    table = feather.read_table(file_path)
    df = table.to_pandas()

    def process_group(group_df):
        target_lists = group_df['target'].tolist()
        numpy_array = np.array(target_lists, dtype=np.float32)
        tensor = torch.tensor(numpy_array)

        if rows is not None and rows < tensor.size(0):
            torch.manual_seed(random_seed)
            indices = torch.randperm(tensor.size(0))[:rows]
            tensor = tensor[indices]

        input_tensor = tensor[:, :input_len]
        output_tensor = tensor[:, input_len:input_len + output_len]
        return input_tensor, output_tensor

    result = {}
    if 'kernel_name' in df.columns:
        grouped = df.groupby('kernel_name')
        for name, group in grouped:
            input_tensor, output_tensor = process_group(group)
            result[name] = (input_tensor, output_tensor)
    else:
        input_tensor, output_tensor = process_group(df)
        result['all_kernel'] = (input_tensor, output_tensor)

    return result

import numpy as np
import torch
import random
from sktime.datasets import load_tsf_to_dataframe
import pandas as pd


def sample_segments(data, segment_count=8000, segment_width=48, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    batch_size, seq_len = data.shape
    segments = []

    for _ in range(segment_count):
        series_idx = random.randint(0, batch_size - 1)
        series = data[series_idx]
        start = random.randint(0, seq_len - segment_width)
        segment = series[start:start + segment_width]
        segments.append(segment.unsqueeze(0))

    return torch.cat(segments, dim=0)

def split_width(data, in_seq=36, out_seq=12):
    if data.shape[1] > (in_seq + out_seq):
        data = data[:, :in_seq + out_seq]
    
    x = data[:, :in_seq]
    y = data[:, in_seq:in_seq + out_seq]
    return x, y

def load_real_data(setup='standard', n_samples=1000):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return_tensors = {}
    
    def get_io_len(setup, dataset):
        io_len = {
            'hospital': (72, 12),
            'covid': (30*3, 30),
            'energy': (48*3, 48),
            'nature': (30*3, 30),
            'exchange_rate': (36, 12),
            'nn5_weekly': (24, 8),
            'fred_md': (36, 12),
            'car_retail': (36, 12),
            'dominick': (24, 8),
            }
        if setup == 'standard':
            in_seq, out_seq = io_len[dataset]
        else:
            in_seq, out_seq = io_len[dataset]
            in_seq = int(in_seq * 2 / 3)
        return in_seq, out_seq

    
    ###############################
    # Finance Dataset
    ###############################
    in_seq, out_seq = get_io_len(setup, 'exchange_rate')
    #-----------------------------#
    df_finance = pd.read_csv('../data/exchange_rate.txt.gz', compression='gzip', sep='\t', header=None)[0]
    array_finance = np.array(df_finance.str.split(',').tolist(), dtype=np.float32)
    tensor_finance = torch.tensor(array_finance).T
    tensor_finance_1 = sample_segments(tensor_finance, segment_count=n_samples, segment_width=in_seq+out_seq)
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_finance_1, in_seq=in_seq, out_seq=out_seq)
    print(f"Finance tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['exchange_rate'] = (in_tensor, out_tensor)
    ###############################

    ################################
    # Finance NN5 Weekly Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'nn5_weekly')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/nn5_weekly_dataset.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names:
        values = df_medical[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"NN5_weekly tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['nn5_weekly'] = (in_tensor, out_tensor)

    ################################
    # Finance NN5 Weekly Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'fred_md')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/fred_md_dataset.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names:
        values = df_medical[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"fred_md tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['fred_md'] = (in_tensor, out_tensor)

    ################################
    # Medical Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'hospital')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/hospital_dataset.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names:
        values = df_medical[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"Hospital tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['hospital'] = (in_tensor, out_tensor)
    ###############################

    ################################
    # Covid Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'covid')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/covid_deaths_dataset.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names:
        values = df_medical[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"Covid tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['covid'] = (in_tensor, out_tensor)
    ###############################

    ################################
    # Retail Car Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'car_retail')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/car_parts_dataset_with_missing_values.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names:
        values = df_medical[0].loc[name]['series_value'].values
        # remove rows with NaN values
        values = values[~np.isnan(values)]
        # remove rows length < in_seq + out_seq
        if len(values) < in_seq + out_seq:
            continue
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"Car_retail tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['car_retail'] = (in_tensor, out_tensor)
    ###############################

    ################################
    # Dominick Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'dominick')
    #-----------------------------#
    df_medical = load_tsf_to_dataframe('../data/dominick_dataset.tsf')
    series_names = df_medical[0].index.get_level_values('series_name').unique()
    series_list = []
    for name in series_names[:int(n_samples*1.2)]:
        values = df_medical[0].loc[name]['series_value'].values
        # remove rows with NaN values
        values = values[~np.isnan(values)]
        # remove rows length < in_seq + out_seq
        if len(values) < in_seq + out_seq:
            continue
        series_tensor = torch.tensor(values, dtype=torch.float32)
        series_tensor = series_tensor[:in_seq + out_seq]
        series_list.append(series_tensor.unsqueeze(0))  # shape: [1, 84]
    tensor_medical = torch.cat(series_list, dim=0)  # shape: [batch, 84]
    if tensor_medical.shape[0] > n_samples:
        indices = torch.randperm(tensor_medical.shape[0])[:n_samples]
        tensor_medical = tensor_medical[indices]
    #-----------------------------#
    in_tensor, out_tensor = split_width(tensor_medical, in_seq=in_seq, out_seq=out_seq)
    print(f"Dominick tensor shape: {in_tensor.shape}, {out_tensor.shape}")
    return_tensors['dominick'] = (in_tensor, out_tensor)
    ###############################

    ################################
    # Energy Dataset
    ################################
    in_seq, out_seq = get_io_len(setup, 'energy')
    #-----------------------------#
    df_energy = load_tsf_to_dataframe('../data/australian_electricity_demand_dataset.tsf')
    series_names = df_energy[0].index.get_level_values('series_name').unique()
    for name in series_names:
        state_name = df_energy[0].loc[name].index.get_level_values('state').unique()[0]
        values = df_energy[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
        tensor_energy_1 = sample_segments(series_tensor, segment_count=n_samples, segment_width=in_seq+out_seq)
        #-----------------------------#
        in_tensor, out_tensor = split_width(tensor_energy_1, in_seq=in_seq, out_seq=out_seq)
        print(f"Energy {state_name} tensor shape: {in_tensor.shape}, {out_tensor.shape}")
        return_tensors[f'energy_{state_name}'] = (in_tensor, out_tensor)
    ###############################
    
    ###############################
    # Weather Dataset
    ###############################
    in_seq, out_seq = get_io_len(setup, 'nature')
    #-----------------------------#
    df_weather = load_tsf_to_dataframe('../data/weather_dataset.tsf')
    series_names = df_weather[0].index.get_level_values('series_name').unique()
    series_names = np.random.choice(series_names, size=100, replace=False)
    all_types = {}
    all_min_len = {}
    for name in series_names:
        weather_type = df_weather[0].loc[name].index.get_level_values('series_type').unique()[0]
        values = df_weather[0].loc[name]['series_value'].values
        series_tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
        if weather_type not in all_min_len:
            all_min_len[weather_type] = series_tensor.shape[1]
        else:
            all_min_len[weather_type] = min(all_min_len[weather_type], series_tensor.shape[1])

        if weather_type not in all_types:
            all_types[weather_type] = [series_tensor]
        else:
            all_types[weather_type].append(series_tensor)

    for key, value in all_types.items():
        for i in range(len(value)):
            if value[i].shape[1] > all_min_len[key]:
                value[i] = value[i][:, :all_min_len[key]]

    for key, value in all_types.items():
        all_types[key] = torch.cat(value, dim=0)
        #-----------------------------#
        series_tensor = all_types[key]
        tensor_energy_1 = sample_segments(series_tensor, segment_count=n_samples, segment_width=in_seq+out_seq)
        in_tensor, out_tensor = split_width(tensor_energy_1, in_seq=in_seq, out_seq=out_seq)
        print(f"Weather {key} tensor shape: {all_types[key].shape} => {in_tensor.shape}, {out_tensor.shape}")
        return_tensors[f'nature_{key}'] = (in_tensor, out_tensor)
    ###############################
    return return_tensors

def load_wireless_data(setup='tdd', n_samples=1000, first_dim=False, short=False):
    return_tensors = {}
    
    snr = {
        'tdd': 18,
        'fdd': 5,
    }
    speed = {
        'tdd': [10, 50, 100],
        'fdd': [10, 50, 100],
    }
    #-----------------------------#
    for s in speed[setup]:
        key = f'{setup}_speed{s}_snr{snr[setup]}'
        # read npy file
        in_tensor = np.load(f'../data/neurips_wireless_data/{key}_input.npy')
        out_tensor = np.load(f'../data/neurips_wireless_data/{key}_output.npy')
        in_tensor = torch.tensor(in_tensor, dtype=torch.float32)[:n_samples]
        out_tensor = torch.tensor(out_tensor, dtype=torch.float32)[:n_samples]
        if first_dim:
            # [2000, 16, 96] => [2000, 16] by taking the first one
            in_tensor = in_tensor[:, :, 0]
            out_tensor = out_tensor[:, :, 0]
        if short:
            in_tensor = in_tensor[:, :8]
        return_tensors[f"wireless_{key}"] = (in_tensor, out_tensor)
    #-----------------------------#
    return return_tensors