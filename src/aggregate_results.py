import pandas as pd
import glob
import os
import json, logging, time
import re

# Read each CSV file in parallel and append to the list
import concurrent.futures

def read_csv_with_metadata(file):
    try:
        df = pd.read_csv(file)
        prep_time = df['prep_time'].head(1)/len(df)
        df['avg_time'] = df['algo_time'] + prep_time
        
        
        lower_err = df['err'].fillna('').str.lower()
        df['is_timeout'] = df['timeout'] = (df['algo_time'] >= 12) | (lower_err.str.contains('timeout'))

        df['is_failed'] = df['err'].notnull() & (~lower_err.str.contains('timeout'))

        # df['source_file'] = os.path.basename(file)
        # df['source_selection_mode'] = os.path.basename(file).split(".")[0].split("_")[-1]
        # df['top_coins'] = int(os.path.basename(file).split(".")[0].split("_")[-2])
        for col in ['distances', 'paths']:
            if col in df.columns:
                df = df.drop(columns=[col])
                
        # Only select columns that exist in the DataFrame
        expected_cols = ['coin', 'err', 'algo_time', 'algo', 'block', 'prep_time',
                        'n_nodes', 'n_edges', 'top_coin_count', 'avg_time', 'is_timeout', 'is_failed']
        
        return df[expected_cols]
    
    except Exception as e:
        logging.error(f"Error reading {file}: {str(e)}")
        return None

def aggregate_csv_files(start_block_number, end_block_number, config):
    # Get all CSV files from the results directory
    csv_files = glob.glob(os.path.join(config['paths']['data'], 'results', 'result_*.csv'))
    total_files = len(csv_files)
    if total_files == 0:
        logging.info("No CSV files found to combine")
        return

    dfs = []
    read_count = 0

    # Filter csv_files to only include those whose block number is between start_block_number and end_block_number
    filtered_csv_files = []
    # Extract the first 4 digits from the start_block_number for matching
    for csv_file in csv_files:
        bn = int(os.path.basename(csv_file).split('.')[0].split('_')[2])
        if start_block_number <= bn <= end_block_number:
            filtered_csv_files.append(csv_file)

    csv_files = filtered_csv_files
    total_files = len(csv_files)
    if total_files == 0:
        logging.info("No CSV files found in the specified block range to combine")
        return

    last_percent_reported = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(read_csv_with_metadata, file): file for file in csv_files}
        for future in concurrent.futures.as_completed(future_to_file):
            df = future.result()
            read_count += 1
            percent = int((read_count / total_files) * 100)
            if percent // 5 > last_percent_reported // 5:
                print(f"Read {read_count}/{total_files} files ({percent}%)")
                last_percent_reported = percent
            if df is not None:
                dfs.append(df)

    # Combine all dataframes
    t = int(time.time())
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # combined_df = combined_df.sort_values(['block_number', 'algorithm_name'])
        os.makedirs(os.path.join(config['paths']['data'], 'aggregate'), exist_ok=True)
        output_file = os.path.join(config['paths']['data'], 'aggregate', f'aggregated_{t}.csv.gz')
        combined_df.to_csv(output_file, index=False, compression='gzip')

        print(f"Successfully combined {len(dfs)} files into {output_file}")
        print(f"Total rows: {len(combined_df)}")
        
        # Filter out failed and timeout rows for average_time calculation
        filtered_df = combined_df[~combined_df['is_failed'] & ~combined_df['is_timeout']]
        summary = combined_df.groupby(['top_coin_count', 'algo']).agg(
            query_count=('coin', 'count'),
            timeout_count=('is_timeout', 'sum'),
            failed_count=('is_failed', 'sum')
        ).reset_index()
        avg_time = filtered_df.groupby(['top_coin_count', 'algo'])['avg_time'].mean().reset_index()
        summary = summary.merge(avg_time, on=['top_coin_count', 'algo'], how='left').rename(columns={'avg_time': 'average_time'})
        print(summary)

    
    else:
        print("No CSV files found to combine")
