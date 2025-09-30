import logging
import os
import pandas as pd
import src.query as query_strings
from src import api
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing as mp

def process_snapshots_for_swaps(config):
    """
    For each pool in the snapshots, fetch the last 10,000 swaps and save them in a CSV file.
    """
    snapshots_directory = os.path.join(config['paths']['data'], 'snapshots')

    # Iterate over all snapshot files
    for snapshot_file in os.listdir(snapshots_directory):
        if snapshot_file.endswith(".csv.gz"):
            fetcher_thread = SwapFetcherThread(config, snapshot_file)
            fetcher_thread.start()
            fetcher_thread.join()  # Wait for the thread to complete

class SwapFetcherThread(threading.Thread):
    def __init__(self, config, snapshot_file):
        super().__init__()
        self.config = config
        self.snapshot_file = snapshot_file
        self.swaps_directory = os.path.join(config['paths']['data'], 'swaps')
        os.makedirs(self.swaps_directory, exist_ok=True)
        self.mp_pool = None

    def run(self):
        snapshot_path = os.path.join(self.config['paths']['data'], 'snapshots', self.snapshot_file)
        snapshot_df = pd.read_csv(snapshot_path)

        self.mp_pool = mp.Pool(processes=8)
        for _, pool in snapshot_df.iterrows():
            pool_id = pool['id']  # Assuming 'id' is the pool identifier
            self.mp_pool.apply_async(self.fetch_and_save_swaps, args=(pool_id,))
        
        self.mp_pool.close()
        self.mp_pool.join()

    def fetch_and_save_swaps(self, pool_id):
        try:
            all_swaps = pd.DataFrame()
            swaps_df = fetch_last_swaps(pool_id, self.config['services']['the_graph_tokens'][0], limit=1000)
            all_swaps = pd.concat([all_swaps, swaps_df], ignore_index=True)

            swaps_file = os.path.join(self.swaps_directory, f"swaps_{pool_id}.csv")
            all_swaps.to_csv(swaps_file, index=False)
            logging.info(f"Saved swaps for pool {pool_id} to {swaps_file}")
        except Exception as e:
            logging.error(f"Error fetching swaps for pool {pool_id}: {e}")

    def stop(self):
        if self.mp_pool:
            self.mp_pool.terminate() 
        
def fetch_last_swaps(pool_id, the_graph_token, limit=1000, skip_count = 0):
    """
    Fetch the last `limit` swaps for a given pool using The Graph API.
    """
    query = query_strings.swaps_query.replace("pool_id", pool_id).replace("count", str(limit)).replace("skip_id", str(skip_count))
    res = api.the_graph_query_wrapper(query, the_graph_token)
    return pd.DataFrame(res['data']['swaps'])


def process_snapshots_for_swaps(config):
    """
    For each pool in the snapshots, fetch the last 10,000 swaps and save them in a CSV file.
    """
    snapshots_directory = os.path.join(config['paths']['data'], 'snapshots')
    swaps_directory = os.path.join(config['paths']['data'], 'swaps')
    os.makedirs(swaps_directory, exist_ok=True)

    # Iterate over all snapshot files
    for snapshot_file in os.listdir(snapshots_directory):
        if snapshot_file.endswith(".csv.gz"):
            snapshot_path = os.path.join(snapshots_directory, snapshot_file)
            # logging.info(f"Processing snapshot: {snapshot_file}")

            # Load the snapshot data
            snapshot_df = pd.read_csv(snapshot_path)
            print(snapshot_df)

            # Iterate over each pool in the snapshot
            for _, pool in snapshot_df.iterrows():
                pool_id = pool['id']  # Assuming 'id' is the pool identifier
                # logging.info(f"This swaps for pool: {pool_id}")

                try:
                    # Fetch the last 10,000 swaps for the pool
                    all_swaps = pd.DataFrame()
                    # for i in range(10):
                    swaps_df = fetch_last_swaps(pool_id, config['services']['the_graph_tokens'][0], limit=1000)
                    all_swaps = pd.concat([all_swaps, swaps_df], ignore_index=True)

                    # Save all the swaps data to a single CSV file
                    swaps_file = os.path.join(swaps_directory, f"swaps_{pool_id}.csv")
                    all_swaps.to_csv(swaps_file, index=False)
                    logging.info(f"Saved swaps for pool {pool_id} to {swaps_file}")
                except Exception as e:
                    logging.error(f"Error fetching swaps for pool {pool_id}: {e}")
                break
