import time
import csv
import logging
import random
import requests
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from web3 import Web3
import pandas as pd
import gzip
import json
from dateutil.relativedelta import relativedelta
import src.query as query_strings
from src import api


class UniswapTransactionFetcher:
    """
    A class to fetch all Uniswap v2, v3, and v4 transactions from January 2024 to June 2025
    using multithreading and save them to monthly CSV.gz files.
    """
    
    def __init__(self, config):
        self.config = config
        self.ethereum_providers = config['services']['ethereum_providers']
        self.the_graph_tokens = config['services']['the_graph_tokens']
        self.graph_endpoints = config['services']['graph_endpoints']
        self.data_dir = config['paths']['data']
        self.batch_size = config.get('download', {}).get('uniswap_batch_size', 5)
        self.n_threads = config.get('download', {}).get('n_threads', 8)
        
        # Create output directory
        self.output_dir = os.path.join(self.data_dir, 'transactions')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Thread-local storage for Web3 instances
        self.thread_local = threading.local()

    def get_web3_instance(self):
        """Get a thread-local Web3 instance"""
        if not hasattr(self.thread_local, 'web3'):
            self.thread_local.web3 = Web3(Web3.HTTPProvider(random.choice(self.ethereum_providers)))
        return self.thread_local.web3

    def get_block_number_by_timestamp(self, timestamp):
        """
        Get the block number for a given timestamp using binary search.
        This is an approximation based on average block time.
        """
        # Ethereum block time is approximately 12 seconds
        # Block number 1 was at timestamp 1438269973 (July 30, 2015)
        genesis_timestamp = 1438269973
        genesis_block = 1
        avg_block_time = 12
        
        # Initial estimation
        estimated_block = genesis_block + (timestamp - genesis_timestamp) // avg_block_time
        
        try:
            web3 = self.get_web3_instance()
            
            # Binary search for the exact block
            left, right = max(1, estimated_block - 1000), estimated_block + 1000
            
            while left <= right:
                mid = (left + right) // 2
                try:
                    block = web3.eth.get_block(mid)
                    block_timestamp = block['timestamp']
                    
                    if block_timestamp <= timestamp:
                        left = mid + 1
                    else:
                        right = mid - 1
                except Exception:
                    # If we can't get the block, adjust the search range
                    right = mid - 1
                    
            return right
        except Exception as e:
            logging.error(f"Error getting block number for timestamp {timestamp}: {e}")
            return estimated_block

    def get_month_date_ranges(self, start_date, end_date):
        """
        Generate date ranges for each month between start_date and end_date
        """
        ranges = []
        current_date = start_date.replace(day=1)  # Start from the first day of the month
        
        while current_date <= end_date:
            # Calculate the end of the current month
            next_month = current_date + relativedelta(months=1)
            month_end = min(next_month - timedelta(days=1), end_date)
            
            ranges.append((current_date, month_end))
            current_date = next_month
            
        return ranges

    def fetch_transactions_for_date_range(self, start_date, end_date, version):
        """
        Fetch all transactions for a specific date range and version
        """
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        start_block = self.get_block_number_by_timestamp(start_timestamp)
        end_block = self.get_block_number_by_timestamp(end_timestamp)
        
        logging.info(f"Fetching {version} transactions from {start_date} to {end_date} (blocks {start_block}-{end_block})")
        
        all_transactions = []
        
        # Query transactions in batches of blocks
        for batch_start in range(start_block, end_block + 1, 100):
            batch_end = min(batch_start + 100, end_block + 1)
            block_numbers = list(range(batch_start, batch_end))
            
            try:
                # Use the appropriate query for each version
                if version == 'v2':
                    query = self.get_v2_transactions_query()
                elif version == 'v3':
                    query = query_strings.block_transactions_query_v3
                elif version == 'v4':
                    query = query_strings.block_transactions_query_v4
                else:
                    continue
                
                query = query.replace("block_numbers", str(block_numbers))
                query = query.replace("first_num", "1000").replace("skip_num", "0")
                
                endpoint = self.graph_endpoints[version]
                token = random.choice(self.the_graph_tokens)
                
                res = api.the_graph_query_wrapper_by_endpoint(query, token, endpoint=endpoint)
                
                if not res or 'data' not in res or 'transactions' not in res['data']:
                    continue
                
                transactions = res['data']['transactions']
                if not transactions:
                    continue
                
                # Process transactions and add version info
                for tx in transactions:
                    tx['version'] = version
                    # For v2, we need to get gasUsed and gasPrice from the transaction
                    if version == 'v2':
                        tx = self.enrich_v2_transaction(tx)
                    all_transactions.append(tx)
                    
            except Exception as e:
                logging.error(f"Error fetching {version} transactions for blocks {block_numbers}: {e}")
                continue
        
        return all_transactions

    def get_v2_transactions_query(self):
        """
        Get the v2 transactions query with gasUsed and gasPrice included
        """
        return """
        {
          transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
          blockNumber_in: block_numbers
          }) 
          {
            id
            blockNumber
            timestamp
            gasUsed
            gasPrice
            swaps {
                id
                timestamp
                pair
                { 
                  id
                  createdAtBlockNumber
                  token0{
                    id
                    decimals
                    symbol
                    name
                  }
                  token1 {
                    id
                    decimals
                    symbol
                    name
                  }
                }
                amount0In
                amount1In
                amount0Out
                amount1Out
                logIndex
              }
             mints {
              id
              timestamp
              amount0
              amount1
              logIndex
              pair
                { 
                  id
                  createdAtBlockNumber
                  token0{
                    id
                    decimals
                    symbol
                    name
                  }
                  token1 {
                    id
                    decimals
                    symbol
                    name
                  }
                }
             }
             burns {
              id
              timestamp
              amount0
              amount1
              logIndex
              pair
                { 
                  id
                  createdAtBlockNumber
                  token0{
                    id
                    decimals
                    symbol
                    name
                  } 
                  token1 {
                    id
                    decimals
                    symbol
                    name
                  }
                }
             }
          }
        }
        """

    def enrich_v2_transaction(self, transaction):
        """
        Enrich v2 transaction with gasUsed and gasPrice from the blockchain
        """
        try:
            web3 = self.get_web3_instance()
            tx_hash = transaction['id']
            
            # Get transaction receipt for gasUsed and transaction for gasPrice
            tx_receipt = web3.eth.get_transaction_receipt(tx_hash)
            tx_data = web3.eth.get_transaction(tx_hash)
            
            transaction['gasUsed'] = str(tx_receipt.gasUsed)
            transaction['gasPrice'] = str(tx_data.gasPrice)
            
        except Exception as e:
            logging.warning(f"Could not enrich transaction {transaction['id']}: {e}")
            transaction['gasUsed'] = None
            transaction['gasPrice'] = None
            
        return transaction

    def flatten_transaction_data(self, transactions):
        """
        Flatten transaction data into a format suitable for CSV
        """
        flattened_data = []
        
        for tx in transactions:
            base_tx = {
                'transaction_id': tx['id'],
                'block_number': tx['blockNumber'],
                'timestamp': tx['timestamp'],
                'version': tx['version'],
                'gas_used': tx.get('gasUsed'),
                'gas_price': tx.get('gasPrice')
            }
            
            # Process swaps
            if 'swaps' in tx:
                for swap in tx['swaps']:
                    row = base_tx.copy()
                    row['event_type'] = 'swap'
                    row['event_id'] = swap['id']
                    row['log_index'] = swap.get('logIndex')
                    
                    if tx['version'] == 'v2':
                        row['pair_id'] = swap['pair']['id']
                        row['pair_created_at_block'] = swap['pair']['createdAtBlockNumber']
                        row['token0_id'] = swap['pair']['token0']['id']
                        row['token0_symbol'] = swap['pair']['token0']['symbol']
                        row['token0_name'] = swap['pair']['token0']['name']
                        row['token0_decimals'] = swap['pair']['token0']['decimals']
                        row['token1_id'] = swap['pair']['token1']['id']
                        row['token1_symbol'] = swap['pair']['token1']['symbol']
                        row['token1_name'] = swap['pair']['token1']['name']
                        row['token1_decimals'] = swap['pair']['token1']['decimals']
                        row['amount0_in'] = swap['amount0In']
                        row['amount1_in'] = swap['amount1In']
                        row['amount0_out'] = swap['amount0Out']
                        row['amount1_out'] = swap['amount1Out']
                    else:
                        row['pool_id'] = swap['pool']['id']
                        row['pool_created_at_block'] = swap['pool']['createdAtBlockNumber']
                        row['fee_tier'] = swap['pool']['feeTier']
                        row['liquidity'] = swap['pool']['liquidity']
                        row['sqrt_price'] = swap['pool']['sqrtPrice']
                        row['token0_id'] = swap['token0']['id']
                        row['token0_symbol'] = swap['token0']['symbol']
                        row['token0_name'] = swap['token0']['name']
                        row['token0_decimals'] = swap['token0']['decimals']
                        row['token1_id'] = swap['token1']['id']
                        row['token1_symbol'] = swap['token1']['symbol']
                        row['token1_name'] = swap['token1']['name']
                        row['token1_decimals'] = swap['token1']['decimals']
                        row['amount0'] = swap['amount0']
                        row['amount1'] = swap['amount1']
                        row['sqrt_price_x96'] = swap.get('sqrtPriceX96')
                        row['tick'] = swap.get('tick')
                    
                    flattened_data.append(row)
            
            # Process mints
            if 'mints' in tx:
                for mint in tx['mints']:
                    row = base_tx.copy()
                    row['event_type'] = 'mint'
                    row['event_id'] = mint['id']
                    row['log_index'] = mint.get('logIndex')
                    
                    if tx['version'] == 'v2':
                        row['pair_id'] = mint['pair']['id']
                        row['pair_created_at_block'] = mint['pair']['createdAtBlockNumber']
                        row['token0_id'] = mint['pair']['token0']['id']
                        row['token0_symbol'] = mint['pair']['token0']['symbol']
                        row['token0_name'] = mint['pair']['token0']['name']
                        row['token0_decimals'] = mint['pair']['token0']['decimals']
                        row['token1_id'] = mint['pair']['token1']['id']
                        row['token1_symbol'] = mint['pair']['token1']['symbol']
                        row['token1_name'] = mint['pair']['token1']['name']
                        row['token1_decimals'] = mint['pair']['token1']['decimals']
                    else:
                        row['pool_id'] = mint['pool']['id']
                        row['pool_created_at_block'] = mint['pool']['createdAtBlockNumber']
                        row['fee_tier'] = mint['pool']['feeTier']
                        row['liquidity'] = mint['pool']['liquidity']
                        row['sqrt_price'] = mint['pool']['sqrtPrice']
                        row['token0_id'] = mint['token0']['id']
                        row['token0_symbol'] = mint['token0']['symbol']
                        row['token0_name'] = mint['token0']['name']
                        row['token0_decimals'] = mint['token0']['decimals']
                        row['token1_id'] = mint['token1']['id']
                        row['token1_symbol'] = mint['token1']['symbol']
                        row['token1_name'] = mint['token1']['name']
                        row['token1_decimals'] = mint['token1']['decimals']
                    
                    row['amount0'] = mint['amount0']
                    row['amount1'] = mint['amount1']
                    row['amount'] = mint.get('amount')
                    
                    flattened_data.append(row)
            
            # Process burns
            if 'burns' in tx:
                for burn in tx['burns']:
                    row = base_tx.copy()
                    row['event_type'] = 'burn'
                    row['event_id'] = burn['id']
                    row['log_index'] = burn.get('logIndex')
                    
                    if tx['version'] == 'v2':
                        row['pair_id'] = burn['pair']['id']
                        row['pair_created_at_block'] = burn['pair']['createdAtBlockNumber']
                        row['token0_id'] = burn['pair']['token0']['id']
                        row['token0_symbol'] = burn['pair']['token0']['symbol']
                        row['token0_name'] = burn['pair']['token0']['name']
                        row['token0_decimals'] = burn['pair']['token0']['decimals']
                        row['token1_id'] = burn['pair']['token1']['id']
                        row['token1_symbol'] = burn['pair']['token1']['symbol']
                        row['token1_name'] = burn['pair']['token1']['name']
                        row['token1_decimals'] = burn['pair']['token1']['decimals']
                    else:
                        row['pool_id'] = burn['pool']['id']
                        row['pool_created_at_block'] = burn['pool']['createdAtBlockNumber']
                        row['fee_tier'] = burn['pool']['feeTier']
                        row['liquidity'] = burn['pool']['liquidity']
                        row['sqrt_price'] = burn['pool']['sqrtPrice']
                        row['token0_id'] = burn['token0']['id']
                        row['token0_symbol'] = burn['token0']['symbol']
                        row['token0_name'] = burn['token0']['name']
                        row['token0_decimals'] = burn['token0']['decimals']
                        row['token1_id'] = burn['token1']['id']
                        row['token1_symbol'] = burn['token1']['symbol']
                        row['token1_name'] = burn['token1']['name']
                        row['token1_decimals'] = burn['token1']['decimals']
                    
                    row['amount0'] = burn['amount0']
                    row['amount1'] = burn['amount1']
                    row['amount'] = burn.get('amount')
                    
                    flattened_data.append(row)
            
            # Process modifyLiquiditys (v4 only)
            if 'modifyLiquiditys' in tx:
                for modify in tx['modifyLiquiditys']:
                    row = base_tx.copy()
                    row['event_type'] = 'modifyLiquidity'
                    row['event_id'] = modify['id']
                    row['log_index'] = modify.get('logIndex')
                    row['pool_id'] = modify['pool']['id']
                    row['pool_created_at_block'] = modify['pool']['createdAtBlockNumber']
                    row['fee_tier'] = modify['pool']['feeTier']
                    row['liquidity'] = modify['pool']['liquidity']
                    row['sqrt_price'] = modify['pool']['sqrtPrice']
                    row['token0_id'] = modify['token0']['id']
                    row['token0_symbol'] = modify['token0']['symbol']
                    row['token0_name'] = modify['token0']['name']
                    row['token0_decimals'] = modify['token0']['decimals']
                    row['token1_id'] = modify['token1']['id']
                    row['token1_symbol'] = modify['token1']['symbol']
                    row['token1_name'] = modify['token1']['name']
                    row['token1_decimals'] = modify['token1']['decimals']
                    row['amount0'] = modify['amount0']
                    row['amount1'] = modify['amount1']
                    row['amount'] = modify.get('amount')
                    
                    flattened_data.append(row)
        
        return flattened_data

    def save_to_csv_gz(self, data, filename):
        """
        Save data to a compressed CSV file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if not data:
            logging.warning(f"No data to save for {filename}")
            return
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, compression='gzip', index=False)
        logging.info(f"Saved {len(data)} records to {filepath}")

    def fetch_month_data(self, start_date, end_date):
        """
        Fetch data for a single month using multithreading
        """
        month_str = start_date.strftime('%Y_%m')
        logging.info(f"Processing month: {month_str}")
        
        # Use ThreadPoolExecutor for concurrent fetching of different versions
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.fetch_transactions_for_date_range, start_date, end_date, version): version
                for version in ['v2', 'v3', 'v4']
            }
            
            all_month_data = []
            for future in as_completed(futures):
                version = futures[future]
                try:
                    transactions = future.result()
                    flattened_data = self.flatten_transaction_data(transactions)
                    all_month_data.extend(flattened_data)
                    logging.info(f"Fetched {len(flattened_data)} records for {version} in {month_str}")
                except Exception as e:
                    logging.error(f"Error fetching {version} data for {month_str}: {e}")
        
        # Save month data to CSV.gz
        filename = f"uniswap_transactions_{month_str}.csv.gz"
        self.save_to_csv_gz(all_month_data, filename)
        
        return len(all_month_data)

    def fetch_all_transactions(self, start_date_str="2024-01-01", end_date_str="2025-06-30"):
        """
        Main function to fetch all transactions from start_date to end_date
        """
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        logging.info(f"Starting transaction fetch from {start_date} to {end_date}")
        
        # Get month ranges
        month_ranges = self.get_month_date_ranges(start_date, end_date)
        
        total_records = 0
        start_time = time.time()
        
        # Process each month
        for start_month, end_month in month_ranges:
            try:
                month_records = self.fetch_month_data(start_month, end_month)
                total_records += month_records
            except Exception as e:
                logging.error(f"Error processing month {start_month}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed fetching {total_records} total records in {elapsed_time:.2f} seconds")
        
        return total_records


def main():
    """
    Main function to run the transaction fetcher
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('transaction_fetcher.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create and run the fetcher
    fetcher = UniswapTransactionFetcher(config)
    fetcher.fetch_all_transactions()


if __name__ == "__main__":
    main()
