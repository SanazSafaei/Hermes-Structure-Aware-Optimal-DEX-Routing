import logging
import random
import time

from web3 import Web3
from src import api
import src.query as query_strings
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_prices_from_sqrt_price(sqrt_price_x96, decimals0=18, decimals1=18):
    """
    Calculate token0Price and token1Price from sqrtPriceX96.
    
    Returns:
        tuple: (token0Price, token1Price)
    """
    try:
        # Convert from Q64.96 to decimal
        Q96 = 2 ** 96
        sqrt_price = float(sqrt_price_x96) / Q96
        
        # Calculate raw price (token1 per token0)
        raw_price = sqrt_price ** 2
        
        # Adjust for token decimals
        decimal_adjustment = (10 ** decimals1) / (10 ** decimals0)
        adjusted_price = raw_price * decimal_adjustment
        
        # token0Price: price of token0 in terms of token1
        token0_price = adjusted_price
        
        # token1Price: price of token1 in terms of token0 (inverse)
        token1_price = 1.0 / adjusted_price if adjusted_price != 0 else 0.0
        
        return token0_price, token1_price
        
    except Exception as e:
        logging.error(f"Error calculating prices from sqrtPriceX96 {sqrt_price_x96}: {e}")
        return None, None


def fetch_changes_transactions(start_block, end_block, count, skip, config):

    #start time to calculate run time
    start_time = time.time()
    
    # Save the updated data to a CSV.gz file grouped by 100 blocks
    update_directory = os.path.join(config['paths']['data'], 'updates')
    os.makedirs(update_directory, exist_ok=True)
    # Create filename with range format
    current_update_path = os.path.join(update_directory, f'updates_{start_block:06d}.csv.gz')
    
    # Check if file already exists and append, otherwise create new
    if os.path.exists(current_update_path):
        logging.error(f"File {current_update_path} already exists, skipping")
        return
    
    # v2
    # Create web3 instance once per thread (for thread safety)

    ethereum_providers = config['services']['ethereum_providers']

    def fetch_block_tx_mapping(block_number):
        try:
            # Each thread gets its own Web3 instance for safety
            web3 = Web3(Web3.HTTPProvider(random.choice(ethereum_providers)))
            block = web3.eth.get_block(block_number, full_transactions=False)
            return {tx_hash.hex(): idx for idx, tx_hash in enumerate(block['transactions'])}
        except Exception as e:
            logging.warning(f"Could not get block {block_number}: {e}")
            return {}

    tx_hash_to_index = {}
    with ThreadPoolExecutor(max_workers=len(ethereum_providers)) as executor:
        future_to_block = {executor.submit(fetch_block_tx_mapping, block_number): block_number for block_number in range(start_block, end_block)}
        for future in as_completed(future_to_block):
            block_mapping = future.result()
            tx_hash_to_index.update(block_mapping)

    # logging.info(f'Pool updates [{start_block}:{end_block}]: Fetched block transaction mappings')

    # Accept batch_size as an input parameter (default to 5 for backward compatibility)
    batch_size = config['download']['uniswap_batch_size']

    # v2
    version = 'v2'
    endpoint = config['services']['graph_endpoints'][version]
    all_transactions = []
    for batch_start in range(start_block, end_block, batch_size):
        batch_end = min(batch_start + batch_size, end_block)
        block_numbers = list(range(batch_start, batch_end))
        try:
            query = query_strings.block_transactions_query_v2.replace("block_numbers", str(block_numbers))
            query = query.replace("first_num", str(count)).replace("skip_num", str(skip))

            res = api.the_graph_query_wrapper_by_endpoint(query, random.choice(config['services']['the_graph_tokens']), endpoint=endpoint)
            
            if not res or 'data' not in res or 'transactions' not in res['data']:
                logging.debug(f"No transaction data found for version {version} blocks {block_numbers}")
                continue
            
            transactions = res['data']['transactions']
            if not transactions:
                logging.debug(res)
                logging.debug(f"No transactions found for blocks {block_numbers}")
                continue
            
            # Collect all transactions for later processing
            for tx in transactions:
                tx['block_numbers'] = block_numbers  # Optionally store block_numbers for later use
                all_transactions.append(tx)

        except Exception as e:
            logging.error(f"Error in fetch_changes_transactions for version {version} block {block_numbers}: {e}")
            logging.exception(e)
            continue

    pool_updates = []
    all_transactions.sort(key=lambda tx: tx_hash_to_index.get(tx['id'], float('inf')))
    for transaction in all_transactions:
        block_number = transaction.get('blockNumber')
        extracted_updates = extract_pool_data_v2(block_number, transaction)
        pool_updates.extend(extracted_updates)
        if not pool_updates:
            logging.info(f"No pool updates found for block {block_number}")
            continue
    updates_df_v2 = pd.DataFrame(pool_updates)
    updates_df_v2['version'] = version
    # logging.info(f'Pool updates [{start_block}:{end_block}]: Fetched v2 updates')

    # v3
    version = 'v3'
    endpoint = config['services']['graph_endpoints'][version]
    all_transactions = []
    for batch_start in range(start_block, end_block, batch_size):
        batch_end = min(batch_start + batch_size, end_block)
        block_numbers = list(range(batch_start, batch_end))
        try:
            query = query_strings.block_transactions_query_v3.replace("block_numbers", str(block_numbers))
            query = query.replace("first_num", str(count)).replace("skip_num", str(skip))
            res = api.the_graph_query_wrapper_by_endpoint(query, random.choice(config['services']['the_graph_tokens']), endpoint=endpoint)
            if not res or 'data' not in res or 'transactions' not in res['data']:
                logging.debug(f"No transaction data found for version {version} blocks {block_numbers}")
                continue
            transactions = res['data']['transactions']
            if not transactions:
                logging.debug(res)
                logging.debug(f"No transactions found for blocks {block_numbers}")
                continue
            for tx in transactions:
                tx['block_numbers'] = block_numbers
                # Add logIndex to all events in the transaction if missing
                for event_type in ['swaps', 'mints', 'burns']:
                    if event_type in tx and isinstance(tx[event_type], list):
                        for idx, event in enumerate(tx[event_type]):
                            if 'logIndex' not in event:
                                # Use idx as fallback if logIndex is missing
                                event['logIndex'] = idx
                all_transactions.append(tx)
        except Exception as e:
            logging.error(f"Error in fetch_changes_transactions for version {version} block {block_numbers}: {e}")
            logging.exception(e)
            continue

    pool_updates = {}
    all_transactions.sort(key=lambda tx: tx_hash_to_index.get(tx['id'], float('inf')))
    for transaction in all_transactions:
        block_number = transaction.get('blockNumber')
        extracted_updates = extract_pool_data_v3(block_number, transaction)
        pool_updates.update(extracted_updates)
        if not pool_updates:
            logging.info(f"No pool updates found for block {block_number}")
            continue
    updates_df_v3 = pd.DataFrame(list(pool_updates.values()))
    updates_df_v3['version'] = version
    # logging.info(f'Pool updates [{start_block}:{end_block}]: Fetched v3 updates')
    
    # v4
    version = 'v4'
    endpoint = config['services']['graph_endpoints'][version]
    all_transactions = []
    for batch_start in range(start_block, end_block, batch_size):
        batch_end = min(batch_start + batch_size, end_block)
        block_numbers = list(range(batch_start, batch_end))
        try:
            query = query_strings.block_transactions_query_v4.replace("block_numbers", str(block_numbers))
            query = query.replace("first_num", str(count)).replace("skip_num", str(skip))
            res = api.the_graph_query_wrapper_by_endpoint(query, random.choice(config['services']['the_graph_tokens']), endpoint=endpoint)
            if not res or 'data' not in res or 'transactions' not in res['data']:
                logging.debug(f"No transaction data found for version {version} blocks {block_numbers}")
                continue
            transactions = res['data']['transactions']
            if not transactions:
                logging.debug(res)
                logging.debug(f"No transactions found for blocks {block_numbers}")
                continue
            for tx in transactions:
                tx['block_numbers'] = block_numbers
                # Add logIndex to all events in the transaction if missing
                for event_type in ['swaps', 'modifyLiquiditys']:
                    if event_type in tx and isinstance(tx[event_type], list):
                        for idx, event in enumerate(tx[event_type]):
                            if 'logIndex' not in event:
                                event['logIndex'] = idx
                all_transactions.append(tx)
        except Exception as e:
            logging.error(f"Error in fetch_changes_transactions for version {version} block {block_numbers}: {e}")
            logging.exception(e)
            continue

    pool_updates = {}
    all_transactions.sort(key=lambda tx: tx_hash_to_index.get(tx['id'], float('inf')))
    for transaction in all_transactions:
        block_number = transaction.get('blockNumber')
        extracted_updates = extract_pool_data_v4(block_number, transaction)
        pool_updates.update(extracted_updates)
        if not pool_updates:
            logging.info(f"No pool updates found for block {block_number}")
            continue
    updates_df_v4 = pd.DataFrame(list(pool_updates.values()))
    updates_df_v4['version'] = version
    # logging.info(f'Pool updates [{start_block}:{end_block}]: Fetched v4 updates')

    # Combine all versions
    updates_df = pd.concat([updates_df_v2, updates_df_v3, updates_df_v4], ignore_index=True)
            
            
    updates_df.to_csv(current_update_path, compression='gzip', index=False)

    # Log execution time and event breakdown
    elapsed = time.time() - start_time
    if not updates_df.empty:
        event_counts = updates_df['event'].value_counts().to_dict()
        version_counts = updates_df['version'].value_counts().to_dict()
        logging.info(
            f"Pool updates [{start_block}:{end_block}]: {len(updates_df)} ({event_counts}) ({version_counts}) in {elapsed:.2f}s"
        )
    else:
        logging.info(f"Pool updates [{start_block}:{end_block}]: Empty!")

    return updates_df


def extract_pool_data_v2(block_number, transaction):
    # Combine and sort all events (swaps, mints, burns) by logIndex to ensure proper order
    pool_updates = []
    all_events = []
    if 'swaps' in transaction and len(transaction['swaps']) > 0:
        swaps_df = pd.DataFrame(transaction['swaps'])
        swaps_df['event_type'] = 'swap'
        all_events.append(swaps_df)
    if 'mints' in transaction and len(transaction['mints']) > 0:
        mints_df = pd.DataFrame(transaction['mints'])
        mints_df['event_type'] = 'mint'
        all_events.append(mints_df)
    if 'burns' in transaction and len(transaction['burns']) > 0:
        burns_df = pd.DataFrame(transaction['burns'])
        burns_df['event_type'] = 'burn'
        all_events.append(burns_df)
    
    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        sorted_events = all_events_df.sort_values('logIndex', key=lambda x: pd.to_numeric(x, errors='coerce')).to_dict('records')
    else:
        sorted_events = []
    
    for event in sorted_events:

        if 'pair' not in event:
            continue

        if event['event_type'] == 'swap':
            amount0 = float(event['amount0In']) - float(event['amount0Out'])
            amount1 = float(event['amount1In']) - float(event['amount1Out'])
            token0_data = {**event['pair']['token0'], 'amount0': amount0}
            token1_data = {**event['pair']['token1'], 'amount1': amount1}
        else:
            token0_data = {**event['pair']['token0'], 'amount0': event['amount0']}
            token1_data = {**event['pair']['token1'], 'amount1': event['amount1']}
        
        pool_id = event['pair']['id']

        # Store the latest update for each pool (last event in block wins)
        if(block_number == event['pair']['createdAtBlockNumber']):
            pool_status = 'new'
        else:
            pool_status = 'updated'
        
        pool_updates.append({
            'pool_id': pool_id,
            'pool_status': pool_status,
            'block_number': block_number,
            'token0': token0_data,
            'token1': token1_data,
            'event': event['event_type'],
        })
    return pool_updates

def extract_pool_data_v3(block_number, transaction):
    # Combine and sort all events (swaps, mints, burns) by logIndex to ensure proper order
    pool_updates = {}
    all_events = []
    if 'swaps' in transaction:
        swaps_df = pd.DataFrame(transaction['swaps'])
        swaps_df['event_type'] = 'swap'
        all_events.append(swaps_df)
    if 'mints' in transaction:
        mints_df = pd.DataFrame(transaction['mints'])
        mints_df['event_type'] = 'mint'
        all_events.append(mints_df)
    if 'burns' in transaction:
        burns_df = pd.DataFrame(transaction['burns'])
        burns_df['event_type'] = 'burn'
        all_events.append(burns_df)

    sorted_events = []    
    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        if len(all_events_df):
            sorted_events = all_events_df.sort_values('logIndex', key=lambda x: pd.to_numeric(x, errors='coerce')).to_dict('records')
    
    for event in sorted_events:

        if 'pool' not in event:
            continue
        
        pool_id = event['pool']['id']
        tick = event.get('tick', 0)

        if event['event_type'] == 'swap':           
            sqrt_price_x96 = event['sqrtPriceX96']
            
            if 'token0' in event and 'decimals' in event['token0']:
                try:
                    token0_decimals = int(event['token0']['decimals'])
                except (ValueError, TypeError):
                    token0_decimals = 18
                    
            if 'token1' in event and 'decimals' in event['token1']:
                try:
                    token1_decimals = int(event['token1']['decimals'])
                except (ValueError, TypeError):
                    token1_decimals = 18
            
            # Calculate new prices
            token0_price, token1_price = calculate_prices_from_sqrt_price(
                int(sqrt_price_x96), token0_decimals, token1_decimals
            )

        if event['event_type'] == 'swap':
            token0_data = {**event['token0'], 'price': token0_price, 'amount0': event['amount0']}
            token1_data = {**event['token1'], 'price': token1_price, 'amount1': event['amount1']    }
        else:
            token0_data = {**event['token0'], 'amount0': event['amount0']}
            token1_data = {**event['token1'], 'amount1': event['amount1']}

        # Store the latest update for each pool (last event in block wins)
        if(block_number == event['pool']['createdAtBlockNumber']):
            pool_status = 'new'
        else:
            pool_status = 'updated'
        
        pool_updates[pool_id] = {
            'pool_id': pool_id,
            'pool_status': pool_status,
            'block_number': block_number,
            'feeTier': event['pool']['feeTier'],
            'token0': token0_data,
            'token1': token1_data,
            'sqrtPrice': sqrt_price_x96 if event['event_type'] == 'swap' else None,
            'tick': tick,
            'event': event['event_type']
        }
    return pool_updates
            

def extract_pool_data_v4(block_number, transaction):
    # Combine and sort all events (swaps, mints, burns) by logIndex to ensure proper order
    pool_updates = {}
    all_events = []
    if 'swaps' in transaction:
        swaps_df = pd.DataFrame(transaction['swaps'])
        swaps_df['event_type'] = 'swap'
        all_events.append(swaps_df)
    if 'modifyLiquiditys' in transaction:
        modifyLiquiditys_df = pd.DataFrame(transaction['modifyLiquiditys'])
        modifyLiquiditys_df['event_type'] = 'modifyLiquidity'
        all_events.append(modifyLiquiditys_df)
    
    sorted_events = []
    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        if len(all_events_df):
            sorted_events = all_events_df.sort_values('logIndex', key=lambda x: pd.to_numeric(x, errors='coerce')).to_dict('records')
    
    for event in sorted_events:

        if 'pool' not in event:
            continue
        
        pool_id = event['pool']['id']
        tick = event.get('tick', 0)

        if event['event_type'] == 'swap':           
            sqrt_price_x96 = event['sqrtPriceX96']
            
            if 'token0' in event and 'decimals' in event['token0']:
                try:
                    token0_decimals = int(event['token0']['decimals'])
                except (ValueError, TypeError):
                    token0_decimals = 18
                    
            if 'token1' in event and 'decimals' in event['token1']:
                try:
                    token1_decimals = int(event['token1']['decimals'])
                except (ValueError, TypeError):
                    token1_decimals = 18
            
            # Calculate new prices
            token0_price, token1_price = calculate_prices_from_sqrt_price(
                int(sqrt_price_x96), token0_decimals, token1_decimals
            )

        if event['event_type'] == 'swap':
            token0_data = {**event['token0'], 'price': token0_price, 'amount0': event['amount0']}
            token1_data = {**event['token1'], 'price': token1_price, 'amount1': event['amount1']}
        else:
            token0_data = {**event['token0'], 'amount0': event['amount0']}
            token1_data = {**event['token1'], 'amount1': event['amount1']}

        # Store the latest update for each pool (last event in block wins)
        if(block_number == event['pool']['createdAtBlockNumber']):
            pool_status = 'new'
        else:
            pool_status = 'updated'
        
        pool_updates[pool_id] = {
            'pool_id': pool_id,
            'pool_status': pool_status,
            'block_number': block_number,
            'feeTier': event['pool']['feeTier'],
            'token0': token0_data,
            'token1': token1_data,
            'sqrtPrice': sqrt_price_x96 if event['event_type'] == 'swap' else None,
            'tick': tick,
            'event': event['event_type']
        }
    return pool_updates