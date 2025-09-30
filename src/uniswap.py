import math
import pandas as pd
import networkx as nx
import subprocess
import itertools as it
from ast import literal_eval
from pandarallel import pandarallel


MIN_TICK = -887272
MAX_TICK = 887272

def tick_to_sqrtPriceX96(tick):
    return int(1.0001 ** (tick / 2) * (2 ** 96))


def get_reserves(liquidity, sqrtPriceX96, sqrtPriceLowerX96, sqrtPriceUpperX96):
    if sqrtPriceX96 <= sqrtPriceLowerX96:
        amount0 = liquidity * (sqrtPriceUpperX96 - sqrtPriceLowerX96) // (sqrtPriceLowerX96 * sqrtPriceUpperX96)
        amount1 = 0
    elif sqrtPriceX96 < sqrtPriceUpperX96:
        amount0 = liquidity * (sqrtPriceUpperX96 - sqrtPriceX96) // (sqrtPriceX96 * sqrtPriceUpperX96)
        amount1 = liquidity * (sqrtPriceX96 - sqrtPriceLowerX96) // (2 ** 96)
    else:
        amount0 = 0
        amount1 = liquidity * (sqrtPriceUpperX96 - sqrtPriceLowerX96) // (2 ** 96)
    return amount0, amount1

def compute_reserves(row):
    try:
        sqrtPriceLowerX96 = tick_to_sqrtPriceX96(MIN_TICK)
        sqrtPriceUpperX96 = tick_to_sqrtPriceX96(MAX_TICK)

        tick = int(row['other']['tick'])
        sqrtPriceX96 = int(row['other']['sqrtPrice'])
        liquidity = int(row['liquidity'])
        reserve0, reserve1 = get_reserves(liquidity, sqrtPriceX96, sqrtPriceLowerX96, sqrtPriceUpperX96)
        return pd.Series({'token0_reserve': reserve0, 'token1_reserve': reserve1})
    except Exception:
        return pd.Series({'token0_reserve': None, 'token1_reserve': None})

def add_reserves(df):
    
    pandarallel.initialize(progress_bar=False, verbose=0)
    reserves = df.parallel_apply(compute_reserves, axis=1)
    # Drop all-NA columns to avoid FutureWarning
    reserves = reserves.dropna(axis=1, how='all')
    df[['token0_reserve', 'token1_reserve']] = reserves
    return df



    
def get_top_tokens(df, top_token_counts):
    """
    Returns a list of (token_id, token_symbol) tuples for the top tokens by totalValueLockedUSD.
    """
    # Sum totalValueLockedUSD for token0_id and token1_id separately
    tvl_token0 = df[['totalValueLockedUSD', 'token0_id']].groupby('token0_id').sum()
    tvl_token1 = df[['totalValueLockedUSD', 'token1_id']].groupby('token1_id').sum()

    # Rename columns for merging
    tvl_token0 = tvl_token0.rename(columns={'totalValueLockedUSD': 'tvl_token0'})
    tvl_token1 = tvl_token1.rename(columns={'totalValueLockedUSD': 'tvl_token1'})

    # Combine both by index (token id), fill NaN with 0, and sum to get total TVL per token id
    tvl_combined = tvl_token0.join(tvl_token1, how='outer').fillna(0)
    tvl_combined['tvl_token0'] = pd.to_numeric(tvl_combined['tvl_token0'], errors='coerce').fillna(0)
    tvl_combined['tvl_token1'] = pd.to_numeric(tvl_combined['tvl_token1'], errors='coerce').fillna(0)
    tvl_combined['totalValueLockedUSD'] = tvl_combined['tvl_token0'] + tvl_combined['tvl_token1']

    # Reset index to have token id as a column
    tvl_combined = tvl_combined[['totalValueLockedUSD']]
    tvl_combined = tvl_combined.reset_index().rename(columns={'index': 'token_id'})

    # Map token_id to token_symbol using original df
    token0_map = df[['token0_id', 'token0_symbol']].drop_duplicates().set_index('token0_id')['token0_symbol']
    token1_map = df[['token1_id', 'token1_symbol']].drop_duplicates().set_index('token1_id')['token1_symbol']
    token_symbol_map = token0_map.combine_first(token1_map)
    token_symbol_map = token_symbol_map[~token_symbol_map.index.duplicated(keep='first')]

    # Add token_symbol column to tvl_combined
    tvl_combined['token_symbol'] = tvl_combined['token_id'].map(token_symbol_map)

    tvl_combined = tvl_combined.sort_values(by='totalValueLockedUSD', ascending=False)
    top_tokens = tvl_combined[['token_id', 'token_symbol']].head(top_token_counts)
    
    return top_tokens
