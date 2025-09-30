# in v2 liquidity is equal to reserveUSD
pools_query_v2 = """ {
    pairs(first:first_num, skip:skip_num, orderBy: order_by_param, orderDirection: desc, where: {reserve0_gt: 0, reserve1_gt: 0}){
        id
        createdAtBlockNumber
        token0 {
          id
          symbol
          name
          decimals
        }
        token1 {
          id
          symbol
          name
          decimals
        }

        # liquidity = Math.sqrt(reserve0, reserve1)
        reserve0
        reserve1

        totalSupply

        # derived liquidity
        reserveETH
        reserveUSD

        # used for separating per pair reserves and global
        trackedReserveETH

        token0Price
        token1Price
    }
}
"""

pools_query = """{
    pools(first:first_num, skip:skip_num, orderBy: order_by_param, orderDirection: desc, where: { liquidity_gt: 0 }){
      id
      createdAtBlockNumber
      token0 {
        id
        symbol
        name
        decimals
      }
      token1 {
        id
        symbol
        name
        decimals
      }
      feeTier
      liquidity
      sqrtPrice
      token0Price
      token1Price
      tick
      feesUSD
      totalValueLockedUSD
    }
  }
"""

swaps_by_block_query_v3_v4 = """
{
  swaps(first:first_num, skip:skip_num, where: {transaction_: {blockNumber: "block_number"}}) {
    id
    transaction {
      id
      blockNumber
      timestamp
    }
    timestamp
    pool {
      id
    }
    token0 {
      id
      symbol
      name
      decimals
    }
    token1 {
      id
      symbol
      name
      decimals
    }
    sender
    origin
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
    logIndex
  }
}
"""


transactions_query = """
{
  transactions(first:count, skip:skip_id, where: {swaps_: {id: "swap_id"}}) {
    id
    blockNumber
    timestamp
    gasUsed
    gasPrice,
    swaps {
            id
            transaction
            timestamp
            pool: {
              id
              createdAtTimestamp
              createdAtBlockNumber
              token0: {
                id
                symbol
                name
              }
              token1: {
                id
                symbol
                name
              }
              feeTier
              liquidity
              sqrtPrice
              token0Price
              token1Price
              tick
              observationIndex
              volumeToken0
              volumeToken1
              volumeUSD
              untrackedVolumeUSD
              feesUSD
              txCount
              collectedFeesToken0
              collectedFeesToken1
              collectedFeesUSD
              totalValueLockedToken0
              totalValueLockedToken1
              totalValueLockedETH
              totalValueLockedUSD
              totalValueLockedUSDUntracked
              liquidityProviderCount
            }
            token0: {
              id
              symbol
              name
              decimals
              totalSupply
              volume
              volumeUSD
              untrackedVolumeUSD
              feesUSD
              txCount
              poolCount
              totalValueLocked
              totalValueLockedUSD
              totalValueLockedUSDUntracked
              derivedETH
            }
            token1: {
              id
              symbol
              name
              decimals
              totalSupply
              volume
              volumeUSD
              untrackedVolumeUSD
              feesUSD
              txCount
              poolCount
              totalValueLocked
              totalValueLockedUSD
              totalValueLockedUSDUntracked
              derivedETH
            }
            sender
            recipient
            origin
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
            logIndex
        }
  }
}
"""

block_transactions_query_v2 = """
{
  transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
  blockNumber_in: block_numbers
  }) 
  {
    id
    blockNumber
    timestamp
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

block_transactions_query_v3 = """
{
    transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
    blockNumber_in: block_numbers
    }) 
  {
    id
    blockNumber
    timestamp
    swaps {
        id
        timestamp
        pool
        { 
          id
          feeTier
          liquidity
          sqrtPrice
          createdAtBlockNumber
        }
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
        amount0
        amount1
        sqrtPriceX96
        tick
        logIndex
      }
     mints {
      id
      timestamp
      amount
      amount0
      amount1
      logIndex
      pool
        { 
          id
          feeTier
          liquidity
          sqrtPrice
          createdAtBlockNumber
        }
        token0{
          id
          decimals
          symbol
          name
        }
        token1{
          id
          decimals
          symbol
          name
        }
     }
     burns {
      id
      timestamp
      amount
      amount0
      amount1
      logIndex
      pool
        { 
          id
          feeTier
          liquidity
          sqrtPrice
          createdAtBlockNumber
        }
        token0{
          id
          decimals
          symbol
          name
        }
        token1{
          id
          decimals
          symbol
          name
        }
     }
  }
}
"""

block_transactions_query_v4 = """
{
  transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
  blockNumber_in: block_numbers
  }) 
  {
    id
    blockNumber
    timestamp
    swaps {
        id
        timestamp
        pool
        { 
          id
          feeTier
          liquidity
          sqrtPrice
          createdAtBlockNumber
        }
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
        amount0
        amount1
        sqrtPriceX96
        tick
        logIndex
      }
    modifyLiquiditys {
      id
      timestamp
      pool
      { 
        id
        feeTier
        liquidity
        sqrtPrice
        createdAtBlockNumber
      }
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
      amount
      amount0
      amount1
      logIndex
    }
  }
}
"""


all_transactions_query_v4 = """
{
  transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
  timestamp_gte: timestamp_gte,
  swaps_: {},        # This means at least one swap exists ("swaps" array is non-empty)
  mints_: {},        # This means at least one mint exists, but we want NO mints, so use mints_: null
  burns_: null       # This means the "burns" array is empty
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
        pool
        { 
          id
        }
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
        amount0
        amount1
        sqrtPriceX96
        tick
        logIndex
      }
    modifyLiquiditys {
      id
      timestamp
      pool
      { 
        id
        createdAtBlockNumber
      }
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
      amount
      amount0
      amount1
      logIndex
    }
  }
}
"""

all_transactions_query_v3 = """
{
    transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
    timestamp_gte: timestamp_gte,
    swaps_: {},        # This means at least one swap exists ("swaps" array is non-empty)
    mints_: {},        # This means at least one mint exists, but we want NO mints, so use mints_: null
    burns_: null       # This means the "burns" array is empty
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
        pool
        { 
          id
          createdAtBlockNumber
        }
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
        amount0
        amount1
        sqrtPriceX96
        tick
        logIndex
      }
     mints {
      id
      timestamp
      amount
      amount0
      amount1
      logIndex
      pool
        { 
          id
          createdAtBlockNumber
        }
        token0{
          id
          decimals
          symbol
          name
        }
        token1{
          id
          decimals
          symbol
          name
        }
     }
     burns {
      id
      timestamp
      amount
      amount0
      amount1
      logIndex
      pool
        { 
          id
          createdAtBlockNumber
        }
        token0{
          id
          decimals
          symbol
          name
        }
        token1{
          id
          decimals
          symbol
          name
        }
     }
  }
}
"""

block_transactions_query_v2 = """
{
  transactions(first: first_num, skip: skip_num, orderBy: id, orderDirection: asc, where: {
  timestamp_gte: timestamp_gte,
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