import asyncio
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from decimal import Decimal
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_updated_pool_prices(block_number: int, subgraph_endpoint: str) -> list:
    """
    Fetch transactions for a given block number and return updated pool prices.
    
    Args:
        block_number (int): Ethereum block number to query.
        subgraph_endpoint (str): GraphQL API endpoint for Uniswap V3 subgraph.
    
    Returns:
        list: List of dicts with pool_id, token0Price, and token1Price for affected pools.
    
    Raises:
        Exception: If the GraphQL query fails or data is malformed.
    """
    try:
        # Initialize GraphQL client with SSL verification
        transport = AIOHTTPTransport(url=subgraph_endpoint, ssl=True)
        async with Client(transport=transport, fetch_schema_from_transport=True) as client:
            # GraphQL query to fetch transactions and their swaps for the block
            query = gql("""
                query($blockNumber: BigInt!) {
                    transactions(where: { blockNumber: $blockNumber }) {
                        id
                        swaps {
                            pool {
                                id
                            }
                            sqrtPriceX96
                        }
                    }
                }
            """)
            
            # Execute query
            params = {"blockNumber": str(block_number)}
            result = await client.execute(query, variable_values=params)
            
            # Process swaps to extract updated prices
            updated_pools = {}
            for tx in result.get("transactions", []):
                for swap in tx.get("swaps", []):
                    pool_id = swap["pool"]["id"]
                    sqrt_price_x96 = int(swap["sqrtPriceX96"])
                    
                    # Calculate prices
                    Q96 = 2 ** 96
                    sqrt_price = Decimal(sqrt_price_x96) / Decimal(Q96)
                    token0_price = sqrt_price ** 2  # token1 per token0
                    token1_price = Decimal(1) / token0_price if token0_price != 0 else Decimal(0)
                    
                    # Store latest price for each pool (last swap in block prevails)
                    updated_pools[pool_id] = {
                        "pool_id": pool_id,
                        "token0Price": float(token0_price),
                        "token1Price": float(token1_price)
                    }
            
            # Convert to list
            return list(updated_pools.values())
    
    except Exception as e:
        logger.error(f"Error fetching pool prices for block {block_number}: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Example subgraph endpoint (replace with actual Uniswap V3 subgraph URL)
    SUBGRAPH_ENDPOINT = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    BLOCK_NUMBER = 22772679  # Replace with desired block number
    
    async def main():
        try:
            pools = await get_updated_pool_prices(BLOCK_NUMBER, SUBGRAPH_ENDPOINT)
            for pool in pools:
                logger.info(f"Pool ID: {pool['pool_id']}, Token0Price: {pool['token0Price']}, Token1Price: {pool['token1Price']}")
        except Exception as e:
            logger.error(f"Failed to run main: {str(e)}")
    
    asyncio.run(main())