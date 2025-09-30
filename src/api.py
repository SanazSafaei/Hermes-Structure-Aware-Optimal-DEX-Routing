import os
import requests
import time

def the_graph_query_wrapper(query, the_graph_token, variables = {} , config = None, version = 3):

    try:
        if config :
            endpoint = config['services']['graph_endpoints'][version]
        else:
            # id is uniswap v3 subgraph
            endpoint = 'https://gateway.thegraph.com/api/subgraphs/id/{your-subgraph-id}'

        all_data = []
        headers = {
            'Authorization': f"Bearer {the_graph_token}",
            'Content-Type': 'application/json'
        }
        response = requests.post(
            endpoint,
            json={"query":query, "variables":variables},
            headers=headers
        )
    except Exception as error:
        raise Exception(f"Error in the_graph_query_wrapper: {error}")

    if response.status_code == 200:
        if "errors" in response.json():
            print(response.json()['errors'], the_graph_token)
            # raise Exception(f"Error in the_graph_query_wrapper: {response.json()['error']}")
        return response.json()


def the_graph_query_wrapper_by_endpoint(query, the_graph_token, endpoint, variables = {} ):

    try:
        
        all_data = []
        headers = {
            'Authorization': f"Bearer {the_graph_token}",
            'Content-Type': 'application/json'
        }
        response = requests.post(
            endpoint,
            json={"query":query, "variables":variables},
            headers=headers
        )
    except Exception as error:
        raise Exception(f"Error in the_graph_query_wrapper: {error}")

    if response.status_code == 200:
        return response.json()
    
def get_eth_block_number(rpc_url):
        """
        Calls the eth_blockNumber method on a JSON-RPC Ethereum endpoint.

        Args:
            rpc_url (str): The Ethereum JSON-RPC endpoint URL.

        Returns:
            int: The latest block number.

        Raises:
            Exception: If the request fails or the response is invalid.
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1
            }
            headers = {
                "Content-Type": "application/json"
            }
            response = requests.post(rpc_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json().get("result")
            if result is None:
                raise Exception("No result in eth_blockNumber response")
            return int(result, 16)
        except Exception as error:
            raise Exception(f"Error in get_eth_block_number: {error}")
