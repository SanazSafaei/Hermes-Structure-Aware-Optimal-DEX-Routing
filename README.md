# Hermes: Uniswap Arbitrage Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **"Hermes: Scalable and Robust Structure-Aware Optimal Routing for Decentralized Exchanges"** - A high-performance tool for detecting arbitrage opportunities in Uniswap DEX using treewidth-based shortest path queries.

## 🎯 Overview

This project implements the **Hermes Tool**, a breakthrough approach for finding optimal trading routes in decentralized exchanges. By leveraging treewidth decomposition, Hermes achieves **O(n · (tw(G) + 1)²) query time** where tw is the treewidth - dramatically faster than traditional O(VE) algorithms for large DeFi trading graphs.

### Why Hermes?

- **Ultra-Fast Queries**: O(n · (tw(G) + 1)²) time complexity vs O(VE) for traditional algorithms
- **Scalable**: Handles thousands of tokens and trading pairs efficiently  
- **Arbitrage Detection**: Perfect for finding profitable trading cycles
- **⚡ Real-time**: Optimized for high-frequency arbitrage detection

## How Hermes Works

The Hermes algorithm transforms complex DeFi trading graphs into efficiently queryable structures:

1. **Tree Decomposition**: Uses FlowCutter to compute optimal treewidth of the trading graph
2. **Graph Triangulation**: Adds strategic "shortcut" edges to create a triangulated graph
3. **Perfect Elimination Ordering**: Creates optimal vertex ordering for lightning-fast queries
4. **Two-Pass Algorithm**: 
   - **Downward sweep**: Compute distances from source to all vertices
   - **Upward sweep**: Refine distances using elimination ordering
5. **Path Reconstruction**: Expand shortcut edges to reveal actual trading paths

**Key Innovation**: Preprocessing trade-off - one-time expensive setup enables thousands of fast queries

## 🛠️ Installation

### Prerequisites
- **Python 3.8+** (tested with 3.8-3.11)
- **GCC 7.5+** (for FlowCutter compilation)
- **Unix-like system** (Linux/macOS/WSL)
- **8GB+ RAM** (recommended for large graphs)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/uniswap-crawler.git
cd uniswap-crawler

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Build FlowCutter (required for treewidth computation)
cd flow-cutter-pace20
./build.sh
cd ..

# 5. Configure API keys
cp config.json.example config.json
# Edit config.json with your The Graph API tokens and Ethereum RPC endpoints
```


## 🚀 Using Hermes Algorithm

### Python API Usage

```python
from src.uniprice import UniPrice
import networkx as nx

# 1. Create your trading graph (weights are -log(price) for arbitrage detection)
G = nx.DiGraph()
G.add_edge('USDC', 'WETH', weight=-5.2)  # USDC → WETH
G.add_edge('WETH', 'USDT', weight=-3.1)  # WETH → USDT
G.add_edge('USDT', 'USDC', weight=-4.8)  # USDT → USDC (completing cycle)

# 2. Initialize Hermes algorithm
uniprice = UniPrice()
uniprice.accept_graph(G)

# 3. Query shortest path (finds arbitrage opportunities)
result = uniprice.query_path('USDC', 'USDC')  # Self-loop for arbitrage
print(f"Arbitrage Profit: {result['distances']}")
print(f"Trading Path: {result['path']}")

# 4. Query between different tokens
result = uniprice.query_path('USDC', 'USDT')
print(f"Best Exchange Rate: {result['distances']}")
print(f"Optimal Path: {result['path']}")
```

### Command Line Usage

```bash
# Download Uniswap data for a specific block range
python main.py download --block-number 18000000

# Run Hermes algorithm on real Uniswap data
python main.py run \
  --algorithm ours \
  --start-block-number 18000000 \
  --end-block-number 18000100 \
  --top-coins-count 100 \
  --query-count 1000

# Compare Hermes with traditional algorithms
python main.py run --algorithm bf --start-block-number 18000000 --end-block-number 18000100 --top-coins-count 100

# Generate performance statistics
python main.py stats --start-block-number 18000000 --end-block-number 18000100
```

### Advanced Usage

```bash
# Multi-threaded processing for large datasets
python main.py run --algorithm ours --max-workers 24 --query-count 10000

# Force re-computation (skip cached results)
python main.py run --algorithm ours --force --query-count 1000

# Run on specific token subset
python main.py run --algorithm ours --coin WETH --query-count 500
```


## 🏗️ Project Structure

```
uniswap-crawler/
├── src/
│   ├── uniprice.py          # 🧠 Hermes algorithm implementation
│   ├── sssp.py              # Traditional shortest path algorithms
│   ├── download.py          # Uniswap data fetching
│   ├── graph.py             # Graph utilities
│   └── algo_per_coin.py     # Benchmarking framework
├── flow-cutter-pace20/      # Tree decomposition library
├── data/                    # Processed data and results
├── notebooks/               # Analysis and visualization
└── main.py                  # Command-line interface
```

## 🔧 Configuration

Edit `config.json` to customize your setup:

```json
{
  "services": {
    "ethereum_providers": ["https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY"],
    "the_graph_tokens": ["YOUR_GRAPH_TOKEN"],
    "graph_endpoints": ["https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"]
  },
  "paths": {
    "data": "./data"
  },
  "download": {
    "uniswap_batch_size": 5,
    "n_threads": 8
  }
}
```

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**. Always verify arbitrage opportunities and consider gas costs before executing trades in production environments. The authors are not responsible for any financial losses.

