import os
import time
import json
import sys
import argparse
import logging, warnings
from src import download, stats, algo_per_coin, common_coins, aggregate_results, update_graphs, update_stats

# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

def read_config(file_path="config.json"):
    with open(file_path, "r") as file:
        return json.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="We want to make money at the expense of others!")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--block-number", "-b", "--block", type=int, required=False, help="Block number to process")
    parser.add_argument("--log-level", "-l", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the "download" command
    download_parser = subparsers.add_parser("download", help="Download data for a specific block number")
    update_parser = subparsers.add_parser("update", help="Donwload change data for a specific block number")
    update_parser.add_argument("--start", "-s", type=int, required=True, help="Start block number for stats range")
    update_parser.add_argument("--end", "-e", type=int, required=True, help="End block number for stats range")

    # Subparser for the "download-transactions" command
    subparsers.add_parser("download-transactions", help="Download transactions for a specific block number")

    # Subparser for the "add-check-points" command
    add_check_points_parser = subparsers.add_parser("check-points", help="Add check points")
    add_check_points_parser.add_argument("--start", "-s", type=int, required=True, help="Start block number for stats range")
    add_check_points_parser.add_argument("--end", "-e", type=int, required=True, help="End block number for stats range")

    # NEW: Subparser for run algorithms on a specific coin
    run_algorithms_parser = subparsers.add_parser("run", help="Run algorithms on a specific coin")

    run_algorithms_parser.add_argument(
        "--algorithm", "--alg", '-a',
        type=str,
        required=True,
        choices=["bf", "mmbf", "bf2", "ours"],
        # nargs="+",
        help="Algorithm(s) to run"
    )
    
    run_algorithms_parser.add_argument(
        "--query-count", "-q", "--queries",
        type=int,
        required=False,
        default=100,
        help="Number of queries to run (default: 100)"
    )
     
    run_algorithms_parser.add_argument(
        "--max-workers", "-w", "--workers",
        type=int,
        required=False,
        default=None,
        help="Number of workers to use (default: 24)"
    )
    
       
    run_algorithms_parser.add_argument('--coin', type=str, required=False, help="Name of the coin to run algorithms on", default=None)
    
    run_algorithms_parser.add_argument("--start-block-number", '-s', type=int, required=True, help="Start block number to run algorithms on")
    run_algorithms_parser.add_argument("--end-block-number", '-e', type=int, required=True, help="End block number to run algorithms on")
    
    run_algorithms_parser.add_argument(
        "--top-coins-count", "--tc", "-t",
        type=int,
        # nargs="+",
        required=True,
        help="Number(s) of top coins to consider (space/comma separated list)"
    )
    run_algorithms_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force execution even if results exist"
    )


    # Subparser for the "aggregate" command
    aggregate_parser = subparsers.add_parser("agg", help="Aggregate results")
    aggregate_parser.add_argument("--start-block-number", "--start", "-s", type=int, required=False, default=None, help="Start block number for stats range")
    aggregate_parser.add_argument("--end-block-number", "--end", "-e", type=int, required=False, default=None, help="End block number for stats range")
    
    # Subparser for the "solve" command
    solve_parser = subparsers.add_parser("solve", help="Solve data for a specific block number")
    

    # Subparser for the "stats" command
    stats_parser = subparsers.add_parser("stats", help="Generate stats")
    stats_parser.add_argument("--start-block-number", "--start", "-s", type=int, required=False, default=None, help="Start block number for stats range")
    stats_parser.add_argument("--end-block-number", "--end", "-e", type=int, required=False, default=None, help="End block number for stats range")
    stats_parser.add_argument(
        "--max-workers", "-w", "--workers",
        type=int,
        required=False,
        default=None,
        help="Number of workers to use (default: 24)"
    )

    # Subparser for the "update-stats" command
    update_stats_parser = subparsers.add_parser("update-stats", help="Generate update stats")

    args = parser.parse_args()

    config = read_config(args.config)



    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback", category=RuntimeWarning)
    

    if args.command == "download":
        download.download_all(config)

    elif args.command == "update":
        download.update(config, args.start, args.end)

    elif args.command == "check-points":
        update_graphs.add_check_points(config, args.start, args.end)

    elif args.command == "solve":
        raise NotImplementedError()
    
    elif args.command == "run":
        start_time = time.time()
        algo_per_coin.run_algorithm_from_to(
            algorithm_name=args.algorithm, 
            start_block_number=args.start_block_number,
            end_block_number=args.end_block_number,
            top_coins_count=args.top_coins_count,
            query_count=args.query_count,
            force=args.force,
            max_workers=args.max_workers,
            config=config,
        )
        print(f"######## Total: {time.time() - start_time:.2f}s ########")

    elif args.command == "agg":
        aggregate_results.aggregate_csv_files(start_block_number=args.start_block_number, end_block_number=args.end_block_number, config=config)

    elif args.command == "stats":
        stats.aggregate_stats(start_block_number = args.start_block_number, end_block_number=args.end_block_number, max_workers=args.max_workers, config=config)

    elif args.command == "update-stats":
        update_stats.update_stats(config)

