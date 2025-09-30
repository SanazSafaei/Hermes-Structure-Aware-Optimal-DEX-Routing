import api
import query
import config
import pandas as pd
from datetime import datetime

def get_top_tvl_tokens(config):
    count = 100000
    all_data = []
    
    for i in range(0, count, 1000):
        query_str = query.most_tvl_tokens_day_query.replace("first_num", str(i)).replace("skip_num", str(count-i))
        res = api.the_graph_query_wrapper(query_str, config['services']['the_graph_tokens'], config=config)
        data = res['data']['tokenDayDatas']
        all_data.extend(data)
        print(f"Fetched {len(data)} records, total so far: {len(all_data)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)

    #sum duplicate tokens
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"top_tvl_tokens_{timestamp}.csv.gz"
    
    # Save as compressed CSV
    df.to_csv(filename, compression='gzip', index=False)
    print(f"Data saved to {filename} ({len(df)} records)")
    
    return df


if __name__ == "__main__":
    config = config.load_config()
    res = get_top_tvl_tokens(config)
    print(f"Final dataset shape: {res.shape}")