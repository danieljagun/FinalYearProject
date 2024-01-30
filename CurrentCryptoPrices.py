import csv
import json
import os
import urllib.request
from datetime import datetime
import time  # Import the time module

url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1' \
      '&sparkline=false'
req = urllib.request.Request(url)

r = urllib.request.urlopen(req).read()

cont = json.loads(r.decode('utf-8'))

# Create a new folder to save CSV files
output_folder = 'coin_data_csv'
os.makedirs(output_folder, exist_ok=True)

time.sleep(1)  # Pause for 1 second between requests

csv_file_path = os.path.join(output_folder, 'coingecko_top_10.csv')

with open(csv_file_path, 'w', encoding='utf-8') as csvfile:
    headers = ['Coin', 'Item', 'Prices', 'MrkCap', 'TolVol']
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    writer.writeheader()
    for coin in cont:
        coin_id = coin['id']
        coin_url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=max&interval' \
                   f'=daily&precision=1'
        coin_req = urllib.request.Request(coin_url)

        while True:
            try:
                coin_r = urllib.request.urlopen(coin_req).read()
                break  # Exit the loop if the request is successful
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print("Too many requests. Waiting for 5 seconds...")
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    raise

        coin_cont = json.loads(coin_r.decode('utf-8'))

        flatten_data = {}
        for key in coin_cont:
            for value in coin_cont[key]:
                if value[0] not in flatten_data:
                    flatten_data[value[0]] = {}
                flatten_data[value[0]].update({key: value[1]})

        for k, v in flatten_data.items():
            v.update({'Coin': coin['name'], 'Item': datetime.utcfromtimestamp(k / 1000).strftime('%Y-%m-%d %H:%M:%S')})
            v['Prices'] = v.pop('prices')
            v['MrkCap'] = v.pop('market_caps')
            v['TolVol'] = v.pop('total_volumes')
            writer.writerow(v)

print(f"CSV file creation complete. Saved at {csv_file_path}")
