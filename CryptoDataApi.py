# Using MongoDB and Coin data from when created.

import csv
import json
import urllib.request
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1' \
      '&sparkline=false'
req = urllib.request.Request(url)

r = urllib.request.urlopen(req).read()

cont = json.loads(r.decode('utf-8'))

with open('coingecko_top_10.csv', 'w', encoding='utf-8') as csvfile:
    headers = ['Coin', 'Item', 'Prices', 'MrkCap', 'TolVol']
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    writer.writeheader()
    for coin in cont:
        coin_id = coin['id']
        coin_url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=max&interval' \
                   f'=daily&precision=1'
        coin_req = urllib.request.Request(coin_url)
        coin_r = urllib.request.urlopen(coin_req).read()
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

print("CSV file creeation complete")

client = MongoClient('mongodb://localhost:27017/')
db = client['crypto-data']
collection = db['coins']

# Read the CSV file into a DataFrame
df = pd.read_csv('coingecko_top_10.csv')

# Convert DataFrame to dictionary records
data = df.to_dict(orient='records')

# Insert data only if it doesn't already exist
for record in data:
    if collection.count_documents(record, limit=1) == 0:
        collection.insert_one(record)
        print(f"Data for {record['Coin']} inserted into MongoDB successfully.")
    else:
        print(f"Data for {record['Coin']} already exists in MongoDB. Skipping insertion.")

print("Coin data added.")
