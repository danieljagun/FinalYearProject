import csv
import json
import os
import urllib.request
from datetime import datetime
import time
import pandas as pd
from pymongo import MongoClient

mongo_uri = "mongodb+srv://danieljagun:Daniel202@cluster.tokbwgs.mongodb.net/?retryWrites=true&w=majority"

coins_list = ['bitcoin', 'ethereum', 'solana', 'dogecoin', 'cardano', 'avalanche-2']
coins_ids = ','.join(coins_list)

url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coins_ids}&order=market_cap_desc' \
      f'&per_page={len(coins_list)}&page=1&sparkline=false'

req = urllib.request.Request(url)
r = urllib.request.urlopen(req).read()
cont = json.loads(r.decode('utf-8'))

client = MongoClient(mongo_uri)
db = client['crypto']

folder_name = "crypto_data"
os.makedirs(folder_name, exist_ok=True)


def normalize_coin_name(coin_name):
    return coin_name.replace('avalanche-2', 'avalanche').lower()


for coin in cont:
    coin_id = coin['id']
    normalized_coin_name = normalize_coin_name(coin['name'])

    coin_url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=365&interval' \
               f'=daily&precision=1'
    coin_req = urllib.request.Request(coin_url)
    coin_r = urllib.request.urlopen(coin_req).read()
    coin_cont = json.loads(coin_r.decode('utf-8'))

    folder_name = "crypto_data"
    os.makedirs(folder_name, exist_ok=True)

    time.sleep(1)

    flatten_data = {}
    for key in coin_cont:
        for value in coin_cont[key]:
            if value[0] not in flatten_data:
                flatten_data[value[0]] = {}
            flatten_data[value[0]].update({key: value[1]})

    csv_filename = os.path.join(folder_name, f"{normalized_coin_name}.csv")
    with open(csv_filename, 'w', encoding='utf-8') as csvfile:
        headers = ['Coin', 'Timestamp', 'Prices', 'MrkCap', 'TolVol']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for k, v in flatten_data.items():
            v.update({'Coin': normalized_coin_name, 'Timestamp': datetime.utcfromtimestamp(k / 1000).strftime('%Y-%m-%d %H:%M:%S')})
            v['Prices'] = v.pop('prices')
            v['MrkCap'] = v.pop('market_caps')
            v['TolVol'] = v.pop('total_volumes')
            writer.writerow(v)

        print(f"CSV file '{csv_filename}' creation complete")

        coin_collection = db[normalized_coin_name]
        df = pd.read_csv(csv_filename)
        data = df.to_dict(orient='records')

        for record in data:
            if coin_collection.count_documents(record, limit=1) == 0:
                coin_collection.insert_one(record)
                print(f"Data for {normalized_coin_name} inserted into MongoDB Atlas successfully.")
            else:
                print(f"Data for {normalized_coin_name} already exists in MongoDB Atlas. Skipping insertion.")

        print(f"Coin data for '{normalized_coin_name}' added to MongoDB Atlas")

