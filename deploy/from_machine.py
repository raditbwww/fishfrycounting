import csv
import json
import requests
import pandas as pd

# Step 1: Convert the CSV to JSON
csv_file_path = 'dummy_data.csv'
json_file_path = 'dummy_data.json'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert DataFrame to JSON
json_data = df.to_dict(orient='records')

# # Step 2: Use requests to get data from localhost:5000
response_device = requests.get('http://localhost:5000/api/devices')
response_pool = requests.get('http://localhost:5000/api/pools')
response_location = requests.get('http://localhost:5000/api/locations')
response_fish_type = requests.get('http://localhost:5000/api/fish-types')
response_fish_size = requests.get('http://localhost:5000/api/fish-sizes')

devices = response_device.json()
pools = response_pool.json()
locations = response_location.json()
fish_types = response_fish_type.json()
fish_sizes = response_fish_size.json()

# Create dictionaries for quick lookup, making the keys lowercase
device_dict = {device['device_name'].lower(): device['id'] for device in devices}
pool_dict = {pool['pool_name'].lower(): pool['id'] for pool in pools}
location_dict = {location['location_name'].lower(): location['id'] for location in locations}
fish_type_dict = {fish_type['fish_type_name'].lower(): fish_type['id'] for fish_type in fish_types}
fish_size_dict = {fish_size['size_category'].lower(): fish_size['id'] for fish_size in fish_sizes}

# Step 3: Convert the names from the JSON to the IDs from the response
for record in json_data:
    record['device_id'] = device_dict.get(record['device_id'].lower())
    record['pool_id'] = pool_dict.get(record['pool_id'].lower())
    record['location_id'] = location_dict.get(record['location_id'].lower())
    record['fish_type_id'] = fish_type_dict.get(record['fish_type_id'].lower())
    record['fish_size_id'] = fish_size_dict.get(record['fish_size_id'].lower())


# print(json.dumps(json_data, indent=4))

# Step 4: Post the JSON data to localhost:5000/api/data 
post_url = 'http://localhost:5000/api/data' 
headers = {'Content-Type': 'application/json'} 

# Check the response from the server 
for record in json_data: 
    response = requests.post(post_url, headers=headers, json=record)
    if response.status_code == 201: 
        print('Data posted successfully!') 
    else: 
        print('Failed to post data:', response.status_code, response.text)