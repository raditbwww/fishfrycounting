import csv
import json
import requests
import pandas as pd

# Supabase credentials
supabase_url = 'https://ghzbymqxacfunikvkzbv.supabase.co'
supabase_api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdoemJ5bXF4YWNmdW5pa3ZremJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ4NDEyMzEsImV4cCI6MjA1MDQxNzIzMX0.M4kwIXgPOUrkE84mhlXDeQd18l7W9QJAJELSKEEdYJY'
supabase_table = 'fish_data'

# Headers for Supabase API
headers = {
    'apikey': supabase_api_key,
    'Authorization': f'Bearer {supabase_api_key}',
    'Content-Type': 'application/json'
}

# Step 1: Convert the CSV to JSON
csv_file_path = 'dummy_data.csv'
json_file_path = 'dummy_data.json'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Convert DataFrame to JSON
json_data = df.to_dict(orient='records')

# Step 2: Use requests to get data from Supabase
response_device = requests.get(f'{supabase_url}/rest/v1/devices', headers=headers)
response_pool = requests.get(f'{supabase_url}/rest/v1/pools', headers=headers)
response_location = requests.get(f'{supabase_url}/rest/v1/locations', headers=headers)
response_fish_type = requests.get(f'{supabase_url}/rest/v1/fish_types', headers=headers)
response_fish_size = requests.get(f'{supabase_url}/rest/v1/fish_sizes', headers=headers)

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

# Step 4: Post the JSON data to Supabase
for record in json_data:
    response = requests.post(f'{supabase_url}/rest/v1/{supabase_table}', headers=headers, json=record)
    if response.status_code in (200, 201):
        print('Data posted successfully!')
    else:
        print('Failed to post data:', response.status_code, response.text)
