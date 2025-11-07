#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import csv

# Hard-coded configuration
URL = 'https://liquipedia.net/rocketleague/List_of_player_camera_settings'
OUTPUT_FILE = 'camera_settings.csv'

# Fetch the page
print(f"Fetching {URL}...")
response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})
html = response.text

# Parse HTML
soup = BeautifulSoup(html, 'html.parser')
table = soup.find('table', class_='sortable')

# Extract headers
headers = [th.get_text(strip=True) for th in table.find('tr').find_all('th')]

# Extract rows
rows = []
for tr in table.find_all('tr')[1:]:
    row = [td.get_text(strip=True) for td in tr.find_all('td')]
    if row:
        rows.append(row)

# Save to CSV
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
