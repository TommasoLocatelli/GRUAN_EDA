import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gruanpy as gp
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle

folders= [
    r'gdp\products_RS41-GDP-1_PAY_2024',
    r'gdp\products_RS41-GDP-1_SOD_2024',
    r'gdp\products_RS41-GDP-1_TEN_2024'
]  # open folders with chosen GDP files
# I have downloaded GDP files from GRUAN archive for 3 sites for the whole year of 2024

flights = {'PAY': [], 'SOD': [], 'TEN': []}

for folder, label in zip(folders, ['PAY', 'SOD', 'TEN']):
    file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    print(f"{folder}: {file_count} files")

    break # comment this break to produce the code_examples\flights.pkl file
    for file in os.listdir(folder):
        break
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            gdp = gp.read(file_path)  # Assuming read_gdp is the correct function to read GDP files
            print(f"Read GDP data from {file_path}")
            print(f"Global Attributes for {file}:")
            print(f"Global Attributes:")
            print(f"- g.Site.Id: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Id']['Value'].values[0]}")
            print(f"- g.Site.Key: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Key']['Value'].values[0]}")
            print(f"- g.Site.Name: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Name']['Value'].values[0]}")
            print(f"- g.Site.Institution: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Institution']['Value'].values[0]}")
            print(f"- g.Site.Longitude: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Longitude']['Value'].values[0]}")
            print(f"- g.Site.Latitude: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Latitude']['Value'].values[0]}")
            print(f"- g.Site.Altitude: {gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Site.Altitude']['Value'].values[0]}")
            print("________________________________________________________________________________________________________________________________")
            break  # Remove this break to process all files in the folder
    
    for file in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path) and file.endswith('.nc'):
            gdp = gp.read(file_path, only_global_attrs=True)  # Read only global attributes
            start_time = gdp.global_attrs[gdp.global_attrs['Attribute'] == 'g.Measurement.StartTime']['Value'].values[0]
            start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            flights[label].append(start_time)
            #break  # Remove this break to process all files in the folder

    with open(r'code_examples\flights.pkl', 'wb') as f:
        pickle.dump(flights, f)


with open(r'code_examples\flights.pkl', 'rb') as f:
    flights = pickle.load(f)

    records = []
    for site, datetimes in flights.items():
        for dt in datetimes:
            records.append({'site': site, 'datetime': dt})

    df = pd.DataFrame(records)
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour

    flights_per_month = df.groupby(['site', 'month']).size().reset_index(name='COUNT')
    flights_per_hour = df.groupby(['site', 'hour']).size().reset_index(name='COUNT')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=flights_per_month, x='month', y='COUNT', hue='site')
    plt.title('Number of GDP Flights per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Flights')
    plt.legend(title='Site')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=flights_per_hour, x='hour', y='COUNT', hue='site')
    plt.title('Number of GDP Flights per Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Flights')
    plt.legend(title='Site')
    plt.show()