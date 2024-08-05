from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import urllib.parse

# Database connection information
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'db_evcharger_0731'
}

# Create the database URL (URL-encode the password to handle special characters)
db_url = f"mysql+pymysql://{db_config['user']}:{urllib.parse.quote_plus(db_config['password'])}@{db_config['host']}/{db_config['database']}"

# Create the SQLAlchemy engine
engine = create_engine(db_url)

# Updated SQL query, including weather information
query = """
SELECT 
    a.city_id,
    a.station_name,
    a.connector_id,
    si.coordinates_x,
    si.coordinates_y,
    si.postcode,
    s.tariff_amount,
    s.tariff_connectionfee,
    c.max_chargerate,
    c.plug_type,
    c.connector_type,
    a.date,
    a.hour,
    w.weather,
    a.is_available
FROM 
    availability a
JOIN connector c ON a.station_name = c.station_name AND a.connector_id = c.connector_id
JOIN station s ON a.station_name = s.station_name
JOIN site si ON s.site_id = si.site_id
LEFT JOIN weather w ON a.city_id = w.city_id AND a.date = w.date AND a.hour = w.hour
"""

try:
    # Read SQL query results using pandas
    df = pd.read_sql_query(query, engine)

    # Data processing
    # Combine date and hour into a datetime column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')

    # Remove the original date and hour columns
    df = df.drop(['date', 'hour'], axis=1)

    # Rearrange the order of columns
    columns_order = [
        'city_id', 'station_name', 'connector_id', 'coordinates_x', 'coordinates_y',
        'postcode', 'tariff_amount', 'tariff_connectionfee', 'max_chargerate',
        'plug_type', 'connector_type', 'datetime', 'weather', 'is_available'
    ]
    df = df[columns_order]


    # Create CSV filename
    csv_filename = f'charging_station_data.csv'

    # Save the data as a CSV file
    df.to_csv(csv_filename, index=False)

    print(f"File has been saved in {csv_filename}.")

except Exception as e:
    print(f"Error: {e}")

finally:
    engine.dispose()
    print("The database connection has been closed.")


# Import necessary libraries
import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('charging_station_data.csv')
df = df.reset_index(drop=True)
# Display basic information of the data
print(df.info())

# Check the number of null values ​​in the tariff_amount column
print("\n Number of null value:")
print(df['tariff_amount'].isnull().sum())

# Define a function to fill in the empty value of tariff_amount
def fill_tariff_amount(group):
    mask = group['tariff_amount'].isnull()
    group.loc[mask, 'tariff_amount'] = group['tariff_amount'].mean()
    return group

# Group by max_chargerate, plug_type, connector_type and apply fill function
df = df.groupby(['max_chargerate', 'plug_type', 'connector_type'], group_keys=False).apply(fill_tariff_amount)

# Check the number of null values ​​in the tariff_amount column again
print("\n The number of empty values ​​after filling:")
print(df['tariff_amount'].isnull().sum())

# Display the padded data sample
print("\n Data sample after padding:")
print(df[['max_chargerate', 'plug_type', 'connector_type', 'tariff_amount']].sample(10))

# Check if the padding is correct
print("\n The average of each group's tariff_amount:")
print(df.groupby(['max_chargerate', 'plug_type', 'connector_type'])['tariff_amount'].mean())

# Check the number of null values ​​in the tariff_connectionfee column
print("\nNumber of empty values ​​in tariff_connectionfee column (before filling):")
print(df['tariff_connectionfee'].isnull().sum())

# Fill empty values ​​in the tariff_connectionfee column with 0
df['tariff_connectionfee'] = df['tariff_connectionfee'].fillna(0)

# Check the number of null values ​​in the tariff_connectionfee column again to ensure that the fill was successful
print("\nThe number of null values ​​in the tariff_connectionfee column (after filling):")
print(df['tariff_connectionfee'].isnull().sum())

# Display the padded data sample
print("\nSample data after padding:")
print(df[['station_name', 'tariff_amount', 'tariff_connectionfee']].sample(10))

# Check basic statistics of the tariff_connectionfee column
print("\nBasic statistics for the tariff_connectionfee column:")
print(df['tariff_connectionfee'].describe())

# Display the number of rows of original data
print(f"Number of original data rows: {len(df)}")

# Check the number of 'UNKNOWN' in each column
columns_to_check = ['max_chargerate', 'plug_type', 'connector_type']
for col in columns_to_check:
    unknown_count = (df[col] == 'UNKNOWN').sum()
    print(f"Number of 'UNKNOWN' in {col}: {unknown_count}")

# Delete the lines containing 'UNKNOWN'
df_cleaned = df[~df[columns_to_check].isin(['UNKNOWN']).any(axis=1)]

# Display the number of rows of data after cleaning
print(f"\nNumber of cleaned data rows: {len(df_cleaned)}")

# Check again if there are any 'UNKNOWN' values
for col in columns_to_check:
    unknown_count = (df_cleaned[col] == 'UNKNOWN').sum()
    print(f"Number of remaining 'UNKNOWN' in {col}: {unknown_count}")

# Display the cleaned data sample
print("\nSample of cleaned data:")
print(df_cleaned[columns_to_check + ['tariff_amount']].sample(10))

# Convert max_chargerate to a numeric type (if it isn't already)
df_cleaned['max_chargerate'] = pd.to_numeric(df_cleaned['max_chargerate'], errors='coerce')

# Display basic statistics of the cleaned data
print("\nBasic statistics of the cleaned data:")
print(df_cleaned[columns_to_check + ['tariff_amount']].describe())

# Update the original DataFrame
df = df_cleaned.copy()

# Ensure that the datetime column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by station_name and datetime
df = df.sort_values(['station_name', 'datetime'])

# Check the number of null values ​​in the weather column (before processing)
print("Number of null values ​​in Weather column (before processing):")
print(df['weather'].isnull().sum())

# Define a function to fill in the empty values ​​of the weather column
def fill_weather(group):
    group['weather'] = group['weather'].fillna(method='ffill')
    group['weather'] = group['weather'].fillna(method='bfill')
    return group

# Apply the fill function to each charging station group
df = df.groupby('station_name').apply(fill_weather)

# Reset index
df = df.reset_index(drop=True)

# Check the number of null values ​​in the weather column (after processing)
print("\nNumber of null values ​​in Weather column (after processing):")
print(df['weather'].isnull().sum())

# Print index information to confirm
print("\nIndex information:")
print(df.index)

# Define a function to remove the quotation marks at both ends of the string
def remove_quotes(text):
    if isinstance(text, str):
        return text.strip("'\"")
    return text

# Apply a function to the weather column
df['weather'] = df['weather'].apply(remove_quotes)

# Check the processing results
print("The processed unique values ​​of the weather column:")
print(df['weather'].unique())

# Display weather types and their frequencies
print("\nWeather types and their frequency:")
print(df['weather'].value_counts())

# Display weather records containing quotes (if any)
quotes = df[df['weather'].str.contains("'|\"", na=False)]
if not quotes.empty:
    print("\nRecords that still contain quotes:")
    print(quotes[['station_name', 'datetime', 'weather']])
else:
    print("\nAll quotes removed successfully.")

# Display the processed data samples
print("\nSample of processed data:")
print(df[['station_name', 'datetime', 'weather']].sample(10))

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_holiday'] = ((df['datetime'].dt.dayofweek.isin([5, 6])) | 
                    ((df['datetime'].dt.dayofweek == 4) & (df['hour'] >= 18))).astype(int)
df['time_of_day'] = pd.cut(df['hour'], 
                           bins=[0, 6, 12, 18, 24], 
                           labels=False,
                           include_lowest=True)
df['is_work_hour'] = ((df['day_of_week'].isin([0,1,2,3,4])) & (df['hour'].between(9, 17))).astype(int)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Create a OneHotEncoder object
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# One-hot encode 'plug_type'
plug_type_encoded = ohe.fit_transform(df[['plug_type']])
plug_type_columns = [f'plug_type_{cat}' for cat in ohe.categories_[0]]
df_plug_type = pd.DataFrame(plug_type_encoded, columns=plug_type_columns, index=df.index)

# One-hot encode 'connector_type'
connector_type_encoded = ohe.fit_transform(df[['connector_type']])
connector_type_columns = [f'connector_type_{cat}' for cat in ohe.categories_[0]]
df_connector_type = pd.DataFrame(connector_type_encoded, columns=connector_type_columns, index=df.index)

# One-hot encode 'weather'
weather_encoded = ohe.fit_transform(df[['weather']])
weather_columns = [f'weather_{cat}' for cat in ohe.categories_[0]]
df_weather = pd.DataFrame(weather_encoded, columns=weather_columns, index=df.index)

#Merge the encoded features into the original data frame
df = pd.concat([df, df_plug_type, df_connector_type, df_weather], axis=1)

# Make sure station_name and connector_id are both string types
df['station_name'] = df['station_name'].astype(str)
df['connector_id'] = df['connector_id'].astype(str)

# Use str.cat method to create connector_unique_id
df['connector_unique_id'] = df['station_name'].str.cat(df['connector_id'], sep='_')
df['connector_avg_usage'] = 1 - df.groupby('connector_unique_id')['is_available'].transform('mean')
df['station_avg_usage'] = 1 - df.groupby('station_name')['is_available'].transform('mean')
df = df.sort_values(['station_name', 'connector_id', 'datetime'])
df['usage_last_24h'] = df.groupby('connector_unique_id')['is_available'].transform(
    lambda x: 1 - x.rolling(window=24, min_periods=1).mean()
)

df['usage_last_7d'] = df.groupby('connector_unique_id')['is_available'].transform(
    lambda x: 1 - x.rolling(window=24*7, min_periods=1).mean()
)

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean


# Distance to city center
city_centers = df.groupby('city_id')[['coordinates_x', 'coordinates_y']].mean()

def distance_to_center(row):
    city_center = city_centers.loc[row['city_id']]
    return euclidean((row['coordinates_x'], row['coordinates_y']), 
                     (city_center['coordinates_x'], city_center['coordinates_y']))

df['distance_to_center'] = df.apply(distance_to_center, axis=1)

# Charging station density in each city
stations_per_city = df.groupby('city_id')['station_name'].nunique()

def safe_city_area(group):
    x_range = group['coordinates_x'].max() - group['coordinates_x'].min()
    y_range = group['coordinates_y'].max() - group['coordinates_y'].min()
    area = x_range * y_range
    return max(area, 1e-10)  # Use a small positive number instead of 0

city_areas = df.groupby('city_id').apply(safe_city_area)
city_density = stations_per_city / city_areas
city_density = city_density.replace([np.inf, -np.inf], np.nan)
city_density = city_density.fillna(city_density.mean())

df['city_station_density'] = df['city_id'].map(city_density)

# Use quantiles to divide density levels
def density_to_level_quantile(density, q_dict):
    for level, threshold in sorted(q_dict.items(), key=lambda x: x[1], reverse=True):
        if density >= threshold:
            return level
    return 1  # If the density is less than all thresholds, return the lowest level

quantiles = city_density.quantile([0.2, 0.4, 0.6, 0.8])
q_dict = {5: quantiles[0.8], 4: quantiles[0.6], 3: quantiles[0.4], 2: quantiles[0.2], 1: 0}

df['city_density_level'] = df['city_id'].map(city_density).apply(lambda x: density_to_level_quantile(x, q_dict))

# View the results
print("The first few lines of the new feature:")
print(df[['city_id', 'station_name', 'distance_to_center', 'city_station_density', 'city_density_level']].head(10))

print("\nStatistics of the new features:")
print(df[['distance_to_center', 'city_station_density']].describe())

print("\nDistribution of density levels:")
print(df['city_density_level'].value_counts().sort_index())

print("\nDensity range for each density level:")
for level, threshold in sorted(q_dict.items()):
    if level == 5:
        print(f"Level {level}: >= {threshold:.2f}")
    elif level == 1:
        print(f"Level {level}: < {q_dict[2]:.2f}")
    else:
        print(f"Level {level}: {threshold:.2f} - {q_dict[level+1]:.2f}")

# Checking for null values
print("\nCheck for null values ​​for new features:")
print(df[['distance_to_center', 'city_station_density', 'city_density_level']].isnull().sum())

# Calculate the number of connectors per charging station
connectors_per_station = df.groupby('station_name')['connector_id'].nunique()
df['station_connector_count'] = df['station_name'].map(connectors_per_station)

# Calculate the average maximum charging rate for each charging station
avg_max_chargerate = df.groupby('station_name')['max_chargerate'].mean()
df['station_avg_max_chargerate'] = df['station_name'].map(avg_max_chargerate)

# View Results
print("The first few lines of the new feature:")
print(df[['station_name', 'station_connector_count', 'station_avg_max_chargerate']].head(10))

print("\nStatistics of the new features:")
print(df[['station_connector_count', 'station_avg_max_chargerate']].describe())

# Checking for null values
print("\nCheck for null values ​​for new features:")
print(df[['station_connector_count', 'station_avg_max_chargerate']].isnull().sum())
import pandas as pd
import numpy as np

# Ensure that the datetime column is of datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by charging station and time
df = df.sort_values(['connector_unique_id', 'datetime'])

# Calculate the availability at the same time the previous day
df['availability_24h_ago'] = df.groupby('connector_unique_id')['is_available'].shift(24)

# Calculate availability at the same time the previous week
df['availability_1week_ago'] = df.groupby('connector_unique_id')['is_available'].shift(24 * 7)

# Fill the empty value with the current is_available value
df['availability_24h_ago'] = df['availability_24h_ago'].fillna(df['is_available'])
df['availability_1week_ago'] = df['availability_1week_ago'].fillna(df['is_available'])

# View Results
print("The first few lines of the new feature:")
print(df[['connector_unique_id', 'datetime', 'is_available', 'availability_24h_ago', 'availability_1week_ago']].head(20))

print("\nStatistics of the new features:")
print(df[['is_available', 'availability_24h_ago', 'availability_1week_ago']].describe())

# Check for null values ​​(should all be 0)
print("\n检查新特征的空值：")
print(df[['availability_24h_ago', 'availability_1week_ago']].isnull().sum())

# Check if the filled values ​​are in the same proportion as is_available
print("\nRatio of availability 24 hours ago to current availability:")
print((df['availability_24h_ago'] == df['is_available']).mean())

print("\nRatio of availability one week ago to current availability:")
print((df['availability_1week_ago'] == df['is_available']).mean())
import numpy as np
from sklearn.neighbors import BallTree

def calculate_unique_station_density(df, radius_km=5):
    # First, we need to get the unique charging station location
    unique_stations = df.drop_duplicates(subset=['station_name', 'coordinates_x', 'coordinates_y'])
    
    # Convert longitude and latitude to radians
    earth_radius = 6371 # The radius of the Earth in kilometers
    lat_rad = np.radians(unique_stations['coordinates_y'])
    lon_rad = np.radians(unique_stations['coordinates_x'])
    
    # Create BallTree
    coords_rad = np.column_stack((lat_rad, lon_rad))
    tree = BallTree(coords_rad, metric='haversine')
    
    # Count the number of neighbors within a given radius
    radius_rad = radius_km / earth_radius
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    
    # Create a dictionary mapping density values ​​to each unique station_id
    density_dict = dict(zip(unique_stations['station_name'], counts - 1))
    
    # Map the density values ​​back to the original DataFrame
    return df['station_name'].map(density_dict)

# Use the function to calculate the density
df['station_density_10km'] = calculate_unique_station_density(df, radius_km=10)
df['station_density_1km'] = calculate_unique_station_density(df, radius_km=1)
df['station_density_20km'] = calculate_unique_station_density(df, radius_km=20)

# Print some statistics
print(df[['station_density_1km', 'station_density_10km', 'station_density_20km']].describe())

# df = df.drop(['station_name', 'connector_id', 'postcode','plug_type','connector_type','datetime','weather','connector_unique_id'], axis=1)
df = df.drop(['postcode','plug_type','connector_type','weather'], axis=1)
df['availability_change'] = df['is_available'] - df['availability_24h_ago']

# Ensure 'datetime' columns are of type datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Find the earliest date
earliest_date = df['datetime'].min()

df['relative_days'] = (df['datetime'] - earliest_date).dt.days

# Display some basic statistics for the new column
print(df['relative_days'].describe())

# Check that the new column was added correctly
print(df[['datetime', 'relative_days']].head())
print(df[['datetime', 'relative_days']].tail())

df = df.drop('datetime', axis=1)



import os
from datetime import datetime

# Create output file name
output_filename = "cleaned_charging_station_data.csv"

# Export the data frame to a CSV file
df.to_csv(output_filename, index=False)

# Get the full path of the current directory
current_dir = os.getcwd()

# Build the complete output path
output_path = os.path.join(current_dir, output_filename)

print(f"The data was successfully output to the file:{output_path}")

import pandas as pd
import pymysql



# 1. Read the CSV file, specify the data type and set low_memory=False
dtype_spec = {
    'city_id': 'int64',
    'station_name': 'object',
    'connector_id': 'int64',
    'coordinates_x': 'float64',
    'coordinates_y': 'float64',
    'tariff_amount': 'float64',
    'tariff_connectionfee': 'float64',
    'max_chargerate': 'int64',
    'plug_type_ccs': 'float64',
    'plug_type_chademo': 'float64',
    'plug_type_type_2_plug': 'float64',
    'connector_type_AC': 'float64',
    'connector_type_AC Controller/Receiver': 'float64',
    'connector_type_Rapid': 'float64',
    'connector_type_Ultra-Rapid': 'float64',
    'connector_type_iCharging': 'float64',
    'connector_avg_usage': 'float64',
    'station_avg_usage': 'float64',
    'distance_to_center': 'float64',
    'city_station_density': 'float64',
    'station_connector_count': 'int64',
    'station_avg_max_chargerate': 'float64',
    'station_density_10km': 'int64',
    'station_density_1km': 'int64',
    'station_density_20km': 'int64'
}

df = pd.read_csv('cleaned_charging_station_data.csv', dtype=dtype_spec, low_memory=False)

# 2. Rename the column names to remove special characters
df = df.rename(columns={
    'connector_type_AC Controller/Receiver': 'connector_type_AC_Controller_Receiver'
})

# 3. Remove unnecessary features
columns_to_drop = [
    'is_weekend', 'time_of_day', 'is_holiday', 'is_work_hour',
    'connector_unique_id', 'usage_last_24h', 'usage_last_7d',
    'city_density_level', 'availability_24h_ago',
    'availability_1week_ago', 'availability_change', 'relative_days',
    'is_available'
]
df = df.drop(columns=columns_to_drop)

# 4. List of retained features
features_to_keep = [
    'city_id', 'station_name', 'connector_id', 'coordinates_x', 'coordinates_y',
    'tariff_amount', 'tariff_connectionfee', 'max_chargerate', 'plug_type_ccs',
    'plug_type_chademo', 'plug_type_type_2_plug', 'connector_type_AC',
    'connector_type_AC_Controller_Receiver', 'connector_type_Rapid',
    'connector_type_Ultra-Rapid', 'connector_type_iCharging',
    'connector_avg_usage', 'station_avg_usage', 'distance_to_center',
    'city_station_density', 'station_connector_count', 'station_avg_max_chargerate',
    'station_density_10km', 'station_density_1km', 'station_density_20km'
]

# 5. Aggregate data
grouped_df = df.groupby(['station_name', 'connector_id']).first().reset_index()

# 6. Check the consistency of aggregated data
consistent_df = grouped_df[features_to_keep]

# 7. Create a database connection
connection = pymysql.connect(
    host=db_config['host'],
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database']
)

# 8. Create the table (if it doesn't exist)
create_table_query = """
CREATE TABLE IF NOT EXISTS PredictionInput (
    city_id INT,
    station_name VARCHAR(255) NOT NULL,
    connector_id INT NOT NULL,
    coordinates_x FLOAT,
    coordinates_y FLOAT,
    tariff_amount FLOAT,
    tariff_connectionfee FLOAT,
    max_chargerate INT,
    plug_type_ccs FLOAT,
    plug_type_chademo FLOAT,
    plug_type_type_2_plug FLOAT,
    connector_type_AC FLOAT,
    connector_type_AC_Controller_Receiver FLOAT,
    connector_type_Rapid FLOAT,
    connector_type_Ultra_Rapid FLOAT,
    connector_type_iCharging FLOAT,
    connector_avg_usage FLOAT,
    station_avg_usage FLOAT,
    distance_to_center FLOAT,
    city_station_density FLOAT,
    station_connector_count INT,
    station_avg_max_chargerate FLOAT,
    station_density_10km INT,
    station_density_1km INT,
    station_density_20km INT,
    PRIMARY KEY (station_name, connector_id)
)
"""

with connection.cursor() as cursor:
    cursor.execute(create_table_query)
connection.commit()

# 9. Insert data into the database
insert_query = """
INSERT INTO PredictionInput (
    city_id, station_name, connector_id, coordinates_x, coordinates_y,
    tariff_amount, tariff_connectionfee, max_chargerate, plug_type_ccs,
    plug_type_chademo, plug_type_type_2_plug, connector_type_AC,
    connector_type_AC_Controller_Receiver, connector_type_Rapid,
    connector_type_Ultra_Rapid, connector_type_iCharging,
    connector_avg_usage, station_avg_usage, distance_to_center,
    city_station_density, station_connector_count, station_avg_max_chargerate,
    station_density_10km, station_density_1km, station_density_20km
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
) ON DUPLICATE KEY UPDATE
    coordinates_x = VALUES(coordinates_x),
    coordinates_y = VALUES(coordinates_y),
    tariff_amount = VALUES(tariff_amount),
    tariff_connectionfee = VALUES(tariff_connectionfee),
    max_chargerate = VALUES(max_chargerate),
    plug_type_ccs = VALUES(plug_type_ccs),
    plug_type_chademo = VALUES(plug_type_chademo),
    plug_type_type_2_plug = VALUES(plug_type_type_2_plug),
    connector_type_AC = VALUES(connector_type_AC),
    connector_type_AC_Controller_Receiver = VALUES(connector_type_AC_Controller_Receiver),
    connector_type_Rapid = VALUES(connector_type_Rapid),
    connector_type_Ultra_Rapid = VALUES(connector_type_Ultra_Rapid),
    connector_type_iCharging = VALUES(connector_type_iCharging),
    connector_avg_usage = VALUES(connector_avg_usage),
    station_avg_usage = VALUES(station_avg_usage),
    distance_to_center = VALUES(distance_to_center),
    city_station_density = VALUES(city_station_density),
    station_connector_count = VALUES(station_connector_count),
    station_avg_max_chargerate = VALUES(station_avg_max_chargerate),
    station_density_10km = VALUES(station_density_10km),
    station_density_1km = VALUES(station_density_1km),
    station_density_20km = VALUES(station_density_20km)
"""

with connection.cursor() as cursor:
    for index, row in consistent_df.iterrows():
        cursor.execute(insert_query, (
            row['city_id'], row['station_name'], row['connector_id'], row['coordinates_x'], row['coordinates_y'],
            row['tariff_amount'], row['tariff_connectionfee'], row['max_chargerate'], row['plug_type_ccs'],
            row['plug_type_chademo'], row['plug_type_type_2_plug'], row['connector_type_AC'],
            row['connector_type_AC_Controller_Receiver'], row['connector_type_Rapid'],
            row['connector_type_Ultra-Rapid'], row['connector_type_iCharging'],
            row['connector_avg_usage'], row['station_avg_usage'], row['distance_to_center'],
            row['city_station_density'], row['station_connector_count'], row['station_avg_max_chargerate'],
            row['station_density_10km'], row['station_density_1km'], row['station_density_20km']
        ))
    connection.commit()

# 10. Close the database connection
connection.close()
print(f"The database connection has been closed, and now the machine learning model training begins.")


df = pd.read_csv('cleaned_charging_station_data.csv')
df = df.reset_index(drop=True)
df = df.drop(['is_weekend', 'time_of_day', 'is_holiday','is_work_hour','connector_unique_id','usage_last_24h','usage_last_7d','city_density_level','availability_24h_ago','availability_1week_ago','availability_change','relative_days'], axis=1)

import joblib
from sklearn.preprocessing import LabelEncoder


# Encode station_name
station_encoder = LabelEncoder()
df['station_name_encoded'] = station_encoder.fit_transform(df['station_name'].astype(str))

# Save station_encoder
joblib.dump(station_encoder, 'station_encoder.joblib')

# Encode city_id
city_encoder = LabelEncoder()
df['city_id_encoded'] = city_encoder.fit_transform(df['city_id'].astype(str))

# Save city_encoder
joblib.dump(city_encoder, 'city_encoder.joblib')

# Delete the original column
df = df.drop(['station_name', 'city_id'], axis=1)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder


X = df.drop('is_available', axis=1)
y = df['is_available']


# Create periodic features
X['hour_sin'] = np.sin(X['hour'] * (2 * np.pi / 24))
X['hour_cos'] = np.cos(X['hour'] * (2 * np.pi / 24))
X['day_of_week_sin'] = np.sin(X['day_of_week'] * (2 * np.pi / 7))
X['day_of_week_cos'] = np.cos(X['day_of_week'] * (2 * np.pi / 7))
X = X.drop('hour', axis=1)
X = X.drop('day_of_week', axis=1)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
rf.fit(X, y)

import joblib
import os

# Specify the directory and file name to save the model
model_directory = 'saved_models'
model_filename = 'random_forest_model.joblib'

# If the directory does not exist, create it
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Complete model save path
model_path = os.path.join(model_directory, model_filename)

# Save the random forest model
joblib.dump(rf, model_path)

print(f"Model saved to: {model_path}")


