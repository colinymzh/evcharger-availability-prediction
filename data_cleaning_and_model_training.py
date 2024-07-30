from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import urllib.parse

# 数据库连接信息
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'db_evcharger_0707'
}

# 创建数据库 URL（对密码进行 URL 编码以处理特殊字符）
db_url = f"mysql+pymysql://{db_config['user']}:{urllib.parse.quote_plus(db_config['password'])}@{db_config['host']}/{db_config['database']}"

# 创建 SQLAlchemy 引擎
engine = create_engine(db_url)

# 更新后的 SQL 查询，包含 weather 信息
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
    # 使用 pandas 读取 SQL 查询结果
    df = pd.read_sql_query(query, engine)

    # 数据处理
    # 将 date 和 hour 合并为一个 datetime 列
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')

    # 删除原始的 date 和 hour 列
    df = df.drop(['date', 'hour'], axis=1)

    # 重新排列列的顺序
    columns_order = [
        'city_id', 'station_name', 'connector_id', 'coordinates_x', 'coordinates_y',
        'postcode', 'tariff_amount', 'tariff_connectionfee', 'max_chargerate',
        'plug_type', 'connector_type', 'datetime', 'weather', 'is_available'
    ]
    df = df[columns_order]


    # 创建 CSV 文件名
    csv_filename = f'charging_station_data.csv'

    # 将数据保存为 CSV 文件
    df.to_csv(csv_filename, index=False)

    print(f"数据已保存到 {csv_filename}")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    engine.dispose()
    print("数据库连接已关闭")


# 导入必要的库
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('charging_station_data.csv')
df = df.reset_index(drop=True)
# 显示数据的基本信息
print(df.info())

# 检查tariff_amount列的空值数量
print("\n空值数量:")
print(df['tariff_amount'].isnull().sum())

# 定义一个函数来填充tariff_amount的空值
def fill_tariff_amount(group):
    mask = group['tariff_amount'].isnull()
    group.loc[mask, 'tariff_amount'] = group['tariff_amount'].mean()
    return group

# 按max_chargerate, plug_type, connector_type分组，并应用填充函数
df = df.groupby(['max_chargerate', 'plug_type', 'connector_type'], group_keys=False).apply(fill_tariff_amount)

# 再次检查tariff_amount列的空值数量
print("\n填充后的空值数量:")
print(df['tariff_amount'].isnull().sum())

# 显示填充后的数据样本
print("\n填充后的数据样本:")
print(df[['max_chargerate', 'plug_type', 'connector_type', 'tariff_amount']].sample(10))

# 检查填充是否正确
print("\n每组的平均tariff_amount:")
print(df.groupby(['max_chargerate', 'plug_type', 'connector_type'])['tariff_amount'].mean())

# 检查 tariff_connectionfee 列的空值数量
print("\ntariff_connectionfee 列的空值数量(填充前):")
print(df['tariff_connectionfee'].isnull().sum())

# 将 tariff_connectionfee 列的空值填充为 0
df['tariff_connectionfee'] = df['tariff_connectionfee'].fillna(0)

# 再次检查 tariff_connectionfee 列的空值数量,确保填充成功
print("\ntariff_connectionfee 列的空值数量(填充后):")
print(df['tariff_connectionfee'].isnull().sum())

# 显示填充后的数据样本
print("\n填充后的数据样本:")
print(df[['station_name', 'tariff_amount', 'tariff_connectionfee']].sample(10))

# 检查 tariff_connectionfee 列的基本统计信息
print("\ntariff_connectionfee 列的基本统计信息:")
print(df['tariff_connectionfee'].describe())

# 显示原始数据的行数
print(f"原始数据行数: {len(df)}")

# 检查每列中 'UNKNOWN' 的数量
columns_to_check = ['max_chargerate', 'plug_type', 'connector_type']
for col in columns_to_check:
    unknown_count = (df[col] == 'UNKNOWN').sum()
    print(f"{col} 中 'UNKNOWN' 的数量: {unknown_count}")

# 删除包含 'UNKNOWN' 的行
df_cleaned = df[~df[columns_to_check].isin(['UNKNOWN']).any(axis=1)]

# 显示清理后的数据行数
print(f"\n清理后的数据行数: {len(df_cleaned)}")

# 再次检查是否还有 'UNKNOWN' 值
for col in columns_to_check:
    unknown_count = (df_cleaned[col] == 'UNKNOWN').sum()
    print(f"{col} 中剩余 'UNKNOWN' 的数量: {unknown_count}")

# 显示清理后的数据样本
print("\n清理后的数据样本:")
print(df_cleaned[columns_to_check + ['tariff_amount']].sample(10))

# 将 max_chargerate 转换为数值类型（如果还不是的话）
df_cleaned['max_chargerate'] = pd.to_numeric(df_cleaned['max_chargerate'], errors='coerce')

# 显示清理后数据的基本统计信息
print("\n清理后数据的基本统计信息:")
print(df_cleaned[columns_to_check + ['tariff_amount']].describe())

# 更新原始 DataFrame
df = df_cleaned.copy()

# 确保 datetime 列是日期时间格式
df['datetime'] = pd.to_datetime(df['datetime'])

# 按 station_name 和 datetime 排序
df = df.sort_values(['station_name', 'datetime'])

# 检查 weather 列的空值数量（处理前）
print("Weather 列空值数量（处理前）:")
print(df['weather'].isnull().sum())

# 定义一个函数来填充 weather 列的空值
def fill_weather(group):
    group['weather'] = group['weather'].fillna(method='ffill')
    group['weather'] = group['weather'].fillna(method='bfill')
    return group

# 对每个充电站分组应用填充函数
df = df.groupby('station_name').apply(fill_weather)

# 重置索引
df = df.reset_index(drop=True)

# 检查 weather 列的空值数量（处理后）
print("\nWeather 列空值数量（处理后）:")
print(df['weather'].isnull().sum())

# 打印索引信息以确认
print("\n索引信息:")
print(df.index)

# 定义一个函数来去除字符串两端的引号
def remove_quotes(text):
    if isinstance(text, str):
        return text.strip("'\"")
    return text

# 应用函数到 weather 列
df['weather'] = df['weather'].apply(remove_quotes)

# 检查处理结果
print("处理后的 weather 列唯一值：")
print(df['weather'].unique())

# 显示天气种类及其频率
print("\n天气种类及其频率：")
print(df['weather'].value_counts())

# 显示包含引号的天气记录（如果还有的话）
quotes = df[df['weather'].str.contains("'|\"", na=False)]
if not quotes.empty:
    print("\n仍然包含引号的记录：")
    print(quotes[['station_name', 'datetime', 'weather']])
else:
    print("\n所有引号已成功移除。")

# 显示处理后的数据样本
print("\n处理后的数据样本：")
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

# 创建 OneHotEncoder 对象
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 对 'plug_type' 进行独热编码
plug_type_encoded = ohe.fit_transform(df[['plug_type']])
plug_type_columns = [f'plug_type_{cat}' for cat in ohe.categories_[0]]
df_plug_type = pd.DataFrame(plug_type_encoded, columns=plug_type_columns, index=df.index)

# 对 'connector_type' 进行独热编码
connector_type_encoded = ohe.fit_transform(df[['connector_type']])
connector_type_columns = [f'connector_type_{cat}' for cat in ohe.categories_[0]]
df_connector_type = pd.DataFrame(connector_type_encoded, columns=connector_type_columns, index=df.index)

# 对 'weather' 进行独热编码
weather_encoded = ohe.fit_transform(df[['weather']])
weather_columns = [f'weather_{cat}' for cat in ohe.categories_[0]]
df_weather = pd.DataFrame(weather_encoded, columns=weather_columns, index=df.index)

# 将编码后的特征合并到原始数据框
df = pd.concat([df, df_plug_type, df_connector_type, df_weather], axis=1)

# 确保 station_name 和 connector_id 都是字符串类型
df['station_name'] = df['station_name'].astype(str)
df['connector_id'] = df['connector_id'].astype(str)

# 使用 str.cat 方法创建 connector_unique_id
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

# 假设 df 是你的主数据框
# 如果 df 还没有被定义，你需要先读取你的数据
# 例如：df = pd.read_csv('your_data.csv')

# 1. 到市中心的距离
city_centers = df.groupby('city_id')[['coordinates_x', 'coordinates_y']].mean()

def distance_to_center(row):
    city_center = city_centers.loc[row['city_id']]
    return euclidean((row['coordinates_x'], row['coordinates_y']), 
                     (city_center['coordinates_x'], city_center['coordinates_y']))

df['distance_to_center'] = df.apply(distance_to_center, axis=1)

# 2. 每个城市的充电站密度
stations_per_city = df.groupby('city_id')['station_name'].nunique()

def safe_city_area(group):
    x_range = group['coordinates_x'].max() - group['coordinates_x'].min()
    y_range = group['coordinates_y'].max() - group['coordinates_y'].min()
    area = x_range * y_range
    return max(area, 1e-10)  # 使用一个很小的正数来替代 0

city_areas = df.groupby('city_id').apply(safe_city_area)
city_density = stations_per_city / city_areas
city_density = city_density.replace([np.inf, -np.inf], np.nan)
city_density = city_density.fillna(city_density.mean())

df['city_station_density'] = df['city_id'].map(city_density)

# 3. 使用分位数划分密度等级
def density_to_level_quantile(density, q_dict):
    for level, threshold in sorted(q_dict.items(), key=lambda x: x[1], reverse=True):
        if density >= threshold:
            return level
    return 1  # 如果密度小于所有阈值，返回最低等级

quantiles = city_density.quantile([0.2, 0.4, 0.6, 0.8])
q_dict = {5: quantiles[0.8], 4: quantiles[0.6], 3: quantiles[0.4], 2: quantiles[0.2], 1: 0}

df['city_density_level'] = df['city_id'].map(city_density).apply(lambda x: density_to_level_quantile(x, q_dict))

# 4. 查看结果
print("新特征的前几行：")
print(df[['city_id', 'station_name', 'distance_to_center', 'city_station_density', 'city_density_level']].head(10))

print("\n新特征的统计信息：")
print(df[['distance_to_center', 'city_station_density']].describe())

print("\n密度等级的分布：")
print(df['city_density_level'].value_counts().sort_index())

print("\n每个密度等级的密度范围：")
for level, threshold in sorted(q_dict.items()):
    if level == 5:
        print(f"等级 {level}: >= {threshold:.2f}")
    elif level == 1:
        print(f"等级 {level}: < {q_dict[2]:.2f}")
    else:
        print(f"等级 {level}: {threshold:.2f} - {q_dict[level+1]:.2f}")

# 5. 检查空值
print("\n检查新特征的空值：")
print(df[['distance_to_center', 'city_station_density', 'city_density_level']].isnull().sum())

# 1. 计算每个充电站的连接器数量
connectors_per_station = df.groupby('station_name')['connector_id'].nunique()
df['station_connector_count'] = df['station_name'].map(connectors_per_station)

# 2. 计算每个充电站的平均最大充电率
avg_max_chargerate = df.groupby('station_name')['max_chargerate'].mean()
df['station_avg_max_chargerate'] = df['station_name'].map(avg_max_chargerate)

# 3. 查看结果
print("新特征的前几行：")
print(df[['station_name', 'station_connector_count', 'station_avg_max_chargerate']].head(10))

print("\n新特征的统计信息：")
print(df[['station_connector_count', 'station_avg_max_chargerate']].describe())

# 4. 检查空值
print("\n检查新特征的空值：")
print(df[['station_connector_count', 'station_avg_max_chargerate']].isnull().sum())
import pandas as pd
import numpy as np

# 假设 df 是你的主数据框
# 如果 df 还没有被定义，你需要先读取你的数据
# 例如：df = pd.read_csv('your_data.csv', parse_dates=['datetime'])

# 确保 datetime 列是 datetime 类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 按充电站和时间排序
df = df.sort_values(['connector_unique_id', 'datetime'])

# 1. 计算前一天同一时间的可用性
df['availability_24h_ago'] = df.groupby('connector_unique_id')['is_available'].shift(24)

# 2. 计算前一周同一时间的可用性
df['availability_1week_ago'] = df.groupby('connector_unique_id')['is_available'].shift(24 * 7)

# 3. 用当前的 is_available 值填充空值
df['availability_24h_ago'] = df['availability_24h_ago'].fillna(df['is_available'])
df['availability_1week_ago'] = df['availability_1week_ago'].fillna(df['is_available'])

# 4. 查看结果
print("新特征的前几行：")
print(df[['connector_unique_id', 'datetime', 'is_available', 'availability_24h_ago', 'availability_1week_ago']].head(20))

print("\n新特征的统计信息：")
print(df[['is_available', 'availability_24h_ago', 'availability_1week_ago']].describe())

# 5. 检查空值（应该都是0了）
print("\n检查新特征的空值：")
print(df[['availability_24h_ago', 'availability_1week_ago']].isnull().sum())

# 6. 可选：检查填充后的值是否与 is_available 相同的比例
print("\n24小时前可用性与当前可用性相同的比例：")
print((df['availability_24h_ago'] == df['is_available']).mean())

print("\n一周前可用性与当前可用性相同的比例：")
print((df['availability_1week_ago'] == df['is_available']).mean())
import numpy as np
from sklearn.neighbors import BallTree

def calculate_unique_station_density(df, radius_km=5):
    # 首先，我们需要获取唯一的充电站位置
    unique_stations = df.drop_duplicates(subset=['station_name', 'coordinates_x', 'coordinates_y'])
    
    # 将经纬度转换为弧度
    earth_radius = 6371  # 地球半径，单位为公里
    lat_rad = np.radians(unique_stations['coordinates_y'])
    lon_rad = np.radians(unique_stations['coordinates_x'])
    
    # 创建BallTree
    coords_rad = np.column_stack((lat_rad, lon_rad))
    tree = BallTree(coords_rad, metric='haversine')
    
    # 计算给定半径内的邻居数量
    radius_rad = radius_km / earth_radius
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    
    # 创建一个字典，将密度值映射到每个唯一的station_id
    density_dict = dict(zip(unique_stations['station_name'], counts - 1))
    
    # 将密度值映射回原始DataFrame
    return df['station_name'].map(density_dict)

# 使用函数计算密度
df['station_density_10km'] = calculate_unique_station_density(df, radius_km=10)
df['station_density_1km'] = calculate_unique_station_density(df, radius_km=1)
df['station_density_20km'] = calculate_unique_station_density(df, radius_km=20)

# 打印一些统计信息
print(df[['station_density_1km', 'station_density_10km', 'station_density_20km']].describe())

# df = df.drop(['station_name', 'connector_id', 'postcode','plug_type','connector_type','datetime','weather','connector_unique_id'], axis=1)
df = df.drop(['postcode','plug_type','connector_type','weather'], axis=1)
df['availability_change'] = df['is_available'] - df['availability_24h_ago']

# 确保 'datetime' 列是 datetime 类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 找到最早的日期
earliest_date = df['datetime'].min()

df['relative_days'] = (df['datetime'] - earliest_date).dt.days

# 显示新列的一些基本统计信息
print(df['relative_days'].describe())

# 检查新列是否已正确添加
print(df[['datetime', 'relative_days']].head())
print(df[['datetime', 'relative_days']].tail())

df = df.drop('datetime', axis=1)



import os
from datetime import datetime

# 创建输出文件名
output_filename = "cleaned_charging_station_data.csv"

# 将数据框输出到CSV文件
df.to_csv(output_filename, index=False)

# 获取当前目录的完整路径
current_dir = os.getcwd()

# 构建完整的输出路径
output_path = os.path.join(current_dir, output_filename)

print(f"数据已成功输出到文件：{output_path}")

import pandas as pd
import pymysql



# 1. 读取CSV文件，指定数据类型并设置 low_memory=False
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

# 2. 重命名列名以去除特殊字符
df = df.rename(columns={
    'connector_type_AC Controller/Receiver': 'connector_type_AC_Controller_Receiver'
})

# 3. 删除不需要的特征
columns_to_drop = [
    'is_weekend', 'time_of_day', 'is_holiday', 'is_work_hour',
    'connector_unique_id', 'usage_last_24h', 'usage_last_7d',
    'city_density_level', 'availability_24h_ago',
    'availability_1week_ago', 'availability_change', 'relative_days',
    'is_available'
]
df = df.drop(columns=columns_to_drop)

# 4. 保留的特征列表
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

# 5. 聚合数据
grouped_df = df.groupby(['station_name', 'connector_id']).first().reset_index()

# 6. 检查聚合后数据的一致性
consistent_df = grouped_df[features_to_keep]

# 7. 创建数据库连接
connection = pymysql.connect(
    host=db_config['host'],
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database']
)

# 8. 创建表（如果不存在）
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

# 9. 将数据插入数据库
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

# 10. 关闭数据库连接
connection.close()

df = pd.read_csv('cleaned_charging_station_data.csv')
df = df.reset_index(drop=True)
df = df.drop(['is_weekend', 'time_of_day', 'is_holiday','is_work_hour','connector_unique_id','usage_last_24h','usage_last_7d','city_density_level','availability_24h_ago','availability_1week_ago','availability_change','relative_days'], axis=1)

import joblib
from sklearn.preprocessing import LabelEncoder

# 假设您的数据框架名为df

# 对station_name进行编码
station_encoder = LabelEncoder()
df['station_name_encoded'] = station_encoder.fit_transform(df['station_name'].astype(str))

# 保存station_encoder
joblib.dump(station_encoder, 'station_encoder.joblib')

# 对city_id进行编码
city_encoder = LabelEncoder()
df['city_id_encoded'] = city_encoder.fit_transform(df['city_id'].astype(str))

# 保存city_encoder
joblib.dump(city_encoder, 'city_encoder.joblib')

# 删除原始列
df = df.drop(['station_name', 'city_id'], axis=1)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 假设你的数据框名为 df
X = df.drop('is_available', axis=1)
y = df['is_available']


# 创建周期性特征
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

# 假设 X 和 y 已经准备好

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 训练模型
rf.fit(X, y)

import joblib
import os

# 指定保存模型的目录和文件名
model_directory = 'saved_models'
model_filename = 'random_forest_model.joblib'

# 如果目录不存在，创建它
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# 完整的模型保存路径
model_path = os.path.join(model_directory, model_filename)

# 保存随机森林模型
joblib.dump(rf, model_path)

print(f"模型已保存到: {model_path}")


