import pandas as pd
import pymysql
from pymysql import Error

# 数据库配置信息
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'db_evcharger_0707'
}

# 指定数据类型
dtype_spec = {
    'city_id': 'int64',
    'station_name': 'object',
    'connector_id': 'int64',
    'coordinates_x': 'float64',
    'coordinates_y': 'float64',
    'tariff_amount': 'float64',
    'tariff_connectionfee': 'float64',
    'max_chargerate': 'int64',
    'is_available': 'int64',
    'hour': 'int32',
    'day_of_week': 'int32',
    'is_weekend': 'int32',
    'is_holiday': 'int32',
    'time_of_day': 'int64',
    'is_work_hour': 'int32',
    'plug_type_ccs': 'float64',
    'plug_type_chademo': 'float64',
    'plug_type_type_2_plug': 'float64',
    'connector_type_AC': 'float64',
    'connector_type_AC Controller/Receiver': 'float64',
    'connector_type_Rapid': 'float64',
    'connector_type_Ultra-Rapid': 'float64',
    'connector_type_iCharging': 'float64',
    'weather_Clear': 'float64',
    'weather_Clouds': 'float64',
    'weather_Drizzle': 'float64',
    'weather_Fog': 'float64',
    'weather_Haze': 'float64',
    'weather_Mist': 'float64',
    'weather_Rain': 'float64',
    'connector_unique_id': 'object',
    'connector_avg_usage': 'float64',
    'station_avg_usage': 'float64',
    'usage_last_24h': 'float64',
    'usage_last_7d': 'float64',
    'distance_to_center': 'float64',
    'city_station_density': 'float64',
    'city_density_level': 'int64',
    'station_connector_count': 'int64',
    'station_avg_max_chargerate': 'float64',
    'availability_24h_ago': 'float64',
    'availability_1week_ago': 'float64',
    'station_density_10km': 'int64',
    'station_density_1km': 'int64',
    'station_density_20km': 'int64',
    'availability_change': 'float64',
    'relative_days': 'int64'
}

# 读取CSV文件，并指定数据类型
df = pd.read_csv('cleaned_charging_station_data.csv', dtype=dtype_spec)

# 删除不需要的特征
columns_to_drop = [
    'is_weekend', 'time_of_day', 'is_holiday', 'is_work_hour',
    'connector_unique_id', 'usage_last_24h', 'usage_last_7d',
    'city_density_level', 'availability_24h_ago', 'availability_1week_ago',
    'availability_change', 'relative_days', 'is_available'
]
df = df.drop(columns=columns_to_drop)

# 最终保留的特征
columns_to_keep = [
    'station_name', 'connector_id', 'coordinates_x', 'coordinates_y',
    'tariff_amount', 'tariff_connectionfee', 'max_chargerate', 'plug_type_ccs',
    'plug_type_chademo', 'plug_type_type_2_plug', 'connector_type_AC',
    'connector_type_AC Controller/Receiver', 'connector_type_Rapid',
    'connector_type_Ultra-Rapid', 'connector_type_iCharging', 'connector_avg_usage',
    'station_avg_usage', 'distance_to_center', 'city_station_density',
    'station_connector_count', 'station_avg_max_chargerate', 'station_density_10km',
    'station_density_1km', 'station_density_20km'
]

# 只保留所需的特征
df = df[columns_to_keep]

# 按 station_name 和 connector_id 聚合数据，确保其他特征完全相等
df_grouped = df.groupby(['station_name', 'connector_id'], as_index=False).first()

df = df_grouped

# 连接到MySQL数据库
try:
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )

    cursor = connection.cursor()

    # 创建表，如果不存在
    create_table_query = """
    CREATE TABLE IF NOT EXISTS PredictionInput (
        station_name VARCHAR(255),
        connector_id INT,
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
    );
    """
    cursor.execute(create_table_query)

    # 插入数据到表中
    for _, row in df.iterrows():
        insert_query = """
        INSERT INTO PredictionInput (
            station_name, connector_id, coordinates_x, coordinates_y, tariff_amount, 
            tariff_connectionfee, max_chargerate, plug_type_ccs, plug_type_chademo, 
            plug_type_type_2_plug, connector_type_AC, connector_type_AC_Controller_Receiver, 
            connector_type_Rapid, connector_type_Ultra_Rapid, connector_type_iCharging, 
            connector_avg_usage, station_avg_usage, distance_to_center, city_station_density, 
            station_connector_count, station_avg_max_chargerate, station_density_10km, 
            station_density_1km, station_density_20km
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            coordinates_x=VALUES(coordinates_x), coordinates_y=VALUES(coordinates_y),
            tariff_amount=VALUES(tariff_amount), tariff_connectionfee=VALUES(tariff_connectionfee),
            max_chargerate=VALUES(max_chargerate), plug_type_ccs=VALUES(plug_type_ccs),
            plug_type_chademo=VALUES(plug_type_chademo), plug_type_type_2_plug=VALUES(plug_type_type_2_plug),
            connector_type_AC=VALUES(connector_type_AC), connector_type_AC_Controller_Receiver=VALUES(connector_type_AC_Controller_Receiver),
            connector_type_Rapid=VALUES(connector_type_Rapid), connector_type_Ultra_Rapid=VALUES(connector_type_Ultra_Rapid),
            connector_type_iCharging=VALUES(connector_type_iCharging), connector_avg_usage=VALUES(connector_avg_usage),
            station_avg_usage=VALUES(station_avg_usage), distance_to_center=VALUES(distance_to_center),
            city_station_density=VALUES(city_station_density), station_connector_count=VALUES(station_connector_count),
            station_avg_max_chargerate=VALUES(station_avg_max_chargerate), station_density_10km=VALUES(station_density_10km),
            station_density_1km=VALUES(station_density_1km), station_density_20km=VALUES(station_density_20km)
        """
        cursor.execute(insert_query, tuple(row))

    # 提交事务
    connection.commit()

except Error as e:
    print(f"Error: {e}")
finally:
    try:
        cursor.close()
        connection.close()
    except NameError:
        pass  # 如果连接没有建立，不进行关闭操作

print("数据已成功写入数据库中的 PredictionInput 表")
