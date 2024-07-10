from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import urllib.parse

# 数据库连接信息
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'db_evcharger'
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

    # 生成当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建 CSV 文件名
    csv_filename = f'charging_station_data_{timestamp}.csv'

    # 将数据保存为 CSV 文件
    df.to_csv(csv_filename, index=False)

    print(f"数据已保存到 {csv_filename}")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    engine.dispose()
    print("数据库连接已关闭")