import json
import requests
import pymysql
from datetime import datetime
import time


def run_task():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '123456',
        'database': 'db_evcharger_dynamic'
    }

    with open('extracted_data.json') as f:
        data = json.load(f)

    chargepoint_api_key = 'c3VwcG9ydCtjcHNhcHBAdmVyc2FudHVzLmNvLnVrOmt5YlRYJkZPJCEzcVBOJHlhMVgj'

    while True:
        current_time = datetime.now()
        date = current_time.strftime('%Y-%m-%d')
        hour = current_time.hour

        batch_size = 100

        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()

        total_entries = len(data)
        for i in range(0, total_entries, batch_size):
            batch = data[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {total_entries // batch_size + 1}")

            start_time = time.time()

            chargePointIDs = ','.join([entry['name'] for entry in batch])
            cities = {entry['name']: entry['city'] for entry in batch}

            chargepoint_url = 'https://account.chargeplacescotland.org/api/v2/poi/chargepoint/dynamic'
            headers = {
                'api-auth': chargepoint_api_key,
                'chargePointIDs': chargePointIDs
            }
            response = requests.get(chargepoint_url, headers=headers)
            chargepoint_data = response.json()

            for chargePoint in chargepoint_data['chargePoints']:
                chargePoint_info = chargePoint['chargePoint']
                chargePoint_name = chargePoint_info['name']
                city = cities[chargePoint_name]
                for connectorGroup in chargePoint_info['connectorGroups']:
                    for connector in connectorGroup['connectors']:
                        connector_id = connector['connectorID']
                        is_available = connector['connectorStatus'] == "AVAILABLE"
                        cursor.execute(
                            """
                            INSERT INTO availability (station_name, connector_id, city, date, hour, is_available)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                is_available = VALUES(is_available)
                            """,
                            (chargePoint_name, connector_id, city, date, hour, is_available)
                        )

            conn.commit()
            print(f"Batch {i // batch_size + 1} processed and committed.")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"执行时间为：{elapsed_time} 秒")

        cursor.close()
        conn.close()

        # 暂停一小时
        time.sleep(3600 - (time.time() - start_time))


# 启动任务
run_task()
