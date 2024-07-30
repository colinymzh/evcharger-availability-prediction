import json
import requests
import pymysql
from datetime import datetime
import time

def hourly_task():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '123456',
        'database': 'db_evcharger_dynamic'
    }

    with open('extracted_data.json') as f:
        data = json.load(f)

    weather_api_key = '0eb291587bd8e82f8b1303770c4ba16d'
    last_successful_weather = {}

    while True:
        current_time = datetime.now()
        date = current_time.strftime('%Y-%m-%d')
        hour = current_time.hour

        batch_size = 100
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        processed_cities = set()

        total_entries = len(data)
        for i in range(0, total_entries, batch_size):
            batch = data[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {total_entries // batch_size + 1}")

            start_time = time.time()

            for entry in batch:
                coordinates = entry['coordinates']
                city = entry['city']

                if city not in processed_cities:
                    try:
                        weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={coordinates[0]}&lon={coordinates[1]}&appid={weather_api_key}'
                        weather_response = requests.get(weather_url)
                        weather_response.raise_for_status()  # Raises stored HTTPError, if one occurred.
                        weather_data = weather_response.json()
                        weather_main = weather_data['weather'][0]['main']
                        print(weather_main)

                        # Update last successful weather data
                        last_successful_weather[city] = json.dumps(weather_main)
                    except requests.exceptions.RequestException as e:
                        print(f"Error fetching weather data for {city}: {e}")
                        weather_main = last_successful_weather.get(city, "{}")

                    cursor.execute(
                        """
                        INSERT INTO geography (date, hour, city, weather)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            weather = VALUES(weather)
                        """,
                        (date, hour, city, weather_main)
                    )
                    processed_cities.add(city)

            conn.commit()
            print(f"Batch {i // batch_size + 1} processed and committed.")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"执行时间为：{elapsed_time} 秒")

        cursor.close()
        conn.close()

        # 暂停直到下一个小时
        time_to_wait = 3600 - (time.time() - start_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

# 启动任务
hourly_task()
