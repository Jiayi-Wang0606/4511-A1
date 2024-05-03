import sys
loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/' ## change to your own root directory
sys.path.append(loading_dir)

import run_model_daily
import update_data_hourly
import schedule
import time

def hourly_task():
  print('updating data...')
  update_data_hourly.update_data(loading_dir=loading_dir)


def daily_task():
  print('updating model...')
  run_model_daily.update_model(loading_dir=loading_dir)
#   run_model_daily.merge_analysis(loading_dir=loading_dir)

schedule.every().hour.do(hourly_task)
schedule.every().day.at("00:00").do(daily_task)

while True:
    schedule.run_pending()
    time.sleep(600)