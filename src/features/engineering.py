from datetime import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_season(date):
    if ((date.month == 12 and date.day >= 21) or
        (date.month in [1, 2]) or
        (date.month == 3 and date.day < 21)):
        return 'winter'
    # You can add other seasons similarly:
    elif (date.month == 3 and date.day >= 21) or (date.month == 4) or (date.month == 5) or (date.month == 6 and date.day < 21):
        return 'spring'
    elif (date.month == 6 and date.day >= 21) or (date.month in [7, 8]) or (date.month == 9 and date.day < 23):
        return 'summer'
    elif (date.month == 9 and date.day >= 23) or (date.month in [10, 11]) or (date.month == 12 and date.day < 21):
        return 'autumn'
    else:
        return 'unknown'


pd.set_option('display.max_columns', None)  # Don't truncate column content
pd.set_option('display.width', 0)

savannah = pd.read_csv(PROJECT_ROOT / "data/savanna_preserve/1_X_train.csv")
savannah['date'] = pd.to_datetime(savannah['date'])
savannah['month'] = savannah['date'].dt.month
savannah['season'] = savannah['date'].apply(get_season)
savannah['average_humidity'] = savannah[
    [f'relative_humidity_2m_previous_day{i}' for i in range(1, 8)]
].mean(axis=1).round(2)
savannah['average_temp'] = savannah[
    [f'temperature_2m_previous_day{i}' for i in range(1, 8)]
].mean(axis=1).round(2)

print(savannah.head())
savannah.to_csv(PROJECT_ROOT / "data/savanna_preserve/1_X_train.csv", index=False)

urban = pd.read_csv(PROJECT_ROOT / "data/clean_urban_air/2_X_test.csv")
urban['date'] = pd.to_datetime(urban['date'])
urban['month'] = urban['date'].dt.month
urban['weekday'] = urban['date'].dt.weekday
urban['hour'] = urban['date'].dt.hour
urban['average_humidity'] = urban[
    [f'relative_humidity_2m_previous_day{i}' for i in range(1, 8)]
].mean(axis=1).round(2)
urban.to_csv(PROJECT_ROOT / "data/clean_urban_air/2_X_train.csv", index=False)
print(urban.head())

field = pd.read_csv(PROJECT_ROOT / "data/resilient_fields/3_X_train.csv")
field['date'] = pd.to_datetime(field['date'])
field['month'] = field['date'].dt.month
field['season'] = field['date'].apply(get_season)
field['average_precipitation'] = field[
    [f'precipitation_previous_day{i}' for i in range(1, 8)]
].mean(axis=1).round(2)
field.to_csv(PROJECT_ROOT / "data/resilient_fields/3_X_train.csv", index=False)
print(field.head())

#visualize the data
print(tabulate(savannah.head(), headers='keys', tablefmt='psql'))
