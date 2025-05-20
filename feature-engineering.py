from datetime import datetime

import pandas as pd

pd.set_option('display.max_columns', None)  # Don't truncate column content
pd.set_option('display.width', 0)

savannah = pd.read_csv("1 - Savanna Preserve/1_X_test.csv")
savannah['date'] = pd.to_datetime(savannah['date'])
savannah['month'] = savannah['date'].dt.month

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

savannah['season'] = savannah['date'].apply(get_season)
#savannah['season'] = savannah['date'].dt.weekday

print(savannah.head())
savannah.to_csv("1 - Savanna Preserve/1_X_test.csv", index=False)

