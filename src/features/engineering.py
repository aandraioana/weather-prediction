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
    elif (date.month == 3 and date.day >= 21) or (date.month == 4) or (date.month == 5) or (date.month == 6 and date.day < 21):
        return 'spring'
    elif (date.month == 6 and date.day >= 21) or (date.month in [7, 8]) or (date.month == 9 and date.day < 23):
        return 'summer'
    elif (date.month == 9 and date.day >= 23) or (date.month in [10, 11]) or (date.month == 12 and date.day < 21):
        return 'autumn'
    else:
        return 'unknown'


def process_dataset(folder, name, target_cols, has_temp=False, has_precip=False):
    """Process a dataset and add engineered features."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print('='*60)

    for split in ['train', 'test']:
        filepath = PROJECT_ROOT / f"data/{folder}/{split}.csv"
        df = pd.read_csv(filepath)

        # Parse date
        df['date'] = pd.to_datetime(df['date'])

        # Add time features
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'season' not in df.columns:
            df['season'] = df['date'].apply(get_season)

        # Add average humidity if humidity columns exist
        humidity_cols = [f'relative_humidity_2m_previous_day{i}' for i in range(1, 8)]
        if all(col in df.columns for col in humidity_cols):
            if 'average_humidity' not in df.columns:
                df['average_humidity'] = df[humidity_cols].mean(axis=1).round(2)

        # Add average temperature if temperature columns exist
        temp_cols = [f'temperature_2m_previous_day{i}' for i in range(1, 8)]
        if all(col in df.columns for col in temp_cols):
            if 'average_temp' not in df.columns:
                df['average_temp'] = df[temp_cols].mean(axis=1).round(2)

        # Add average precipitation if precipitation columns exist
        precip_cols = [f'precipitation_previous_day{i}' for i in range(1, 8)]
        if all(col in df.columns for col in precip_cols):
            if 'average_precipitation' not in df.columns:
                df['average_precipitation'] = df[precip_cols].mean(axis=1).round(2)

        # Save back
        df.to_csv(filepath, index=False)
        print(f"  {split}.csv: {len(df)} rows, {len(df.columns)} columns")

    # Show sample
    train_df = pd.read_csv(PROJECT_ROOT / f"data/{folder}/train.csv")
    print(f"\nSample from {name}:")
    print(tabulate(train_df.head(3), headers='keys', tablefmt='psql', showindex=False))


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)

    # Process Savanna Preserve
    process_dataset(
        folder='savanna_preserve',
        name='Savanna Preserve',
        target_cols=['temperature_2m'],
        has_temp=True
    )

    # Process Clean Urban Air
    process_dataset(
        folder='clean_urban_air',
        name='Clean Urban Air',
        target_cols=['us_aqi']
    )

    # Process Resilient Fields
    process_dataset(
        folder='resilient_fields',
        name='Resilient Fields',
        target_cols=['global_tilted_irradiance', 'precipitation'],
        has_precip=True
    )

    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()
