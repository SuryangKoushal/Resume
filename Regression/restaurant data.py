import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

def generate_restaurant_sales_data(n_days=100):
    # Initialize start date
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_days)]
    
    # Base data
    data = {
        
        'day_of_week': [d.weekday() for d in dates],
        'is_holiday': [random.random() < 0.1 for _ in range(n_days)],  # 10% are holidays
        
        # Weather features (temperature in Celsius and precipitation in mm)
        'temperature': np.random.normal(20, 8, n_days),  # Mean temp 20°C
        'precipitation': np.random.exponential(2, n_days),  # Rain follows exponential distribution
        
        # Restaurant specific features
        'price_point': np.random.uniform(15, 45, n_days),  # Average meal price
        'neighborhood_income': np.random.normal(60000, 15000, n_days),  # Yearly income
        
        # Events and seasonality
        'local_event_size': np.random.choice([0, 100, 500, 1000, 5000], n_days, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        'season': [d.month % 12 // 3 for d in dates],  # 0=winter, 1=spring, 2=summer, 3=fall
        
        # Operational features
        'staff_count': np.random.randint(5, 15, n_days),
        'marketing_spend': np.random.uniform(100, 500, n_days)
    }
    
    df = pd.DataFrame(data)
    
    # Generate daily revenue with non-linear relationships
    base_revenue = 3000
    
    # Weather impact (moderate temperatures are best)
    temp_impact = -0.5 * ((df['temperature'] - 22) ** 2)  # Optimal at 22°C
    
    # Day of week impact
    weekday_impact = np.where(df['day_of_week'].isin([4, 5]), 500, 0)  # Weekend boost
    
    # Holiday boost with day interaction
    holiday_impact = df['is_holiday'] * (300 + weekday_impact)
    
    # Price point × neighborhood income interaction
    price_income_impact = 0.1 * np.sqrt(df['neighborhood_income'] * df['price_point'])
    
    # Event impact varies by season
    event_impact = df['local_event_size'] * (0.5 + 0.2 * df['season'])
    
    # Staff efficiency non-linear impact
    staff_impact = 100 * np.log(df['staff_count'])
    
    # Marketing with diminishing returns
    marketing_impact = 200 * np.log(df['marketing_spend'] / 100)
    
    # Calculate final revenue
    df['daily_revenue'] = (
        base_revenue +
        temp_impact +
        weekday_impact +
        holiday_impact +
        price_income_impact +
        event_impact +
        staff_impact +
        marketing_impact
    )
    
    # Add some random noise
    df['daily_revenue'] += np.random.normal(0, 200, n_days)
    
    # Ensure revenue is positive and round to 2 decimal places
    df['daily_revenue'] = np.round(np.maximum(df['daily_revenue'], 500), 2)
    
    return df

# Generate the data
restaurant_data = generate_restaurant_sales_data(100)

# Save to CSV
restaurant_data.to_csv('restaurant_sales.csv', index=False)

# Display first few rows and basic statistics
print("\nFirst few rows:")
print(restaurant_data.head())
print("\nBasic statistics:")
print(restaurant_data.describe())

# Display correlations with daily_revenue
print("\nCorrelations with daily_revenue:")
print(restaurant_data.corr()['daily_revenue'].sort_values(ascending=False))