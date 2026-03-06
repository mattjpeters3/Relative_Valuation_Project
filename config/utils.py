from datetime import datetime, timedelta

def get_first_trading_day(year, month):
    date = datetime(year, month, 1)
    while date.weekday() >= 5:
        date += timedelta(days=1)
    return date

def get_trading_days_in_range(start_date, end_date):
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday–Friday
            days.append(current)
        current += timedelta(days=1)
    return days