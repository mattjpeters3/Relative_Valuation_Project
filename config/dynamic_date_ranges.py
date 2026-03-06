from datetime import datetime

# Full dataset range (could later be made dynamic)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Today's data
CURRENT = datetime.today().strftime('%Y-%m-%d')


# Optional: Add validation function to ensure dates are in proper format (for future enhancements)
def validate_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False
