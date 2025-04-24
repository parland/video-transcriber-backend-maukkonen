from datetime import timedelta, datetime

def parse_time(time_str):
    """Convert a string like '0:00:03.500000' into a timedelta object."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        # In case the microseconds are missing
        t = datetime.strptime(time_str, "%H:%M:%S")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
