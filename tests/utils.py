from datetime import datetime, timezone

from datetimerange import DateTimeRange


def date_range(month: int, start_day: int, end_day: int, start_hr: int = 0, end_hr: int = 0):
    return DateTimeRange(
        datetime(2021, month, start_day, start_hr).replace(tzinfo=timezone.utc),
        datetime(2021, month, end_day, end_hr).replace(tzinfo=timezone.utc),
    )
