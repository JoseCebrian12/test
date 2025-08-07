from datetime import datetime, timedelta

# UTC and RTC times from your output
utc_time_str = "2024-10-04 16:30:35"
rtc_time_str = "2024-10-04 16:32:50"

# Convert strings to datetime objects
utc_time = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S")
rtc_time = datetime.strptime(rtc_time_str, "%Y-%m-%d %H:%M:%S")

# Calculate the offset
offset = rtc_time - utc_time

# Print the offset in seconds and the human-readable format
print(f"Offset in seconds: {offset.total_seconds()}")
print(f"Offset: {offset}")
