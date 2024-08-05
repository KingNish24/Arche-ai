import datetime
def get_current_time(bug):
    print(bug)
    """A simple tool to get the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")
