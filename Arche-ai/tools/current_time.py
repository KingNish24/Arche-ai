
def get_current_time():
    """A simple tool to get the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")