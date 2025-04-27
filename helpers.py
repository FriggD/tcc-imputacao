import time
import math


def asMinutes(s):
    """
    Converts seconds into a human readable string format of minutes and seconds.
    Args:
        s (float): Time in seconds
    Returns:
        str: Formatted string in the form of 'Xm Ys'
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """
    Calculates the time elapsed since a given time point and estimates time remaining based on progress.
    Args:
        since (float): Start time in seconds
        percent (float): Progress percentage (0-1)
    Returns:
        str: String containing time elapsed and estimated time remaining
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def split_every_n_elements(string, n_elements=1):
    """
    Splits a string into chunks of specified size.
    Args:
        string (str): Input string to be split
        n_elements (int): Size of each chunk (default=1)
    Returns:
        list: List of string chunks of size n_elements
    """
    return [ "".join(string[i:i+n_elements]) for i in range(0,len(string),n_elements)]