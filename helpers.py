import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def split_every_n_elements(string, n_elements=1):
    return [ "".join(string[i:i+n_elements]) for i in range(0,len(string),n_elements)]