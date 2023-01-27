import re

def valid_date(s):
    pattern = re.compile(r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[0-9]{1,2},\s[0-9]{4}|[0-9]{4}[-][0-9]{1,2}[-][0-9]{2}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}|[0-9]{1,2}[/-][A-Za-z]{3},\s[0-9]{4})")
    return pattern.search(s)