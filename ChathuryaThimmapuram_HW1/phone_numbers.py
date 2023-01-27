import re

def valid_phone_number(s):
    pattern = re.compile(r"(?:\+1|001)?[\s-]?[\(]?[0-9]{3}[\)]?[\s-]?[0-9]{3}[\s-]?[0-9]{4}")
    return pattern.search(s)