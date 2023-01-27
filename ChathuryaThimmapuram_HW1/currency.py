import re

def valid_currency(s):
    pattern = re.compile(r"\$[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?(?:M|B|K)?|[-+]\$[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{2})?|[A-Z]{2,3}[0-9]+(?:M|B)")
    return pattern.search(s)