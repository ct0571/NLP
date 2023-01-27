import re

def valid_html_tag(string):
    pattern = re.compile(r"<.*?>|</.*?>")
    return pattern.search(string)