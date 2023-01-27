import re
from currency import valid_currency
from date import valid_date
from phone_numbers import valid_phone_number
from tags import valid_html_tag

with open("input.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        
        print("Currency:", valid_currency(line))
        print("Date:", valid_date(line))
        print("Phone Number:", valid_phone_number(line))
        print("HTML Tag:", valid_html_tag(line))
        print("Input:", line)
        
