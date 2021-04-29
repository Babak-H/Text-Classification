import re

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer 
Mr Smith 
Ms Davis 
Mrs. Robinson  
Mr. T
'''

# normally python changes text with \n or \t but when you put r'' it will use raw text and ignore
# any special functionality of thos characters

print(r'\tTab')

pattern = re.compile(r'abc')
matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)

# span=(1,4) start and end of the founded match
print(text_to_search[1:4])

# metaCharacters (need to be escaped): . ^ $ * + ? { } [ ] \ | ( )

pattern = re.compile(r'coreyms\.com')
matches = pattern.finditer(text_to_search)

for match in matches:
    print(match)

'''
. - Any Character Except New Line 
\d - Digit (0-9) 
\D - Not a Digit (0-9) 
\w - Word Character (a-z, A-Z, 0-9, _) 
\W - Not a Word Character 
\s - Whitespace (space, tab, newline) 
\S - Not Whitespace (space, tab, newline)
\b - Word Boundary
\B - Not a Word Boundary
^ - Beginning of a String
$ - End of a String 
[] - Matches Characters in brackets, but only one of them
[^ ] - Matches Characters NOT in brackets 
[A-Z] [a-z] [0-9] [A-Za-z0-9] this is range 
| - Either Or
( ) - Group
Quantifiers
*       - 0 or more
+       - 1 or More
?       - 0 or One
{3}     - Exact Number
{3,4}   - Range of Numbers (Minimum, Maximum)
?=.*    - this is positive look ahead, to find at least a character that comes next 
Sample Regex (for email recognition)
[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+
'''

pattern = re.compile(r'.')  # matches everything except a new line
matches = pattern.finditer("hfgffg")
for match in matches:  # will have 6 matches
    print(match)

pattern = re.compile(r'\d')
matches = pattern.finditer("123ghfgh587cvdg")
for match in matches:  # matches all the digits in the string
    print(match)

pattern = re.compile(r'\D')
matches = pattern.finditer("123ghfgh587cvdg")
for match in matches:  # matches all the non-digits in the string
    print(match)

pattern = re.compile(r'\w')
matches = pattern.finditer("Hello World****")
for match in matches:  # match all number digits and characters
    print(match)

pattern = re.compile(r'\W')
matches = pattern.finditer("Hello World****!!?")
# matches only special characters (all except digits and alphabet)
for match in matches:
    print(match)

pattern = re.compile(r'\s')
matches = pattern.finditer("Hello World** **")
for match in matches:  # finds all tabs, white spaces
    print(match)

sent = "Start a sentence and then bring it to an end"
# find an string that starts with 'Start'
resultList = re.findall(pattern=r'^Start', string=sent)
print(resultList)

# find an string that ends with 'end'
print(re.findall(pattern=r'end$', string=sent))

# 3digits 3digits 4digits ,  the '.' can be any character except new line
pattern = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
print(re.findall(pattern, text_to_search))

'''
with open('Files/data.txt', 'r', encoding='utf-8') as f:
    contents = f.read()
pattern = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
re.findall(pattern, contents)[0:5]  # only show the first 5 results
'''

# [.-] : match only . or - but only ONE of them for a match
pattern = re.compile(r'\d\d\d[-.]\d\d\d[-.]\d\d\d\d')
print(re.findall(pattern, text_to_search))

# 8 or 9 00 -or. 3digits -or. 4digits
pattern = re.compile(r'[89]00[-.]\d\d\d[-.]\d\d\d\d')
print(re.findall(pattern, text_to_search))

# find anything except upper and lower case letters, number digits, and it can only be one character
pattern = re.compile(r'[^A-Za-z0-9]')
print(re.findall(pattern, text_to_search))

text = 'cat mat pat bat'
pattern = re.compile(r'[^b]at')   # found character shouldn't be b
print(re.findall(pattern, text))

# \d\d\d = \d{3}
# 3digits | 3digits | can be 3 to 5 digits
pattern = re.compile(r'\d{3}[-.]\d{3}[-.]\d{3,5}')
print(re.findall(pattern, text_to_search))

# Mr | a . or nothing | white space | an upperCase character | 0 or more word characters
pattern = re.compile(r'Mr\.?\s[A-Z]\w*')
print(re.findall(pattern, text_to_search))

# Email Regex Pattern
emails = '''
babak.awesome@gmail.com 
babak.kool@edu.lodz.pl
corey-shefer-123@my-work.net
'''
# + means one or more characters
pattern = re.compile(r'[a-zA-Z0-9.-]+@[a-zA-Z.-]+\.[a-z]{3}')
print(re.findall(pattern, emails))

web_address = '''
https://www.google.com
http://coreyms.com
https://www.nasa.gov
'''
# https? : s? means s is optional  | (www\.)?  : this whole group is optional
# when you have groups () it will put all groups together in a tuple
pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
print(re.findall(pattern, web_address))

text = "apple pie is good"
# sub() fuction finds the given pattern and replaces it with what user wants
re.sub('apple', 'orange', text)

# given a number : from 1 to that number print each number in natural, octagon, hexagon, binary


def print_formatted(number):
    for i in range(1, number+1):
        string = ""
        string = re.sub('[^0-9]', '', str(i)) + "  " + re.sub('[^0-9]', '', str(oct(i))).lstrip("0") + "  " + re.sub(
            '[^0-9]', '', str(hex(i))).lstrip("0") + "  " + re.sub('[^0-9]', '', str(bin(2))).lstrip("0")
        print(string)


print_formatted(10)

# regex split
# re.split(pattern, string) : split string by the occurances of pattern
regex_pattern = r'\W+'  # "\W+" takes one or more characters of non-alphabetical
print("--".join(re.split(regex_pattern, "Hello My Friend @ babak")))

# find all occasions that a character is repeated more than once
# re.match()
# re.findall() : finds all occurrence of the pattern
# r'([a-zA-Z0-9])\1+'  : this means when two or more same alphanumeric characters occur with each other
m = re.match(r'(\w+)@(\w+)\.(\w+)', 'username@gmail.com')
m.group(0)  # username@gmail.com
m.group(1)  # username
m.group(2)  # gmail
m.group(3)  # com
m.groups()  # username gmail com

string = "ohh mmm hmm"
m = re.findall(r'([a-zA-Z0-9])\1+', string)
if len(m) > 0:
    print(len(m))
else:
    print("-1")


# given a text that starts with non "aeiou" (vowls) then there is "aeiou" then non "aeiou" if any of characters of "aeiou" are repeated
# find them
s = '[qwrtypsdfghjklzxcvbnm]'  # silent characters
# ?<= : negative look ahead
m = re.findall('(?<=' + s + ')([aeiou]{2,})' + s, input(), re.I)
# re.I : ignore case sensitive (I and i are both same)
if(len(m) > 0):
    for match in m:
        print(match)
else:
    print("-1")

# we want to find repetition of the phrase "But, um" and "but, um" in a text:
text = "But, um  and something but, um"
print(re.findall('[B,b]ut, um', text))

# Regex password checking
# make sure that the string contains at least one lower case char, upper case char, digit and symbol
pattern = r'(?=.*\d)(?=.*[a-z])(?=.*[A-Z](?=.*\W))'
'''
(?=.*[a-z]) => use positive look ahead to see if at least one lower case letter exists
(?=.*[A-Z]) => use positive look ahead to see if at least one upper case letter exists
(?=.*\d) => use positive look ahead to see if at least one digit exists
(?=.*\W]) => use positive look ahead to see if at least one non-word character exists
'''

# How to split a string by non alpha characters (we can just use .isalpha() instead of regex too)


def split_non_alpha(s):
    pos = 1
    # here we start pos with 1 because we want to split the sting (not to split by first character)
    while pos < len(s) and s[pos].isalpha():
        pos += 1
    return (s[:pos], s[pos], s[pos+1:])


print(split_non_alpha("#include'blah.php"))
