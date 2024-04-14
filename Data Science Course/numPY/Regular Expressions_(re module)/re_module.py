import re
stri = '''I was occupied on sunday and after i've got free,i called my friend and his number was 111-9876543210  and his friend number is +91-9160911716'''

#\d-> single digit
number = re.findall('\d',stri)
print(number)

#\d+-> multiple digit
number = re.findall('\d+',stri)
print(number)


#Pattern
number = re.findall('\d{3}-\d{10}',stri)
print(number)

#search()  ^ ->used to search in string
stri = '''On monday ,their is a chances of heavy rains'''
strin = re.search('^On',stri)
if strin:
    print('matched')
else:
    print('not matched')  


#match

stri = 'Dog is a pet animal'
mat = re.match('D\w\w',stri)
print(mat)
print(mat.group())


#split()

stri = 'Dog is a pet animal'
spli = re.split('\s',stri)
print(spli)
'''
\d -> single digit
\w -> single character A-Z/a-z/0-9
\s -> space

'''

#sub()
#sub(pattern,replaced pattern,string)
stri = 'Football is not a ball of foot'
rep = re.sub('o','e',stri)
print(rep)

rep = re.sub('o','e',stri,2)
print(rep)



