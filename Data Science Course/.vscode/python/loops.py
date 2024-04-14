s = "Python Programming is an easy language to learn"
print(s[::-1])
print(s.swapcase())

var = input().lower()
str =''
for i in var:
    str = i + str
if var==str:
    print("palindrome")
else:
    print("Not palindrome")    