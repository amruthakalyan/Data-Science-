x=1
y=2
print('Before swapping')
print('x=',x)
print('y=',y)
x,y = y,x  #swapping shortcut
print('After swapping')
print('x=',x)
print('y=',y)

'''String Methods'''
name= 'hey my name is kalyan'
newName=name.capitalize()
print(newName)
print(len(newName))
print(name.endswith('n'))#endswith() returns true or false
print(name.center(28,'*'))
# center(len,char)
print(name.upper())
print(name.lower())
address = 'hyderabad telangana india.'
print(address.replace("a","b",10))
print(address[0:8])