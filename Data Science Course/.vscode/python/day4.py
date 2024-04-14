'''day-4'''
#guess the output:
#name ="Monty Python",name[::-1]
name = 'Monty Python'
print(name[::-1])
#name[start:end:skip]

lis=['kalyan','non','pon','amrutha']
lis.remove('amrutha')
print(lis)
lis.pop(-2)
print(lis)
del lis[1]
print(lis)

# sort()
fruits = list()

fruits.append("grapes")
fruits.append("apple")
fruits.append("banana")
fruits.sort()
print(fruits)


#print max and min in the list
largest=[2,5,99,23,45,75,87]
print(max(largest))
print(min(largest))

#sum()
print(sum(largest))

#add two lists
lis1=['hello']
lis2=['world']
lis1.extend(lis2)
print(lis1)
# print(lis1+lis2)

'''clear()
count()
copy()'''