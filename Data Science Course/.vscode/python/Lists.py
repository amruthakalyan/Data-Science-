#Day-3
# List is a python dataStructure similar to arrays except that list accepts different data types(heterogenius).duplication of data. CRUD-->operations(change,read,update,delete)

vegetables =['potatoes','ladyfinger','tomatoes','brinjal']
print(type(vegetables))
print(vegetables[-1])
for i in vegetables:
    for j in vegetables[0]:
        print(j)
    break
print(vegetables[0:3])
stri="hello world"
print(stri[0:9:2])
vegetables[2]='Chicken'
# print(vegetables)
#append()-->Adds the elements at the end of the list
#insert(index,element)-->Add the  element at any position
#extend()-->Adds the elements of two or more dictionary into one


vegetables.append('mutton')
print(vegetables)
vegetables.insert(1,"mutton")
print(vegetables)

items =['mangoes','apples','bananas','grapes','pineapple']
items.insert(1,"mosambi")
print(items)
vegetables.extend(items)
print(vegetables)
items.pop(0)
print(items)
#remove(name)
#del()
#pop(index)
del items[1]
print(items)
a='kalyan'
b='amrutha'
c='non'
d='pon'
lst=[]
lst2=[]
#append()
lst.append(a)
lst.append(b)
lst.append(c)
lst.append(d)
print(lst)
#insert()
lst.insert(3,6)
#xtend()
lst2.extend(lst)
print(lst2)