# sets
s = {11,2,3,4,5,5,3}
print(s)#sets doesnot allow duplicates

#To access any element from the set,it should be typecasted into list then only the values can be accessed.
listi = list(s)
print(listi)
listi[0]=12
print(listi)
#you can perform CRUD operation on sets after typecasting it into a list then we can again typecast the list back into the set


#CRUD operations in sets
s=set(listi)
s.add("kiwi")  #add()
s.remove("kiwi")  #remove()
s.discard("kiwi")  #discard()
s.clear()  #clear()
#remove() gives an error if the element doesnot present in the list
#discard() doesnot give any error,it discards the element if it is present in the set or else it does nothing.
print(s)
s2={"apples","banana","mangoes"}
s.update(s2)
print(s)

setA ={"Facebook","Amazon","Netflix","Google"}
setB = {"TCS","Wipro","Accenture","Jp Morgan","Google"}
 #Union()
setC=setA.union(setB)
print(setC)

#intersection()
setC= setA.intersection(setB)
print(setC)

#difference()
setC= setA.difference(setB)
print(setC)