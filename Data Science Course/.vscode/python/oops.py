# class className:
#     statements

class First():
    def Demo(self,marks):
        if marks > 75:
            print("Pass")
        else:
            print("Fail")    
#Demo("99")  Error ,first we need to create an object to access the members of a class
#syntax:
#object_name = class_name(arguments)
a = First()
a.Demo(99)
a.Demo(10)


#scope  of a variable

var = 10
def scope():
    var=8
    var =var + 20
    return var
print(scope())


def scope():
   global var
   var =var + 20
   return var
print(scope())


#inheritance
class Animal:
    def speak(self):
        print("Animal is speaking")
class Dog():
    def bark(self):
       print("Dog is barking") 
class Puppy(Animal,Dog):
    def eat(self):
       print("puppy is eating")             
# a = Animal()
b = Puppy()
b.bark()
b.speak()
b.eat()

class First:
    def __init__(self,name,age,address):
       self.name = name
       self.age = age
       self.address = address
    def Name(self):
        print(f"Your name is {self.name}")
    def Age(self):
        print(f"Your age is {self.age}")
    def Address(self):
        print(f"Your address is {self.address}")    
a = First('kalyan',20,'Hyderabad')
a.Name()
a.Age()
a.Address()