import numpy as np
arr1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
arr2 = np.array([[10,9,8,7,6],[5,4,3,2,1]])
print(arr1)
print('hello')
print(arr2)
#Addition
print('Addition')
print(arr1 + arr2)
print(arr1 + 20)
#Subtraction
print('Subtraction')
print(arr1 - arr2)
print(arr1 - 100)
#Division
print('Division')
print(arr1 / arr2)
print(arr1 / arr1)
print(arr1 / 4)
#Exponential
print('Exponential')
print(arr1 ** arr2)
#Multiplication (Cross & Dot product)
#Cross Product
print('Cross Product')
print(arr1 * arr2)
# print('Dot Product')
# arr3 = np.array([[11,13,14,1,2],[12,10,15,1,0]])
# print(np.dot(arr1,arr3))



#linear algebra
larr = np.array([[1,2],[3,4]])
#determinant of a matrix
print(np.linalg.det(larr))
print(np.linalg.inv(larr))