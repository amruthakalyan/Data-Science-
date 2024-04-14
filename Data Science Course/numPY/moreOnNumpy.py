import numpy as np
#Stacking ( hstack() , vstack() )

arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[6,5,4],[3,2,1]])
print('vstack:')
print(np.vstack((arr1,arr2)))
print('hstack:')
print(np.hstack((arr1,arr2)))
print('split_array')
arr3 = np.array([1,2,3])
print(np.array_split(arr3,4))

#Searching
print('Searching')
newarr =np.array([[11,12,3,4,5,4,4]])
print(np.where(newarr == 4))
print(np.where(newarr > 4))
print(np.sort(newarr))
print(-np.sort(-newarr))

#Any() & All()

print(np.any(newarr == 3))
print(np.all(newarr == 3))

#lcm()
num1 = 25
num2 = 125
print('lcm:')
print(np.lcm(num1,num2))
lcmarr = np.array([1,2,3,4,5,6,7,8,9,10])
#reduce() func used for finding lcm of more than 2 elements. 
print(np.lcm.reduce(lcmarr))
print('GCD:')
print(np.gcd(num1,num2))
print(np.gcd.reduce(lcmarr))

#Linear Equation
# 3x + y = 9
# x  + 2y = 8
a = np.array([[3,1],[1,2]])
b = np.array([9,8])
#solve()
print(np.linalg.solve(a,b))

#Quadratic Equation
# x^2 + 4x = -4
qarr= np.array([1,4,4])
print(np.roots(qarr)) 
# print(np.vdot(a,b))

#mean()
print(np.mean(a))
print(np.median(a))
