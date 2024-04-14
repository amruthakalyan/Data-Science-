import numpy as np
arr = np.array([[1,2,3,4],[5,6,7,8]])
print(arr)
print(arr.dtype)
a=arr.max()
print(a)
#dimension of the array
print(arr.ndim)
arr =np.zeros([3,3])
print(arr)
arr =np.ones([3,4])
print(arr)
arr = np.empty([2,3])
print(arr)

arr=np.full([4,4],12)
print(arr)
arr=np.eye(4)
print(arr)

arr=np.diag([4,3,4,7])
print(arr)