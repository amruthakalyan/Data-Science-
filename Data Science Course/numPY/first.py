import time
import numpy as np
a=np.random.rand(10)
start = time.time()
mean_np=np.mean(a)
print(time.time()-start)


#Creating an array in numPY (ndarray)
#Zero dimensional array
#syntax: arr=()
arr = np.array(4)
print(arr)
print(type(arr))
print(arr.ndim)
print(arr.shape)


#one dimensional array 
#syntax: arr=([])
one_arr=np.array([2,5,6,7,7,7,8,2,3,9,10])
print(one_arr)

#Two dimensional array
#syntax: arr([[],[]])
Two_arr=np.array([
[1,2,3,4,5],
    [6,7,8]
    ])
print(Two_arr)