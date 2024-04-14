import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
x= np.array([1,2,3,4,5])
y=np.array([10,5,30,20,50])
z=np.array([9,50,35,70,20])
print(x,y)
plt.plot(x,y,'d:r')
plt.plot(y,z,'d--b')
plt.xlabel('year')
plt.ylabel('production')
plt.title("PRODUCTION RATE IN INDIA-2023")
plt.grid()
# plt.subplot(1,2,1)
print(plt.show())


#subplot(rows,cols,plot_number) 
# x= np.array([1,2,3,4,5])
# y=np.array([10,5,30,20,50])
# z=np.array([9,50,35,70,20])
# plt.subplot(2,3,1)
# plt.show()
# #2nd
# x= np.array([1,2,3,4,5])
# y=np.array([10,5,30,20,50])
# z=np.array([9,50,35,70,20])
# plt.subplot(3,2,2)
# plt.show()
# #3rd
# x= np.array([1,2,3,4,5])
# y=np.array([10,5,30,20,50])
# z=np.array([9,50,35,70,20])
# plt.subplot(3,3,3)
# plt.show()

a =['one','two','three','four','five','six','seven','eight','nine','ten']
b=[10,20,15,12,25,40,30,40,60,55]
plt.bar(a,b,color='black')
print(plt.show())

#Histogram

c= np.random.normal(200,10,300)
print(c)
plt.hist(c)
print(plt.show())

#piechart

d = [100,39,45,70,90]
plt.pie(d,labels=['diarymilk','5-star','fuse','munch','kit-kat'],startangle=90,explode=[0.2,0.5,0,0,0],shadow=True,colors=['green','yellow','black','blue','violet'])
plt.legend(title='CHOCOLATES')
print(plt.show())