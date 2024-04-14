#Lambda Function
#syntax:lambda argument(multiple):logic(one)
lam = lambda a,b: (a+b)
print(lam(2,3))

weight = float(input("Enter weight in kgs:"))
height = float(input("Enter height in meters:"))
bmi = weight +(height ** height)
print('Your Body Max Index should be: ',bmi)