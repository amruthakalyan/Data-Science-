lweight = float(input("Enter the luggage weight:"))
if lweight < 50:
    Tax_charge =0
elif lweight >=50:
    Tax_charge = abs(25+(50-lweight )* 2)
else:
    pass
print(Tax_charge)        