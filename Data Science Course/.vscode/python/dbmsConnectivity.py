import sqlite3 as sq
database = sq.connect("company.db")
print("Database successfully Created")
database.execute('''create table redy(
                 name char(50),
                 age int(3)
                 );
                 ''')
print("Table created")
database.execute('''insert into fun values('Kalyan',20),('amrutha',20)''')
print("values inserted")
