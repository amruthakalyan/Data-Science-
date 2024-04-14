#Numpy -> To perform Mathematical Operations
import numpy as np
#Pandas -> Data Manipulation top1
import pandas as pd
#Matplotlib -> Data Visualization tool
import matplotlib.pyplot as plt
#Seaborn -> Data Visualization tool
import seaborn as sns
#Sqlite3 ->Server-less Database
import sqlite3

db = '/content/drive/mydrive/Datasets/movies.sqlite'
conn = sqlite3.connect(db)
cur = conn.cursor()
#getting all the data about movies
#Establishing a connection with the Database
cur.execute("select * from movies")
#Creating a cursor Object
movies = cur.fetchall()
#Displaying the database data
print(movies)
#Creating a DataFrame
movies = pd.DataFrame(movies,columns=['id','Original_Title','Budget','Popularity','Release_date','Revenue','Title','vote_average','vote_count','Overview','Tagline','uid','director_id'])
#Displaying the data
print(movies)
print(movies.info())
#getting all the data about directors
#Fetching the data of movie table from the Database
cur.execute("select * from directors")
#Creating a cursor Object
directors = cur.fetchall()
#Displaying the database data
print(directors )
directors = pd.DataFrame(directors,columns=['name','id','gender','uid','department'])
print(directors)
print(directors.info())
#check how many movies are present in imdb table
#ans:select count(Title) from movies;
cur.execute('select count(Title) from movies')
count = cur.fetchall()
print(count)
#Find these 3 directors:James cameron,Luc Besson,Jhonwoo
#select * from directors where name in('James cameron','Luc Besson','johnwoo') 
cur.execute("select * from directors where name in('James cameron','Luc Besson','johnwoo') ")
three_directors = cur.fetchall()
print(three_directors)
#Find all the names of the directors whose name starts with 'Steven'
#ans:select name from director where name like 'Steven_%';
cur.execute('select name from director where name like "Steven%"')
name_like = cur.fetchall()
print(name_like)
#count the female directors
cur.execute('select count(*) from directors where where gender== 1')
females = cur.fetchall()
print(females[0][0])
#Find the name of the 10th first women directors
#select name from directors where gender== 1
cur.execute("select name from directors where gender== 1")
tenth = cur.fetchall()
print(tenth[9][0])
#What are the 3 most popular movies
cur.execute("select Title from movies order by popularity desc limit 3")
most_popular = cur.fetchall()
print(most_popular)
#what are 3 most bankable movies(budget)
cur.execute("select Title from movies order by budget desc limit 3")
most_bankable = cur.fetchall()
print(most_bankable)
#what is the most awarded avg rated movie since the jan 1st ,2000
cur.execute("select Original_title from movies where Release_date >'2000-01-01' order by vote_average desc limit 1")
most_awarded_avg = cur.fetchall()
print(most_awarded_avg)
#which movies were directed by Brenda Chapman
cur.execute("select Original_title from movies join directors on directors.id = movies.director_id where  directors.name ='Brenda Chapman'")
directed_by = cur.fetchall()
print(directed_by)
#Name the director who has made most movies
cur.execute("select name from directors join movies on directors.id = movie.director_id group by director_id order by count(name) desc limit 1")
director_movie = cur.fetchall()
print(director_movie)

#Name of the director who is most bankable
cur.execute("select name from directors join movies on director.id = movies.director_id group by director_id order by sum(budget) desc limit 1 ")
director_budget = cur.fetchall()
print(director_budget)



#Budget Analysis

#Tell the top 10 highest budget making movies
cur.execute("select * from movies order by budget desc limit 10")
top_10_budget = cur.fetchall()
df_top_10 = pd.DataFrame(top_10_budget ,coloums=['id','Original_Title','Budget','Popularity','Release_date','Revenue','Title','vote_average','vote_count','Overview','Tagline','uid','director_id'])
print(df_top_10)


#Revenue Analysis

#Find top 10 Revenue making movies
cur.execute("select * from movies order by revenue desc limit 10")
top_10_revenue =cur.fetchall()
df_top_10_revenue = pd.DataFrame(top_10_revenue,coloums=['id','Original_Title','Budget','Popularity','Release_date','Revenue','Title','vote_average','vote_count','Overview','Tagline','uid','director_id'])
print(df_top_10_revenue)


#Voting Analysis


#Find the most popular movies with highest vote_average
cur.execute("select * from movies order by vote_average desc")
most_pop = cur.fetchall()
df_most_pop = pd.DataFrame(most_pop,['id','Original_Title','Budget','Popularity','Release_date','Revenue','Title','vote_average','vote_count','Overview','Tagline','uid','director_id'])
print(df_most_pop)



#Director Analysis

#Name all the directors with the no.of movies and revenue
cur.execute("select directors.name,movies.count(Title),movies.sum(Revenue) from directors join movies where directors.id = movies.direcotr_id group by director_id order by sum(Revenue) desc ")
director_revenue = cur.fetchall()
df_director_revenue = pd.DataFrame(director_revenue,['id','Original_Title','Budget','Popularity','Release_date','Revenue','Title','vote_average','vote_count','Overview','Tagline','uid','director_id'])
print(df_director_revenue)

#Give the Title of the movies ,'release_date,'Budget','revenue',Popularity','vote_average'made by Stevel Spielberg
cur.execute("select Title,release_date,Budget,revenue,Popularity,vote_average from movies join directors on movies.director_id = directors.id where directors.name = 'Stevel Spielberg' ")
info = cur.fetchall()
df_info = pd.DataFrame(info)
print(df_info)














