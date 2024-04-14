rounds =int(input("Enter no.of rounds:"))

player1_arr =0
player2_arr=0
for i in  range(rounds):
    player1=input("Enter your choice (R,S,P):").upper()
    player2=input("Enter your choice (R,S,P):").upper()
    if(player1 ==player2):
        print("Match is draw")
    elif player1=='R' and player2=='S':
        player1_arr+=1
        print('player1 wins')
    elif player1=='R' and player2=='P':
        player1_arr+=1
        print('player1 wins')
    elif player1=='S' and player2=='P':
        player1_arr+=1 
        print('player1 wins')
    else:
        player2_arr+=1  
        print('player2 wins')    
if(player1_arr > player2_arr):
    print("Player 1 won the series")
elif(player1_arr < player2_arr):
    print("Player 2 won the series") 
else:
    print("The series is draw")   