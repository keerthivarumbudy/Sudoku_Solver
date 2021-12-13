import os
import csv

boards = ['boards/empty-3x3.txt', 'boards/random-3x3.txt']
opponents = ['greedy_player']
with open('results.csv', 'a', newline='') as csvfile:
    data = csv.writer(csvfile, delimiter=',')
    data.writerow(['Board', 'First Player', 'Second Player', 'Time', 'Winner'])
# os.system("clear screen")
for i in range(10):
    for board in boards:
        for opponent in opponents:
            for time in [0.1, 0.5, 1, 5]:
                command = "python3 simulate_game.py --first team25_A1 --second " + opponent + " --board "+ board + " --time " + str(time)
                os.system(command)
                command = "python3 simulate_game.py --first "+ opponent +" --second  team25_A1 --board " + board + " --time " + str(time)
                os.system(command)