import os
import csv

boards = ['boards/easy-2x2.txt', 'boards/empty-3x3.txt', 'boards/random-3x3.txt', 'boards/hard-3x3.txt', 'boards/empty-4x4.txt', 'boards/random-4x4.txt']
opponents = ['team25_A2', 'greedy_player']
iter = 6
workers = 2

with open('../results.csv', 'a', newline='') as csvfile:
    data = csv.writer(csvfile, delimiter=',')
    data.writerow(['board', 'player1', 'player2', 'p1_wins', 'p2_wins', 'draws', 'time'])
# os.system("clear screen")
for board in boards:
    for opponent in opponents:
        for time in [0.5, 1, 5]:
            command = "python simulate_game_bulk.py --first team25_A2 --second " + opponent + " --board "+ board + " --time " + str(time) + " --iter " + str(iter) + " --workers " + str(workers)
            os.system(command)
            command = "python simulate_game_bulk.py --first "+ opponent +" --second team25_A2 --board " + board +" --time " + str(time) + " --iter " + str(iter) + " --workers " + str(workers)
            os.system(command)
