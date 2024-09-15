import sys
import asyncio

sys.path.append("../src")

from poke_env import RandomPlayer
from poke_env.data import GenData
from poke_env import cross_evaluate
from tabulate import tabulate

# The RandomPlayer is a basic agent that makes decisions randomly,
# serving as a starting point for more complex agent development.
async def play_rand_battle():
    third_player = RandomPlayer()
    random_player = RandomPlayer()
    second_player = RandomPlayer()
    players = [random_player, second_player, third_player]
    print("initiating battle")
    cross_evaluation = await cross_evaluate(players, n_challenges=50)
    
    table = [["-"]+[p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print(tabulate(table))
    # await random_player.battle_against(second_player, n_battles=1)



asyncio.run(play_rand_battle())


# The battle_against method initiates a battle between two players.
# Here we are using asynchronous programming (await) to start the battle.
