import asyncio
from threading import Thread

import numpy as np
from stable_baselines3 import A2C
from poke_env import to_id_str
from poke_env import AccountConfiguration
from poke_env.player import Gen8EnvSinglePlayer, RandomPlayer
from brock import SimpleRLPlayer as Brock
from red import SimpleRLPlayer as Red

# class RandomGen8EnvPlayer(Gen8EnvSinglePlayer):
#     def embed_battle(self, battle):
#         return np.array([0])



def env_algorithm(player, n_battles, **kwargs):
    brokM = kwargs.get('brockM', None)
    redM = kwargs.get('redM', None)

    if brokM:
        model = A2C.load(brokM)
        model.set_env(player)
        playername = "Brock"
    if redM:
        model = A2C.load(redM)
        model.set_env(player)
        playername = "Red"
    obs, reward, done, _, info = player.step(0)
    for i in range(n_battles):
        print("battle number", i)
        done = False
        print("calling reset")
        obs, _ = player.reset()
        
        while not done:
            # print("playing next battle")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = player.step(action)
            if done:
                print("action taken", done)
    print(playername, "won", player.n_won_battles, "battles against", player._opponent, player.n_lost_battles )

async def launch_battles(player, opponent):
    battles_coroutine = asyncio.gather(
        player.send_challenges(
            opponent=to_id_str(opponent.username),
            n_challenges=1,
            to_wait=opponent.logged_in,
        ),
        opponent.accept_challenges(opponent=to_id_str(player.username), n_challenges=1),
    )
    await battles_coroutine


def env_algorithm_wrapper(player, kwargs):
    env_algorithm(player, **kwargs)
    print("i print here cuz i am cool =========================")
    player._start_new_battle = False
    # while True:
    #     try:
    #         player.reset()
    #     except OSError:
    #         break


p1 = Red(log_level=25, opponent=RandomPlayer(), start_challenging=False)

p2 = Brock(log_level=25, opponent=p1, start_challenging=False)

p1.set_opponent(p2)

p1._start_new_battle = True
p2._start_new_battle = False

loop = asyncio.get_event_loop()

env_algorithm_kwargs = {"n_battles": 100, "redM": "randommax100TV5HPMOOD"}

t1 = Thread(target=lambda: env_algorithm_wrapper(p1, env_algorithm_kwargs))
t1.start()

env_algorithm_kwargs = {"n_battles": 100, "brockM": "hotbrock_switching"}

t2 = Thread(target=lambda: env_algorithm_wrapper(p2, env_algorithm_kwargs))
t2.start()

while p1._start_new_battle:
    loop.run_until_complete(launch_battles(p1, p2))
t1.join()
t2.join()