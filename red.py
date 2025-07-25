import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer

import functools
import inspect

def log_caller(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        stack = inspect.stack()
        caller = stack[1]
        print(f"{func.__name__} called by {caller.function} in {caller.filename}:{caller.lineno}")
        return func(*args, **kwargs)
    return wrapper
# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen9EnvSinglePlayer):
   
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        # ensure compatibility with describe embeddings
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=5
        )
    
    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

np.random.seed(0)


model_store = {}

# This is the function that will be used to train the a2c
def a2c_training(player, nb_steps):
    model = A2C("MlpPolicy", player, verbose=1)
    model.learn(total_timesteps=10_000)
    model_store[player] = model
    


def a2c_evaluation(player, nb_episodes):
    # Reset battle statistics
    model = model_store[player]
    player.reset_battles()
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


# NB_TRAINING_STEPS = 20_000
NB_TRAINING_STEPS = 50_000
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)
LEARN = False

if __name__ == "__main__":
    opponent = RandomPlayer()
    second_opponent = MaxDamagePlayer()
    env_player = SimpleRLPlayer(opponent=second_opponent)
    
    if LEARN:
        # model = A2C("MlpPolicy", env_player, verbose=1)
        model = A2C.load("random100TV5HPMOOD")
        model.set_env(env_player)
        model.learn(total_timesteps=NB_TRAINING_STEPS)
        model.save("randommax100TV5HPMOOD")
    # obs, reward, done, _, info = env_player.step(0)
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, info = env_player.step(action)
    model = A2C.load("randommax100TV5HPMOOD")
    model.set_env(env_player)
    finished_episodes = 0

    # # # env_player.close()
    # # env_player.reset_env(restart=False)
    # # obs, _ = env_player.reset()

    # env_player.close()

    # env_player = SimpleRLPlayer(opponent=RandomPlayer())
    # obs, reward, done, _, info = env_player.step(0)
    # while True:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, info = env_player.step(action)

    #     if done:
    #         finished_episodes += 1
    #         if finished_episodes >= TEST_EPISODES:
    #             break
    #         obs, _ = env_player.reset()

    # print("Won", env_player.n_won_battles, "battles against", env_player._opponent)

    # finished_episodes = 0


    # env_player.reset_battles()
    # obs, _ = env_player.reset()

    opponent = MaxDamagePlayer()
    env_player = SimpleRLPlayer(opponent=opponent)
    obs, reward, done, _, info = env_player.step(0)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            obs, _ = env_player.reset()
            if finished_episodes >= TEST_EPISODES:
                break

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)


# When trained on randomplayer wins 83% against rand player and 36% against max player