import numpy as np
import time
from stable_baselines3 import A2C
from gymnasium.spaces import Box, Discrete
from poke_env.data import GenData
from poke_env.environment import Pokemon

from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer

import functools
import inspect

pokemon_types_gen_9 = {
    "BUG": 0,
    "DARK": 1,
    "DRAGON": 2,
    "ELECTRIC": 3,
    "FAIRY": 4,
    "FIGHTING": 5,
    "FIRE": 6,
    "FLYING": 7,
    "GHOST": 8,
    "GRASS": 9,
    "GROUND": 10,
    "ICE": 11,
    "NORMAL": 12,
    "POISON": 13,
    "PSYCHIC": 14,
    "ROCK": 15,
    "STEEL": 16,
    "WATER": 17,
    "STELLAR": 18
}
def encode_type(pokemon:Pokemon, gen_types):
    vec = [0] * len(gen_types)
    vec[gen_types[pokemon.type_1.name]] = 1
    if pokemon.type_2:
        vec[gen_types[pokemon.type_2.name]] = 1

    return vec

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
        active_poke = battle.active_pokemon
        opp_active_poke = battle.opponent_active_pokemon
        current_hp = active_poke.current_hp_fraction
        opponent_hp = opp_active_poke.current_hp_fraction
        current_type = encode_type(active_poke, pokemon_types_gen_9)
        opp_type = encode_type(opp_active_poke, pokemon_types_gen_9)
        # print(battle.observations)
        # print(battle.available_moves)
        
        # include poketypes for available switches.
        # TODO: check if player can see the action space for switching.
        # make sure that the returned observation space is fixed regardless of
        # number of available switches.
        switch_types = np.full((5,len(pokemon_types_gen_9)), -1) 
        # print(battle.available_switches)
        for i, pokemon in enumerate(battle.available_switches):
            # make sure there is an available switch 
            if pokemon: 
                pokemon_type = encode_type(pokemon, pokemon_types_gen_9)
                # print("printing pokemon type--------->", pokemon_type)
                switch_types[i] = np.array(pokemon_type)
        switch_types = switch_types.flatten()
        # print("printing swtich_types", switch_types)


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
                [current_hp, opponent_hp],
                current_type,
                opp_type,
                switch_types
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=4, hp_value=2, victory_value=20
        )
    
    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * len(pokemon_types_gen_9) * 2 + [-1] * 5 * len(pokemon_types_gen_9)
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1] + [1] * len(pokemon_types_gen_9) * 2 + [1] * 5 * len(pokemon_types_gen_9)
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
NB_TRAINING_STEPS = 200_000
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)
LEARN = True

if __name__ == "__main__":
    opponent = RandomPlayer()
    second_opponent = MaxDamagePlayer()
    env_player = SimpleRLPlayer(opponent=second_opponent)
    
    
    if LEARN:
        model = A2C("MlpPolicy", env_player, verbose=1, ent_coef=0.05)
        # model = A2C.load("hotbrock100")
        # model.set_env(env_player)
        model.learn(total_timesteps=NB_TRAINING_STEPS)
        model.save("hotbrock_switching")
    # obs, reward, done, _, info = env_player.step(0)
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, info = env_player.step(action)
    model = A2C.load("hotbrock500w200")
    model.set_env(env_player)
    finished_episodes = 0

    # # # env_player.close()
    # # env_player.reset_env(restart=False)
    # # obs, _ = env_player.reset()

    # env_player.close()
    # env_player = SimpleRLPlayer(opponent=RandomPlayer())
    obs, reward, done, _, info = env_player.step(0)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)
        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)

    finished_episodes = 0


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