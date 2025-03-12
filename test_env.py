import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.environment.pokemon import Pokemon
from poke_env.data import GenData
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer

GEN_9_DATA =  GenData.from_gen(9)

pokemon_types_gen_9 = {
    "BUG": 1,
    "DARK": 2,
    "DRAGON": 3,
    "ELECTRIC": 4,
    "FAIRY": 5,
    "FIGHTING": 6,
    "FIRE": 7,
    "FLYING": 8,
    "GHOST": 9,
    "GRASS": 10,
    "GROUND": 11,
    "ICE": 12,
    "NORMAL": 13,
    "POISON": 14,
    "PSYCHIC": 15,
    "ROCK": 16,
    "STEEL": 17,
    "WATER": 18,
    "STELLAR": 19
}

class SimpleRLPlayer(Gen9EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        # print("battle team values", battle.team.values())
        # print("\n\n\n")
        # print("battle other info", battle.active_pokemon, battle.available_switches, battle.opponent_side_conditions, battle.opponent_active_pokemon)
        # print("player info", battle.team)
        # active_poke = battle.active_pokemon
  
        # print(active_poke.base_stats)
        # print(active_poke.current_hp_fraction)
        # print(battle.active_pokemon)
        # print(battle.active_pokemon)
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        current_hp = battle.active_pokemon.current_hp_fraction
        opponent_hp = battle.opponent_active_pokemon.current_hp_fraction
        print("Printing observation =====>>>>>")
        print(battle.observations)
        actpok = battle.active_pokemon
        # print(actpok.type_1.name, actpok.type_2.name if actpok.type_2 else '')
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
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
                [current_hp, opponent_hp],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=5
        )
    
    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1, 1, 1]
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
        
finished_episodes = 0
TEST_EPISODES = 1
random_player = RandomPlayer()
second_player = MaxDamagePlayer()
env_player = SimpleRLPlayer(opponent=second_player)
# await random_player.battle_against(second_player, n_battles=1)
model = A2C.load("hprandommax100TV5HPMOOD")
model.set_env(env_player)

obs, reward, done, _, info = env_player.step(0)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env_player.step(action)

    if done:
        finished_episodes += 1
        if finished_episodes >= TEST_EPISODES:
            break
        obs, _ = env_player.reset()

