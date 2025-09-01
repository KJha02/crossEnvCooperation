from .environments import (
    # SimpleMPE,
    # SimpleTagMPE,
    # SimpleWorldCommMPE,
    # SimpleSpreadMPE,
    # SimpleCryptoMPE,
    # SimpleSpeakerListenerMPE,
    # SimpleFacmacMPE,
    # SimpleFacmacMPE3a,
    # SimpleFacmacMPE6a,
    # SimpleFacmacMPE9a,
    # SimplePushMPE,
    # SimpleAdversaryMPE,
    # SimpleReferenceMPE,
    # SMAX,
    # HeuristicEnemySMAX,
    # LearnedPolicyEnemySMAX,
    # SwitchRiddle,
    # Ant,
    # Humanoid,
    # Hopper,
    # Walker2d,
    # HalfCheetah,
    # InTheGrid,
    # InTheGrid_2p,
    # Hanabi,
    Overcooked,
    # CoinGame,
    ToyCoop,
)
import inspect
import pdb
def filter_kwargs(kwargs, class_):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(class_.__init__).parameters}
    return filtered_kwargs


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # Toy Coop
    if env_id == "ToyCoop":
        env_kwargs = filter_kwargs(env_kwargs, ToyCoop)
        env = ToyCoop(**env_kwargs)

    # 1. MPE PettingZoo Environments
    if env_id == "MPE_simple_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleMPE)
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_tag_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleTagMPE)
        env = SimpleTagMPE(**env_kwargs)
    elif env_id == "MPE_simple_world_comm_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleWorldCommMPE)
        env = SimpleWorldCommMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleSpreadMPE)
        env = SimpleSpreadMPE(**env_kwargs)
    elif env_id == "MPE_simple_crypto_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleCryptoMPE)
        env = SimpleCryptoMPE(**env_kwargs)
    elif env_id == "MPE_simple_speaker_listener_v4":
        env_kwargs = filter_kwargs(env_kwargs, SimpleSpeakerListenerMPE)
        env = SimpleSpeakerListenerMPE(**env_kwargs)
    elif env_id == "MPE_simple_push_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimplePushMPE)
        env = SimplePushMPE(**env_kwargs)
    elif env_id == "MPE_simple_adversary_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleAdversaryMPE)
        env = SimpleAdversaryMPE(**env_kwargs)
    elif env_id == "MPE_simple_reference_v3":
        env_kwargs = filter_kwargs(env_kwargs, SimpleReferenceMPE)
        env = SimpleReferenceMPE(**env_kwargs)
    elif env_id == "MPE_simple_facmac_v1":
        env_kwargs = filter_kwargs(env_kwargs, SimpleFacmacMPE)
        env = SimpleFacmacMPE(**env_kwargs)
    elif env_id == "MPE_simple_facmac_3a_v1":
        env_kwargs = filter_kwargs(env_kwargs, SimpleFacmacMPE3a)
        env = SimpleFacmacMPE3a(**env_kwargs)
    elif env_id == "MPE_simple_facmac_6a_v1":
        env_kwargs = filter_kwargs(env_kwargs, SimpleFacmacMPE6a)
        env = SimpleFacmacMPE6a(**env_kwargs)
    elif env_id == "MPE_simple_facmac_9a_v1":
        env_kwargs = filter_kwargs(env_kwargs, SimpleFacmacMPE9a)
        env = SimpleFacmacMPE9a(**env_kwargs)

    # 2. Switch Riddle
    elif env_id == "switch_riddle":
        env_kwargs = filter_kwargs(env_kwargs, SwitchRiddle)
        env = SwitchRiddle(**env_kwargs)

    # 3. SMAX
    elif env_id == "SMAX":
        env_kwargs = filter_kwargs(env_kwargs, SMAX)
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env_kwargs = filter_kwargs(env_kwargs, HeuristicEnemySMAX)
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env_kwargs = filter_kwargs(env_kwargs, LearnedPolicyEnemySMAX)
        env = LearnedPolicyEnemySMAX(**env_kwargs)

    # 4. MABrax
    if env_id == "ant_4x2":
        env_kwargs = filter_kwargs(env_kwargs, Ant)
        env = Ant(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env_kwargs = filter_kwargs(env_kwargs, HalfCheetah)
        env = HalfCheetah(**env_kwargs)
    elif env_id == "hopper_3x1":
        env_kwargs = filter_kwargs(env_kwargs, Hopper)
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env_kwargs = filter_kwargs(env_kwargs, Humanoid)
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env_kwargs = filter_kwargs(env_kwargs, Walker2d)
        env = Walker2d(**env_kwargs)

    # 5. InTheGrid
    elif env_id == "storm":
        env_kwargs = filter_kwargs(env_kwargs, InTheGrid)
        env = InTheGrid(**env_kwargs)
    # 5. InTheGrid
    elif env_id == "storm_2p":
        env_kwargs = filter_kwargs(env_kwargs, InTheGrid_2p)
        env = InTheGrid_2p(**env_kwargs)
    
    # 6. Hanabi
    elif env_id == "hanabi":
        env_kwargs = filter_kwargs(env_kwargs, Hanabi)
        env = Hanabi(**env_kwargs)

    # 7. Overcooked
    elif env_id == "overcooked":
        env_kwargs = filter_kwargs(env_kwargs, Overcooked)
        env = Overcooked(**env_kwargs)

    # 8. Coin Game
    elif env_id == "coin_game":
        env_kwargs = filter_kwargs(env_kwargs, CoinGame)
        env = CoinGame(**env_kwargs)

    return env

registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    "storm",
    "storm_2p",
    "hanabi",
    "overcooked",
    "coin_game",
    "ToyCoop"
]
