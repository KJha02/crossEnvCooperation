import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl import make
from jax_tqdm import scan_tqdm
from tqdm import tqdm

from baselines.IPPO.actor_networks import GraphActor, GraphLstmActor, MlpActor, MlpLstmActor

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pickle
from pdb import set_trace as T
from pathlib import Path

def initialize_environment(config):
    env = make('overcooked', **config["ENV_KWARGS"])
    env.training = False  # this is so that we always are using the held out set
    

    temp_reset = lambda key: env.custom_reset(key, random_reset=True, random_flip=False, layout=env.layout)
    reset_env = jax.jit(temp_reset)
    def gen_held_out(runner_state, unused):
        (i,) = runner_state
        _, ho_state = reset_env(jax.random.key(i))
        res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
        carry = (i+1,)
        return carry, res
    carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
    ho_goal, ho_wall, ho_pot = [], [], []
    for layout_name, padded_layout in overcooked_layouts.items():  # add hand crafted ones to heldout set
        if "padded" in layout_name:
            _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, random_flip=False, layout=padded_layout)
            ho_goal.append(ho_state.goal_pos)
            ho_wall.append(ho_state.wall_map)
            ho_pot.append(ho_state.pot_pos)
    ho_goal = jnp.stack(ho_goal, axis=0)
    ho_wall = jnp.stack(ho_wall, axis=0)
    ho_pot = jnp.stack(ho_pot, axis=0)
    ho_goal = jnp.concatenate([res[0], ho_goal], axis=0)
    ho_wall = jnp.concatenate([res[1], ho_wall], axis=0)
    ho_pot = jnp.concatenate([res[2], ho_pot], axis=0)
    env.held_out_goal, env.held_out_wall, env.held_out_pot = (ho_goal, ho_wall, ho_pot)
    return env