"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import gc
from sklearn.manifold import TSNE

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9

from jaxmarl.viz.overcooked_jitted_visualizer import render_fn
import imageio
import matplotlib.pyplot as plt
from graph_layer import make_graph_toy_coop, GATLayer, make_graph_overcooked

import wandb
import functools
import pdb
from jax_tqdm import scan_tqdm
import pandas as pd
from tqdm import tqdm
import tsnex
from actor_networks import ScannedRNN, ActorCriticE3T, ActorCriticRNN

def initialize_environment(config):
    layout_name = config["ENV_KWARGS"]["layout"]
    config['layout_name'] = layout_name
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["ENV_NAME"] == "overcooked":
        def reset_env(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = env.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=False)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
                
            asymm_reset, key = reset_sub_dict(key, make_asymm_advantages_9x9)
            coord_ring_reset, key = reset_sub_dict(key, make_coord_ring_9x9)
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            forced_coord_reset, key = reset_sub_dict(key, make_forced_coord_9x9)
            cramped_room_reset, key = reset_sub_dict(key, make_cramped_room_9x9)
            layout_resets = [asymm_reset, coord_ring_reset, counter_circuit_reset, forced_coord_reset, cramped_room_reset]
            # stack all layouts
            stacked_layout_reset = jax.tree_map(lambda *x: jnp.stack(x), *layout_resets)
            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset = jax.tree_map(lambda x: x[index], stacked_layout_reset)
            return sampled_reset
        @scan_tqdm(100)
        def gen_held_out(runner_state, unused):
            (i,) = runner_state
            _, ho_state = reset_env(jax.random.key(i))
            res = (ho_state.goal_pos, ho_state.wall_map, ho_state.pot_pos)
            carry = (i+1,)
            return carry, res
        carry, res = jax.lax.scan(gen_held_out, (0,), jnp.arange(100), 100)
        ho_goal, ho_wall, ho_pot = [], [], []
        for layout_name, layout_dict in overcooked_layouts.items():  # add hand crafted ones to heldout set
            if "9" in layout_name:
                _, ho_state = env.custom_reset(jax.random.PRNGKey(0), random_reset=False, shuffle_inv_and_pot=False, layout=layout_dict)
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
    elif config["ENV_NAME"] == "ToyCoop":
        # Generate 100 held-out states for ToyCoop
        @scan_tqdm(100)
        def gen_held_out_toycoop(runner_state, unused):
            (i,) = runner_state
            key = jax.random.key(i)
            state = env.custom_reset_fn(key, random_reset=True)
            res = (state.agent_pos, state.goal_pos)
            carry = (i+1,)
            return carry, res
        
        carry, res = jax.lax.scan(gen_held_out_toycoop, (0,), jnp.arange(100), 100)
        ho_agent_pos, ho_goal_pos = res
        
        # Set the held-out states in the environment
        env.held_out_agent_pos = ho_agent_pos
        env.held_out_goal_pos = ho_goal_pos
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env


def get_rollouts(model_param_1, model_param_2, config, env, network_1, network_2, seed=0):
    
    def _step(carry, unused):
        train_state_params_1, train_state_params_2, env_state, last_obs, last_done, hstate_1, hstate_2, rng = carry
        
        # Select action
        rng, _rng = jax.random.split(rng)
        obs_batch = jnp.stack([last_obs[a].flatten() for a in env.agents])

        agent_positions = jnp.stack([env_state.env_state.agent_pos for a in env.agents])
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
            agent_positions[np.newaxis, :]
        )
        res = network_1.apply(train_state_params_1, hstate_1, ac_in)
        hstate_1, pi_1, value_1 = res[0], res[1], res[2]
        pi_1 = distrax.Categorical(logits=pi_1.logits * config["TEST_KWARGS"]["beta"])
        res = network_2.apply(train_state_params_2, hstate_2, ac_in)
        hstate_2, pi_2, value_2 = res[0], res[1], res[2]
        pi_2 = distrax.Categorical(logits=pi_2.logits * config["TEST_KWARGS"]["beta"])

        action_1 = pi_1.sample(seed=_rng)[0]
        action_1 = jnp.where(config["TEST_KWARGS"]["argmax"], jnp.argmax(pi_1.probs, 2)[0], action_1)
        action_1_prob_distrib = pi_1.probs[0, 0, :]
        action_2 = pi_2.sample(seed=_rng)[0]
        action_2 = jnp.where(config["TEST_KWARGS"]["argmax"], jnp.argmax(pi_2.probs, 2)[0], action_2)
        action_2_prob_distrib = pi_2.probs[0, 1, :]
        action_prob_dict = {env.agents[0]: action_1_prob_distrib, env.agents[1]: action_2_prob_distrib}

        # Convert action to env format
        env_act = {env.agents[0]: action_1[0], env.agents[1]: action_2[1]}

        # Step environment
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, env_act)
        
        done_batch = jnp.array([done[a] for a in env.agents])
        transition = (env_state.env_state, obsv, done_batch, env_act, reward, action_prob_dict)
        carry = (train_state_params_1, train_state_params_2, env_state, obsv, done_batch, hstate_1, hstate_2, rng)
        return carry, transition
    
    # Initialize environment and RNN state
    rng = jax.random.PRNGKey(seed)

    def get_rollout(rng, train_state_params_1=model_param_1, train_state_params_2=model_param_2, env=env, config=config):
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng)
        init_hstate_1 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        init_hstate_2 = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        done_batch = jnp.zeros(env.num_agents, dtype=bool)
        
        init_carry = (train_state_params_1, train_state_params_2, env_state, obsv, done_batch, init_hstate_1, init_hstate_2, rng)
        _, trajectory = jax.lax.scan(_step, init_carry, None, config["NUM_STEPS"])
        return trajectory, env_state.env_state, obsv

    rollouts_fn = jax.jit(jax.vmap(get_rollout, in_axes=(0,)))
    rollouts_res = rollouts_fn(jax.random.split(rng, config["TEST_KWARGS"]["num_trajs"]))
    trajectories, init_env_states, init_obsvs = rollouts_res
    return (trajectories, init_env_states, init_obsvs)

def load_ik_models(config):
    param_list = []
    seed_list = []
    for seed in range(6):
        load_path = f"ckpts/ippo/{config['ENV_NAME']}/cramped_room_9/ikTrue/reset_all/graphTrue"
        try:
            with open(f"{load_path}/seed{seed}_ckpt29_improved.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
        except:
            continue
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)
    return param_stack, jnp.array(seed_list)
def load_ik_finetune_models(config):
    param_list = []
    seed_list = []
    for seed in range(6):
        load_path = f"ckpts/ippo/{config['ENV_NAME']}/{config['ENV_KWARGS']['layout']}/ikFalse/reset_all/graphTrue"
        try:
            with open(f"{load_path}/seed{seed}_ckpt1_improved_finetune.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
        except:
            continue
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)
    return param_stack, jnp.array(seed_list)
def load_sk_models(config):
    param_list = []
    seed_list = []
    for seed in range(6):
        load_path = f"ckpts/ippo/{config['ENV_NAME']}/{config['ENV_KWARGS']['layout']}/ikFalse/reset_all/graphTrue"
        try:
            with open(f"{load_path}/seed{seed}_ckpt15_improved.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
        except:
            continue
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)
    return param_stack, jnp.array(seed_list)
def load_e3t_models(config):
    param_list = []
    seed_list = []
    for seed in range(6):
        load_path = f"ckpts/ippo/{config['ENV_NAME']}/{config['ENV_KWARGS']['layout']}/ikFalse/reset_all/graphTrue"
        try:
            with open(f"{load_path}/seed{seed}_ckpt15_e3t.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
        except:
            continue
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)
    return param_stack, jnp.array(seed_list)
def load_fcp_models(config):
    param_list = []
    seed_list = []
    for seed in range(6):
        load_path = f"ckpts/ippo/{config['ENV_NAME']}/{config['ENV_KWARGS']['layout']}/ikFalse/reset_all/graphTrue"
        try:
            with open(f"{load_path}/fcp_seed{seed}_ckpt19_improved.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
        except:
            continue
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)
    return param_stack, jnp.array(seed_list)


@hydra.main(version_base=None, config_path="config", config_name="ippo_final")
def main(config):
    config = OmegaConf.to_container(config)
    if config["FCP"]:
        fcp_str = "fcp_"
    else:
        fcp_str = ""
    if config["TRAIN_KWARGS"]["finetune"]:
        finetune_appendage = "_improved_finetune"
    else:
        finetune_appendage = "_improved"
    # if config['TRAIN_KWARGS']['finetune']:
    #     config['LR'] = config['LR'] / 10
    #     finetune_appendage = "_finetune"
    #     fcp_str = "fcp_"
    # else:
    #     finetune_appendage = "_improved"
    #     fcp_str = ""
    config["ENV_KWARGS"]["shuffle_inv_and_pot"] = False
    config["ENV_KWARGS"]["check_held_out"] = False
    filepath = f"ckpts/ippo/{config['ENV_NAME']}"
    if config["ENV_NAME"] == "overcooked":
        filepath += f"/{config['ENV_KWARGS']['layout']}"
    filepath = f"{filepath}/ik{config["TEST_KWARGS"]["ik"]}/{config['ENV_KWARGS']['random_reset_fn']}/graph{config["GRAPH_NET"]}"
    # make path if it doesn't exist
    os.makedirs(filepath, exist_ok=True)


    ##################
    # Load all models for current ckpt id
    ##################
    ik_param_stack, ik_seed_list = load_ik_models(config)
    sk_param_stack, sk_seed_list = load_sk_models(config)
    fcp_param_stack, fcp_seed_list = load_fcp_models(config)
    e3t_param_stack, e3t_seed_list = load_e3t_models(config)
    ik_finetune_param_stack, ik_finetune_seed_list = load_ik_finetune_models(config)
    assert len(ik_seed_list) > 0
    assert len(sk_seed_list) > 0
    assert len(fcp_seed_list) > 0
    assert len(e3t_seed_list) > 0
    assert len(ik_finetune_seed_list) > 0


    # gc.collect()
    # i want to get all pairs of seeds as a single array of (# pairs, 2)

    ##################
    # Initialize environment and network
    ##################
    env = initialize_environment(config)
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    regular_network = ActorCriticRNN(env.action_space("agent_0").n, config=config)
    e3t_network = ActorCriticE3T(env.action_space("agent_0").n, config=config)

    ik_info = (ik_param_stack, ik_seed_list, regular_network, 'ik')
    sk_info = (sk_param_stack, sk_seed_list, regular_network, 'sk')
    fcp_info = (fcp_param_stack, fcp_seed_list, regular_network, 'fcp')
    e3t_info = (e3t_param_stack, e3t_seed_list, e3t_network, 'e3t')
    ik_finetune_info = (ik_finetune_param_stack, ik_finetune_seed_list, regular_network, 'ik_finetune')

    info_list = [ik_info, sk_info, fcp_info, e3t_info, ik_finetune_info]



    df_dict = {'seed_1': [], 'seed_2': [], 'reward': [], 'algo_1': [], 'algo_2': []}

    for algo_1 in info_list:
        algo_1_params, algo_1_seed_list, algo_1_network, algo_1_name = algo_1
        for algo_2 in info_list:
            algo_2_params, algo_2_seed_list, algo_2_network, algo_2_name = algo_2
            print(f"Evaluating {algo_1_name} vs {algo_2_name}")


            seed_pairs = jnp.array(jnp.meshgrid(jnp.arange(len(algo_1_seed_list)), jnp.arange(len(algo_2_seed_list))))
            seed_pairs = seed_pairs.reshape((2, -1)).T


            ##################
            # Evaluate pairs
            ##################
            def eval_pair(seed_pair, seed_list_1, seed_list_2, param_stack_1, param_stack_2, network_1=algo_1_network, network_2=algo_2_network, config=config, env=env):
                seed_1, seed_2 = seed_pair[0], seed_pair[1]
                param_1 = jax.tree_map(lambda x: x[seed_1], param_stack_1)
                param_2 = jax.tree_map(lambda x: x[seed_2], param_stack_2)

                (trajectories, init_env_states, init_obsvs) = get_rollouts(param_1, param_2, config, env, network_1, network_2)
                rewards = trajectories[4]['agent_0'].sum(axis=1)  # axis 1 is originally each timestep in a single trajectory, want cumulative reward by end
                true_seed_1 = seed_list_1[seed_1]
                true_seed_2 = seed_list_2[seed_2]
                return (true_seed_1, true_seed_2, rewards, trajectories, init_env_states)

            eval_pair_fn = jax.jit(jax.vmap(eval_pair, in_axes=(0, None, None, None, None)))
            eval_pair_res = eval_pair_fn(seed_pairs, algo_1_seed_list, algo_2_seed_list, algo_1_params, algo_2_params)
            true_seed_1, true_seed_2, rewards, trajectories, init_env_states = eval_pair_res
            for i in tqdm(range(len(true_seed_1))):
                for j in range(len(rewards[i])):
                    df_dict['seed_1'].append(true_seed_1[i])
                    df_dict['seed_2'].append(true_seed_2[i])
                    df_dict['reward'].append(rewards[i][j])
                    df_dict['algo_1'].append(algo_1_name)
                    df_dict['algo_2'].append(algo_2_name)
    df = pd.DataFrame(df_dict)
    df.to_csv(f"{filepath}/cross_algo_eval_onIK_{config['ENV_KWARGS']['random_reset']}.csv", index=False)
    print(f"Saved data to {filepath}/cross_algo_eval_onIK_{config['ENV_KWARGS']['random_reset']}.csv")


if __name__ == "__main__":
    main()


    # FOR FUTURE REFERENCE:
    '''
        loop over graph/no graph  (this will be config)
        loop over ik train vs sk train  (this will be test kwargs)
        loop over ckpt id  (this will be train kwargs)
        loop over eval on ik vs eval on sk  (this will be env kwargs)
    '''

    # For overcooked
    '''
    # first eval sk grids on sk model
    for layout in "cramped_room_padded" "counter_circuit_padded" "forced_coord_padded" "asymm_advantages_padded" "coord_ring_padded"
        for graph vs no graph
            for train sk
                for test ik = False vs True
                    for ckpt id
                        run eval
    '''