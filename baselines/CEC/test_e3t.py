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


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree_map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["GRAPH_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            # if self.config["ENV_NAME"] == "overcooked":
            #     reshaped_obs = obs.reshape(-1, 7,7,26)
            # else:
            #     reshaped_obs = obs.reshape(-1, 5,5,3)
            reshaped_obs = obs.reshape(-1, *self.config["obs_dim"])
            # # use 2 conv nets
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"]*2,
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(reshaped_obs)
            # embedding = nn.relu(embedding)
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"],
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(embedding)
            # embedding = nn.relu(embedding)

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            # reshaped_obs = obs.reshape(-1, flattened_obs_dim)
            # reshaped_agent_positions = agent_positions.reshape((-1, 2, 2))
            # make_graph_fn = make_graph_overcooked if self.config["ENV_NAME"] == "overcooked" else make_graph_toy_coop
            # node_feats, adj_mat = jax.vmap(make_graph_fn)(reshaped_obs, reshaped_agent_positions)
            # embedding = GATLayer(self.config["FC_DIM_SIZE"], num_heads=2)(node_feats, adj_mat)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Model of other agent
        #########
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10)  # L2 normalization
        other_pi = distrax.Categorical(logits=prediction_other)

        #########
        # Actor
        #########
        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 2
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 3
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi


def get_rollouts(model_param_1, model_param_2, config, env, network, seed=0):
    
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
        hstate_1, pi_1, value_1, _ = network.apply(train_state_params_1, hstate_1, ac_in)
        pi_1 = distrax.Categorical(logits=pi_1.logits * config["TEST_KWARGS"]["beta"])
        hstate_2, pi_2, value_2, _ = network.apply(train_state_params_2, hstate_2, ac_in)
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


@hydra.main(version_base=None, config_path="config", config_name="ippo_final")
def main(config):
    config = OmegaConf.to_container(config)
    if config["FCP"]:
        fcp_str = "fcp_"
    else:
        fcp_str = ""
    finetune_appendage = "_e3t"
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
    param_list = []
    seed_list = []
    # if False andconfig["ENV_KWARGS"]["layout"] == "harder_counter_circuit" and config["ENV_NAME"] == "overcooked" and not config["ENV_KWARGS"]["random_reset"]:
    #     iter_range = range(6, 12)
    # else:
    #     iter_range = range(6)
    iter_range = range(6)
    for seed in iter_range:
        # try:
        if config["TEST_KWARGS"]["ik"] and config["ENV_NAME"] == "overcooked":  # want to load from ik model if we're testing ik model on overcooked
            load_path = f"ckpts/ippo/{config['ENV_NAME']}/cramped_room_9/ik{config['TEST_KWARGS']['ik']}/{config['ENV_KWARGS']['random_reset_fn']}/graph{config['GRAPH_NET']}"
        else:
            load_path = filepath
        try:
            with open(f"{load_path}/{fcp_str}seed{seed}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl", "rb") as f:
                previous_ckpt = pickle.load(f)
                model_params = previous_ckpt['params']
                param_list.append(model_params)
                seed_list.append(seed)
                del previous_ckpt
        except:
            continue
    if len(param_list) == 0:
        print(f"No models found")
        exit(0)
    seed_list = jnp.array(seed_list)
    # gc.collect()
    param_stack = jax.tree_map(lambda *x: jnp.stack(x), *param_list)      # stack params
    # gc.collect()
    # i want to get all pairs of seeds as a single array of (# pairs, 2)
    seed_pairs = jnp.array(jnp.meshgrid(jnp.arange(len(seed_list)), jnp.arange(len(seed_list))))
    seed_pairs = seed_pairs.reshape((2, -1)).T

    ##################
    # Initialize environment and network
    ##################
    env = initialize_environment(config)
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    env = LogWrapper(env, env_params={'random_reset_fn': config['ENV_KWARGS']['random_reset_fn']})
    network = ActorCriticRNN(env.action_space("agent_0").n, config=config)

    
    ##################
    # Evaluate pairs
    ##################
    @jax.jit
    def eval_pair(seed_pair, seed_list, param_stack, config=config, env=env, network=network):
        seed_1, seed_2 = seed_pair[0], seed_pair[1]
        param_1 = jax.tree_map(lambda x: x[seed_1], param_stack)
        param_2 = jax.tree_map(lambda x: x[seed_2], param_stack)

        (trajectories, init_env_states, init_obsvs) = get_rollouts(param_1, param_2, config, env, network)
        rewards = trajectories[4]['agent_0'].sum(axis=1)  # axis 1 is originally each timestep in a single trajectory, want cumulative reward by end
        true_seed_1 = seed_list[seed_1]
        true_seed_2 = seed_list[seed_2]
        return (true_seed_1, true_seed_2, rewards, trajectories, init_env_states)
    
    if not config['TEST_KWARGS']['plot']:
        eval_pair_fn = jax.jit(jax.vmap(eval_pair, in_axes=(0, None, None)))
        eval_pair_res = eval_pair_fn(seed_pairs, seed_list, param_stack)
        true_seed_1, true_seed_2, rewards, trajectories, init_env_states = eval_pair_res
        ##################
        # Save data
        ##################
        df_dict = {'seed_1': [], 'seed_2': [], 'reward': []}
        for i in tqdm(range(len(true_seed_1))):
            seed_1, seed_2 = true_seed_1[i], true_seed_2[i]
            for j in range(rewards.shape[1]):
                reward = rewards[i][j]
                df_dict['seed_1'].append(seed_1)
                df_dict['seed_2'].append(seed_2)
                df_dict['reward'].append(reward)
        df = pd.DataFrame(df_dict)
        df.to_csv(f"{filepath}/{fcp_str}eval_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.csv", index=False)
        print(f"Saved data to {filepath}/{fcp_str}eval_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.csv")
    else:
        # iterate over all self play pairs
        exit(0)
        for sp_seeds in tqdm(range(len(seed_pairs))):
            seed_val = seed_pairs[sp_seeds]
            seed_0, seed_1 = seed_val[0], seed_val[1]
            if seed_0 != seed_1:
                continue

            seed_0, seed_1, rewards, trajectories, init_env_states = eval_pair([seed_0, seed_1], seed_list, param_stack)
            all_freq_counts = []
            all_coordinate_embeddings = []
            for traj_num in tqdm(range(config['TEST_KWARGS']['num_trajs'])):
                traj = jax.tree_map(lambda x: x[traj_num], trajectories)
                env_states = [jax.tree_map(lambda x: x[traj_num], init_env_states)]
                traj_rewards = [0]
                action_probs_0 = []
                action_probs_1 = []
                action_0 = []
                action_1 = []
                coordinate_embeddings = []
                freq_count = None
                for timepoint in range(config['NUM_STEPS']):
                    timestep = jax.tree_map(lambda x: x[timepoint], traj)

                    obs = timestep[1]
                    # create mask of 1 if channel 0 or 1 is 1, 0 otherwise
                    mask = jnp.logical_or(obs[env.agents[0]][:,:,0], obs[env.agents[0]][:,:,1]).astype(jnp.float32)
                    if freq_count is None:
                        freq_count = mask
                    else:
                        freq_count = freq_count + mask

                    action_dict = timestep[3]
                    action_0.append(action_dict[env.agents[0]])
                    action_1.append(action_dict[env.agents[1]])

                    action_prob_dict = timestep[-1]
                    action_probs_0.append(action_prob_dict[env.agents[0]])
                    action_probs_1.append(action_prob_dict[env.agents[1]])
                    
                    env_state = timestep[0]
                    traj_rewards.append(timestep[4]['agent_0'])
                    env_states.append(env_state)
                    
                    flattened_obs = obs[env.agents[0]].flatten()
                    joint_strat = jnp.concatenate([flattened_obs, action_prob_dict[env.agents[0]], action_prob_dict[env.agents[1]]])
                    coordinate_embeddings.append(np.array(joint_strat))

                coordinate_embeddings = np.stack(coordinate_embeddings)
                tsne = TSNE(n_components=2, random_state=42)
                coordinate_embeddings = tsne.fit_transform(coordinate_embeddings)

                all_coordinate_embeddings.append(coordinate_embeddings)
                all_freq_counts.append(freq_count)

                    
                # add arbitrary final actions and action probs
                action_0.append(4)
                action_1.append(4)
                action_probs_0.append(jnp.ones(env.action_space("agent_0").n) / env.action_space("agent_0").n)
                action_probs_1.append(jnp.ones(env.action_space("agent_1").n) / env.action_space("agent_1").n)
                env_images = [render_fn(s) for s in env_states]
                traj_rewards = jnp.cumsum(jnp.array(traj_rewards))
                
                frames = []
                for t in range(len(env_images)):
                    # Create figure with subplots
                    fig = plt.figure(figsize=(16, 8))
                    
                    # Environment image
                    ax1 = plt.subplot(221)
                    ax1.imshow(env_images[t])
                    ax1.axis('off')
                    ax1.set_title('Environment')
                    
                    # Reward plot
                    ax2 = plt.subplot(222)
                    ax2.plot(range(len(traj_rewards)), traj_rewards, 'b-')
                    ax2.scatter(t, traj_rewards[t], color='red', s=100)
                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Cumulative Reward')
                    ax2.set_title('Reward Over Time')
                    
                    # Agent 0 action probabilities
                    ax3 = plt.subplot(223)
                    probs = action_probs_0[t]
                    bars = ax3.bar(range(len(probs)), probs)
                    chosen_action = action_0[t]
                    bars[chosen_action].set_color('green')
                    ax3.set_xlabel('Action')
                    ax3.set_ylabel('Probability')
                    ax3.set_title('Agent 0 Action Probabilities')
                    
                    # Agent 1 action probabilities  
                    ax4 = plt.subplot(224)
                    probs = action_probs_1[t]
                    bars = ax4.bar(range(len(probs)), probs)
                    chosen_action = action_1[t]
                    bars[chosen_action].set_color('green')
                    ax4.set_xlabel('Action')
                    ax4.set_ylabel('Probability')
                    ax4.set_title('Agent 1 Action Probabilities')
                    
                    plt.tight_layout()
                    
                    # Convert figure to image array
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(image)
                    plt.close()

                # # make a gif out of images that continuously loops
                imageio.mimsave(f"{filepath}/{fcp_str}seed{seed_0}x{seed_1}_{fcp_str}eval_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}_traj{traj_num}.gif", frames, fps=10, loop=True)
                print(f"Saved gif to {filepath}/{fcp_str}seed{seed_0}x{seed_1}_{fcp_str}eval_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}_traj{traj_num}.gif")
            all_freq_counts = np.stack(all_freq_counts).mean(axis=0)
            all_coordinate_embeddings = np.stack(all_coordinate_embeddings).mean(axis=0)
            with open(f"{filepath}/{fcp_str}qualEval_seed{seed_0}_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl", "wb") as f:
                pickle.dump({'visit_freqs': all_freq_counts, 'strat_embeddings': all_coordinate_embeddings}, f)
            print(f"Saved data to {filepath}/{fcp_str}qualEval_seed{seed_0}_on_ik{config['ENV_KWARGS']['random_reset']}_ckpt{config['TRAIN_KWARGS']['ckpt_id']}{finetune_appendage}.pkl")

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