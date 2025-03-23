""" 
Based on PureJaxRL Implementation of PPO
"""
import pdb
import time
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import pickle

from tqdm import tqdm
import cProfile
import pandas as pd
import sys

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt



from actor_networks import GraphActor, GraphLstmActor, MlpActor, MlpLstmActor
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    entropy: jnp.ndarray
    shaped_reward: jnp.ndarray
    actor_hidden: jnp.ndarray
    critic_hidden: jnp.ndarray
    other_action: jnp.ndarray
    past_sa_pairs: dict

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config, team_model_list=None):
    env = make('overcooked', **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    sp_batch_size = config["NUM_ACTORS"] // env.num_agents
    
    temp_reset = lambda key: env.custom_reset(key, random_reset=True, random_flip=False, layout=env.layout)
    reset_env = jax.jit(temp_reset)

    @scan_tqdm(100)
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

    env = LogWrapper(env)
    
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, team_id = 0, team_model_list=None):
        if config['MODEL_TYPE'] == 'graph':
            if config['LSTM']:
                model_init = GraphLstmActor
            else:
                model_init = GraphActor
        else:
            if config['LSTM']:
                model_init = MlpLstmActor
            else:
                model_init = MlpActor
        # INIT NETWORK
        hidden_dim = 256 if config['ENV_KWARGS']['random_reset'] else 128
        network = model_init(env.action_space().n, observation_shape=env.observation_space().shape, activation=config["ACTIVATION"], model_other_agent=config["MODEL_OTHER_AGENT"], e3t_baseline=config["E3T_BASELINE"], hidden_dim=hidden_dim)
        network_other = model_init(env.action_space().n, observation_shape=env.observation_space().shape, activation=config["ACTIVATION"], model_other_agent=config["MODEL_OTHER_AGENT"], e3t_baseline=config["E3T_BASELINE"], hidden_dim=hidden_dim)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        
        init_x = init_x.flatten()
        init_previous_5_state_action = {'obs': jnp.zeros((1, 5, *env.observation_space().shape)), 'action': jnp.zeros((1, 5))}
        # init_hidden = jnp.zeros((1, env.observation_space().shape[0] * env.observation_space().shape[1], 32))
        # init_hidden_actor = init_hidden
        # init_hidden_critic = init_hidden

        network_params = network.init(_rng, init_x, previous_state_action=init_previous_5_state_action)
        _rng, _ = jax.random.split(_rng)
        network_other_params = network_other.init(_rng, init_x, previous_state_action=init_previous_5_state_action)  # used for decentralized training and execution
        
        if team_model_list is None:
            loaded_model = None
            loaded_model_other = None
        else:
            def team_0(unused):
                return team_model_list[0]
            def team_1(unused):
                return team_model_list[1]
            loaded_model, loaded_model_other = jax.lax.cond(team_id == 1, team_1, team_0, None)

        if loaded_model is not None:
            network_params = loaded_model
        if loaded_model_other is not None:
            network_other_params = loaded_model_other

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        train_state_other = TrainState.create(
            apply_fn=network_other.apply,
            params=network_other_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        # TRAIN LOOP
        @scan_tqdm(int(config["NUM_UPDATES"]))
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            # @scan_tqdm(int(config["NUM_STEPS"]))
            def _env_step(runner_state, unused):
                train_state, train_state_other, env_state, last_obs, rng, old_actor_h_1, old_critic_h_1, old_actor_h_2, old_critic_h_2, count, past_5_sa_pairs = runner_state
                
                # SELECT ACTION
                rng, _rng, _rng_other = jax.random.split(rng, 3)

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])  # the first half of axis 0 will be agent 0, the second half agent 1
                
                # Randomly add entropy to the policy
                add_ent_pol_1 = jax.random.bernoulli(rng)
                rng, _rng = jax.random.split(rng)

                def get_e3t_action(args):
                    pi_ego, k = args
                    pi_random = distrax.Categorical(logits=jnp.zeros_like(pi_ego.logits))  # get uniform random policy
                    pi_e3t_probs = (1-config["E3T_EPSILON"])*pi_ego.probs + config["E3T_EPSILON"]*pi_random.probs  # get mixture policy probs
                    pi_e3t = distrax.Categorical(probs=pi_e3t_probs)  # convert probs to policy
                    sampled_a = pi_e3t.sample(seed=k)
                    log_prob_a = pi_e3t.log_prob(sampled_a)
                    entropy_a = pi_e3t.entropy()
                    return sampled_a, log_prob_a, entropy_a
                def get_base_action(args):
                    pi_ego, k = args
                    sampled_a = pi_ego.sample(seed=k)
                    log_prob_a = pi_ego.log_prob(sampled_a)
                    entropy_a = pi_ego.entropy()
                    return sampled_a, log_prob_a, entropy_a

                # Get first agent's actions
                pi_1, value_1, actor_h_1, critic_h_1, pred_pi_2 = train_state.apply_fn(train_state.params, obs_batch[:sp_batch_size, :], old_actor_h_1, old_critic_h_1, past_5_sa_pairs['agent_1'])
                action_1, log_prob_1, entropy_1 = jax.lax.cond(jnp.logical_and(config["E3T_BASELINE"], add_ent_pol_1), get_e3t_action, get_base_action, (pi_1, _rng))

                def use_base_params(args):
                    obs_batch, old_actor_h_2, old_critic_h_2, prev_sa_pairs = args
                    return train_state.apply_fn(train_state.params, obs_batch[sp_batch_size:, :], old_actor_h_2, old_critic_h_2, prev_sa_pairs)
                def use_other_params(args):
                    obs_batch, old_actor_h_2, old_critic_h_2, prev_sa_pairs = args
                    return train_state_other.apply_fn(train_state_other.params, obs_batch[sp_batch_size:, :], old_actor_h_2, old_critic_h_2, prev_sa_pairs)

                # Get second agent's actions
                poli_2_out = jax.lax.cond(config["SHARE_WEIGHTS"], use_base_params, use_other_params, (obs_batch, old_actor_h_2, old_critic_h_2, past_5_sa_pairs['agent_0']))
                
                pi_2, value_2, actor_h_2, critic_h_2, pred_pi_1 = poli_2_out
                
                action_2, log_prob_2, entropy_2 = jax.lax.cond(jnp.logical_and(config["E3T_BASELINE"], ~add_ent_pol_1), get_e3t_action, get_base_action, (pi_2, _rng_other))

                # store hidden states
                actor_hidden_states = {'agent_0': old_actor_h_1, 'agent_1': old_actor_h_2}
                critic_hidden_states = {'agent_0': old_critic_h_1, 'agent_1': old_critic_h_2}

                # stack actions
                action = jnp.concatenate([action_1, action_2], axis=0)
                log_prob = jnp.concatenate([log_prob_1, log_prob_2], axis=0)
                value = jnp.concatenate([value_1, value_2], axis=0)
                entropy = jnp.concatenate([entropy_1, entropy_2], axis=0)
                
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                # Update last sa pairs
                past_5_sa_pairs['agent_0']['obs'] = past_5_sa_pairs['agent_0']['obs'].at[:, :-1, :, :, :].set(past_5_sa_pairs['agent_0']['obs'][:, 1:, :, :, :])
                past_5_sa_pairs['agent_0']['obs'] = past_5_sa_pairs['agent_0']['obs'].at[:, -1, :, :, :].set(last_obs['agent_0'])
                past_5_sa_pairs['agent_0']['action'] = past_5_sa_pairs['agent_0']['action'].at[:, :-1].set(past_5_sa_pairs['agent_0']['action'][:, 1:])
                past_5_sa_pairs['agent_0']['action'] = past_5_sa_pairs['agent_0']['action'].at[:, -1].set(env_act['agent_0'])
                
                past_5_sa_pairs['agent_1']['obs'] = past_5_sa_pairs['agent_1']['obs'].at[:, :-1, :, :, :].set(past_5_sa_pairs['agent_1']['obs'][:, 1:, :, :, :])
                past_5_sa_pairs['agent_1']['obs'] = past_5_sa_pairs['agent_1']['obs'].at[:, -1, :, :, :].set(last_obs['agent_1'])
                past_5_sa_pairs['agent_1']['action'] = past_5_sa_pairs['agent_1']['action'].at[:, :-1].set(past_5_sa_pairs['agent_1']['action'][:, 1:])
                past_5_sa_pairs['agent_1']['action'] = past_5_sa_pairs['agent_1']['action'].at[:, -1].set(env_act['agent_1'])

                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                annealed_shaped_reward_frac = 1 - (count * config["NUM_ENVS"] * 2 / config["TOTAL_TIMESTEPS"])  # want to anneal by halfway through
                annealed_shaped_reward_frac = jnp.clip(annealed_shaped_reward_frac, 0.0, 1.0)
                annealed_shaped_reward_alice = (annealed_shaped_reward_frac * info['shaped_rewards']['agent_0']) 
                annealed_shaped_reward_bob = (annealed_shaped_reward_frac * info['shaped_rewards']['agent_1'])
                reward['agent_0'] += (annealed_shaped_reward_alice * 0)  # actually don't use shaped reward
                reward['agent_1'] += (annealed_shaped_reward_bob * 0)  # actually don't use shaped reward
                count += 1.0  # NEED THIS FOR ANNEALING REWARD
                del info['shaped_rewards']
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    entropy,
                    batchify({'agent_0': annealed_shaped_reward_alice, 'agent_1': annealed_shaped_reward_bob}, env.agents, config["NUM_ACTORS"]).squeeze(), # add shaped reward to transition
                    actor_hidden_states,  
                    critic_hidden_states,
                    None,  # dummy value that has low cost to store,
                    past_5_sa_pairs
                )
                runner_state = (train_state, train_state_other, env_state, obsv, rng, actor_h_1, critic_h_1, actor_h_2, critic_h_2, count, past_5_sa_pairs)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(int(config["NUM_STEPS"])), config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            '''TODO: Get second model to do second half of last_obs_batch'''
            train_state, train_state_other, env_state, last_obs, rng, actor_h_1, critic_h_1, actor_h_2, critic_h_2, count, past_5_sa_pairs = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val_1, _, _, _ = train_state.apply_fn(train_state.params, last_obs_batch[:sp_batch_size, :], actor_h_1, critic_h_1, past_5_sa_pairs['agent_1'])
            
            def use_base_params(args):
                obs_batch, actor_h_2, critic_h_2, prev_sa_pairs = args
                return train_state.apply_fn(train_state.params, obs_batch[sp_batch_size:, :], actor_h_2, critic_h_2, prev_sa_pairs)
            def use_other_params(args):
                obs_batch, actor_h_2, critic_h_2, prev_sa_pairs = args
                return train_state_other.apply_fn(train_state_other.params, obs_batch[sp_batch_size:, :], actor_h_2, critic_h_2, prev_sa_pairs)
        
            poli_2_out = jax.lax.cond(config["SHARE_WEIGHTS"], use_base_params, use_other_params, (last_obs_batch, actor_h_2, critic_h_2, past_5_sa_pairs['agent_0']))
            _, last_val_2, _, _, _ = poli_2_out
            last_val = jnp.concatenate([last_val_1, last_val_2], axis=0)

            '''
            Traj_batch
                done: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                action: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                value: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                reward: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                log_prob: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                obs: (num_timesteps, num_actors * num_envs, flattened_obs_size) - first half of axis 1 is agent 0, second half is agent 1
                info
                    returned episode - (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                    returned_episode_lengths - (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                    returned_episode_returns - (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                entropy: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                shaped_reward: (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
                actor_hidden: tuple - each is (num_timesteps, num_envs, h * w * 32)
                critic_hidden: tuple
                actor_hidden_other: tuple
                critic_hidden_other: tuple
            '''
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)  # (num_timesteps, num_actors * num_envs) - first half of axis 1 is agent 0, second half is agent 1
            
            # UPDATE NETWORK  - TODO: First split traj_batch, advantages, targets, rng in half, then update each one in parallel
            def _update_epoch(update_state, use_other_net, model_other_agent):
                train_state, traj_batch, advantages, targets, rng = update_state
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        def use_base(args):
                            params, obs = args
                            actor_h_1 = traj_batch.actor_hidden['agent_0']  # tuple of (minibatch size, 128)
                            critic_h_1 = traj_batch.critic_hidden['agent_0'] # tuple of (minibatch size, 128)
                            prev_sa_pairs = traj_batch.past_sa_pairs['agent_1']
                            return train_state.apply_fn(params, obs, actor_h_1, critic_h_1, prev_sa_pairs)
                        def use_other(args):
                            params, obs = args
                            actor_h_2 = traj_batch.actor_hidden['agent_1']
                            critic_h_2 = traj_batch.critic_hidden['agent_1']
                            prev_sa_pairs = traj_batch.past_sa_pairs['agent_0']
                            return train_state.apply_fn(params, obs, actor_h_2, critic_h_2, prev_sa_pairs)
                        pi, value, actor_h, critic_h, other_pi = jax.lax.cond(use_other_net, use_other, use_base, (params, traj_batch.obs))  # passing the right hidden state should be done outside this section
                        log_prob = pi.log_prob(traj_batch.action)


                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # model of other agents loss
                        def calc_moa_loss(args):
                            other_pi, other_action = args
                            other_log_prob = other_pi.log_prob(other_action)
                            # NLL loss for other agent
                            other_loss = -other_log_prob
                            other_loss = other_loss.mean()
                            return other_loss
                        dummy_moa_loss = lambda x: jnp.array(0.0)
                        moa_loss = jax.lax.cond(model_other_agent, calc_moa_loss, dummy_moa_loss, (other_pi, traj_batch.other_action))

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["MOA_COEF"] * moa_loss  # TODO: Need to anneal this so it's more valuable later in an episode
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"

                # TODO: Comment out below line if we're doing one model shared weights
                batch_size = batch_size // env.num_agents # factor out number of actors for self play separate weights

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)  # (num timepoints, num envs, feature size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch  # converts everything to (batch, feature size)
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            _update_epoch_moa = lambda x, y: _update_epoch(x, y, jnp.logical_or(config["MODEL_OTHER_AGENT"], config["E3T_BASELINE"]))
            traj_batch_1 = Transition(
                traj_batch.done[:, :sp_batch_size], 
                traj_batch.action[:, :sp_batch_size], 
                traj_batch.value[:, :sp_batch_size], 
                traj_batch.reward[:, :sp_batch_size], 
                traj_batch.log_prob[:, :sp_batch_size], 
                traj_batch.obs[:, :sp_batch_size, :], 
                {
                    'returned_episode': traj_batch.info['returned_episode'][:, :sp_batch_size],
                    'returned_episode_lengths': traj_batch.info['returned_episode_lengths'][:, :sp_batch_size],
                    'returned_episode_returns': traj_batch.info['returned_episode_returns'][:, :sp_batch_size]
                }, 
                traj_batch.entropy[:, :sp_batch_size], 
                traj_batch.shaped_reward[:, :sp_batch_size],
                traj_batch.actor_hidden,  # ((# timesteps, num envs, 128), (#))
                traj_batch.critic_hidden,
                traj_batch.action[:, sp_batch_size:],
                traj_batch.past_sa_pairs
            )
            traj_batch_2 = Transition(
                traj_batch.done[:, sp_batch_size:], 
                traj_batch.action[:, sp_batch_size:], 
                traj_batch.value[:, sp_batch_size:], 
                traj_batch.reward[:, sp_batch_size:], 
                traj_batch.log_prob[:, sp_batch_size:], 
                traj_batch.obs[:, sp_batch_size:, :], 
                {
                    'returned_episode': traj_batch.info['returned_episode'][:, sp_batch_size:],
                    'returned_episode_lengths': traj_batch.info['returned_episode_lengths'][:, sp_batch_size:],
                    'returned_episode_returns': traj_batch.info['returned_episode_returns'][:, sp_batch_size:]
                }, 
                traj_batch.entropy[:, sp_batch_size:], 
                traj_batch.shaped_reward[:, sp_batch_size:],
                traj_batch.actor_hidden,
                traj_batch.critic_hidden,
                traj_batch.action[:, :sp_batch_size],
                traj_batch.past_sa_pairs
            )

            def decentralized_weight_update(args):
                train_state, train_state_other, traj_batch_1, traj_batch_2, advantages, targets, rng = args
                update_state = (train_state, traj_batch_1, advantages[:, :sp_batch_size], targets[:, :sp_batch_size], rng)
                use_other = jnp.repeat(jnp.array(False), config["UPDATE_EPOCHS"] // 2)  # use agent 2 hidden state flag
                update_state, loss_info = jax.lax.scan(
                    _update_epoch_moa, update_state, use_other, config["UPDATE_EPOCHS"] // 2
                )
                train_state = update_state[0]
                rng = update_state[-1]
                update_state = (train_state_other, traj_batch_2, advantages[:, sp_batch_size:], targets[:, sp_batch_size:], rng)
                use_other = jnp.repeat(jnp.array(True), config["UPDATE_EPOCHS"] // 2)  # use agent 2 hidden state flag
                update_state, loss_info = jax.lax.scan(
                    _update_epoch_moa, update_state, use_other, config["UPDATE_EPOCHS"] // 2
                )
                # train_state_other = update_state[0]
                train_state_other = update_state[0]
                rng = update_state[-1]
                return (train_state, train_state_other, rng)
            def centralized_weight_update(args):
                train_state, train_state_other, traj_batch_1, traj_batch_2, advantages, targets, rng = args
                update_state = (train_state, traj_batch_1, advantages[:, :sp_batch_size], targets[:, :sp_batch_size], rng)
                use_other = jnp.repeat(jnp.array(False), config["UPDATE_EPOCHS"] // 2)  # use agent 2 hidden state flag
                update_state, loss_info = jax.lax.scan(
                    _update_epoch_moa, update_state, use_other, config["UPDATE_EPOCHS"] // 2
                )
                train_state = update_state[0]
                rng = update_state[-1]
                update_state = (train_state, traj_batch_2, advantages[:, sp_batch_size:], targets[:, sp_batch_size:], rng)
                use_other = jnp.repeat(jnp.array(True), config["UPDATE_EPOCHS"] // 2)  # use agent 2 hidden state flag
                update_state, loss_info = jax.lax.scan(
                    _update_epoch_moa, update_state, use_other, config["UPDATE_EPOCHS"] // 2
                )
                # train_state_other = update_state[0]
                train_state = update_state[0]
                rng = update_state[-1]
                return (train_state, train_state_other, rng)
            train_state, train_state_other, rng = jax.lax.cond(config["SHARE_WEIGHTS"], centralized_weight_update, decentralized_weight_update, (train_state, train_state_other, traj_batch_1, traj_batch_2, advantages, targets, rng))
            
            metric = traj_batch.info
            metric['policy_entropy'] = traj_batch.entropy
            metric['shaped_reward'] = traj_batch.shaped_reward
            # reset hidden states
            reset_hidden = lambda x: (jnp.zeros_like(x[0]), jnp.zeros_like(x[1]))
            actor_h_1 = reset_hidden(actor_h_1)
            critic_h_1 = reset_hidden(critic_h_1)
            actor_h_2 = reset_hidden(actor_h_2)
            critic_h_2 = reset_hidden(critic_h_2)
            past_5_sa_pairs = {'agent_0': {'obs': jnp.zeros((5, 128)), 'action': jnp.zeros((5, 128))}, 'agent_1': {'obs': jnp.zeros((5, 128)), 'action': jnp.zeros((5, 128))}}
            past_5_sa_pairs['agent_0']['obs'] = last_obs['agent_0'][:, None, :, :, :].repeat(5, axis=1)  # last obs should auto reset at the end of an episode so this is really the first obs
            past_5_sa_pairs['agent_0']['action'] = jnp.ones((config["NUM_ENVS"], 5)) * 4
            past_5_sa_pairs['agent_1']['obs'] = last_obs['agent_1'][:, None, :, :, :].repeat(5, axis=1)
            past_5_sa_pairs['agent_1']['action'] = jnp.ones((config["NUM_ENVS"], 5)) * 4
            runner_state = (train_state, train_state_other, env_state, last_obs, rng, actor_h_1, critic_h_1, actor_h_2, critic_h_2, count, past_5_sa_pairs)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        init_hidden_state = lambda x, y: (jnp.zeros((x, y)), jnp.zeros((x, y)))
        actor_h_1 = init_hidden_state(config["NUM_ENVS"], hidden_dim)
        actor_h_2 = init_hidden_state(config["NUM_ENVS"], hidden_dim)
        critic_h_1 = init_hidden_state(config["NUM_ENVS"], hidden_dim)
        critic_h_2 = init_hidden_state(config["NUM_ENVS"], hidden_dim)
        past_5_sa_pairs = {'agent_0': {'obs': jnp.zeros((5, hidden_dim)), 'action': jnp.zeros((5, hidden_dim))}, 'agent_1': {'obs': jnp.zeros((5, hidden_dim)), 'action': jnp.zeros((5, hidden_dim))}}
        past_5_sa_pairs['agent_0']['obs'] = obsv['agent_0'][:, None, :, :, :].repeat(5, axis=1)
        past_5_sa_pairs['agent_0']['action'] = jnp.ones((config["NUM_ENVS"], 5)) * 4
        past_5_sa_pairs['agent_1']['obs'] = obsv['agent_1'][:, None, :, :, :].repeat(5, axis=1)
        past_5_sa_pairs['agent_1']['action'] = jnp.ones((config["NUM_ENVS"], 5)) * 4

        runner_state = (train_state, train_state_other, env_state, obsv, _rng, actor_h_1, critic_h_1, actor_h_2, critic_h_2, jnp.array(0.), past_5_sa_pairs)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(int(config["NUM_UPDATES"])), config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    temp_train = lambda x, y: train(x, y, team_model_list=team_model_list)
    return temp_train

def moving_average(data, window_size):
    """
    Calculates the moving average of a data array every window_size steps.

    Args:
    data: A Jax array of numerical data.
    window_size: The number of steps for which to calculate the moving average.

    Returns:
    A Jax array containing the moving average at every window_size steps.
    """
    # Check if window size is valid
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")

    # Calculate the number of windows
    num_windows = data.shape[0] // window_size

    start = 0
    end = window_size
    mean_array = []
    for i in range(num_windows):
        if end <= len(data):
            mean_array.append(jnp.mean(data[start:end]))
        else:
            mean_array.append(jnp.mean(data[start:]))
        start += window_size
        end += window_size

    return jnp.array(mean_array)


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_overcooked")
def main(config):
    start = time.time()
    checkpoint_dir = "/mmfs1/gscratch/socialrl/kjha/psiphi-jepa/save/checkpoints"
    data_dir = "/mmfs1/gscratch/socialrl/kjha/psiphi-jepa/save/analysis"
    config = OmegaConf.to_container(config) 
    layout_name = config["ENV_KWARGS"]["layout"]
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]]
    seed = config["SEED"]
    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)
    num_seeds = 2
    # filename = f'{config["ENV_NAME"]}_{layout_name}Fixed_graph{seed}'
    # filename = f'{config["ENV_NAME"]}_proc_gen_graph{seed}'
    if config["ENV_KWARGS"]["random_reset"]:
        generation_type = "proc_gen"
    else:
        generation_type = f"{layout_name}_Fixed"
    filename = f'{config["ENV_NAME"]}_{generation_type}_{config["MODEL_TYPE"]}{seed}'
    lstm_appendage = '_lstm' if config["LSTM"] else ''
    filename += lstm_appendage
    moa_appendage = '_moa' if config["MODEL_OTHER_AGENT"] else ''
    filename += moa_appendage
    share_weights = '_share_weights' if config["SHARE_WEIGHTS"] else ''
    filename += share_weights
    single_appendage = '_single_agent' if config["SINGLE_AGENT"] else ''
    filename += single_appendage
    e3t_appendage = '_e3t' if config["E3T_BASELINE"] else ''
    filename += e3t_appendage

    if config["TRAINING"]:
        checkpoint_step = int(config["TOTAL_TIMESTEPS"] // config["NUM_CHECKPOINTS"])
        tmp_config = pickle.loads(pickle.dumps(config))
        tmp_config["TOTAL_TIMESTEPS"] = checkpoint_step
        for checkpoint_id in tqdm(range(config["NUM_CHECKPOINTS"])):
            if tmp_config["LOAD_MODEL"]:
                # load models across seeds and teams
                team_model_list = []
                rngs = []
                for i in range(num_seeds):
                    team_models = []
                    with open(f'{checkpoint_dir}/{filename}_team{i}_params_{i}.pkl', 'rb') as f:
                        loaded_file = pickle.load(f)
                        loaded_model = loaded_file['params']
                        team_models.append(loaded_model)
                        loaded_train_key = loaded_file['train_key']  # we only load one key per seed
                        rngs.append(loaded_train_key)
                    with open(f'{checkpoint_dir}/{filename}_other_team{i}_params_{i}.pkl', 'rb') as f:
                        loaded_file = pickle.load(f)
                        loaded_model_other = loaded_file['params']
                        team_models.append(loaded_model_other)
                    team_model_list.append(team_models)
                rngs = jnp.stack(rngs, axis=0)
            else:
                team_model_list = None
                rngs = None

            if checkpoint_id > 0:
                assert team_model_list is not None, "ERROR: Model loading failed"

            with jax.disable_jit(False):
                num_devices = jax.device_count()
                print(f"Number of devices: {num_devices}")
                if num_devices < num_seeds or num_seeds == 1:
                    print("Training 1+ seeds on a single device using jax.vmap")
                    train_jit = jax.jit(jax.vmap(make_train(tmp_config, team_model_list)))  # multiple seeds on single gpu
                else:
                    print("Training multiple seeds on different devices using jax.pmap")
                    train_jit = jax.pmap(make_train(tmp_config, team_model_list))  #  multiple gpus
                if rngs is None:
                    rngs = jax.random.split(rng, num_seeds)
                out = train_jit(rngs, jnp.arange(num_seeds))
            # pr.disable()
            # pr.dump_stats('training_rest.prof')


            # save data
            saved_Data = {}
            rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape((num_seeds, -1))
            shaped_rewards = out["metrics"]["shaped_reward"].mean(-1).reshape((num_seeds, -1))
            entropy = out["metrics"]["policy_entropy"].mean(-1).reshape((num_seeds, -1))
            saved_Data["rewards"] = rewards
            saved_Data["entropy"] = entropy
            saved_Data["shaped_rewards"] = shaped_rewards
            with open(f'{checkpoint_dir}/{filename}.pkl', 'wb') as f:  
                pickle.dump(saved_Data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # save models across seeds and teams
            for i in range(num_seeds):
                train_state = jax.tree_map(lambda x: x[i], out["runner_state"][0])  # agent 0, team i
                train_key = jax.tree_map(lambda x: x[i], out["runner_state"][4])
                with open(f'{checkpoint_dir}/{filename}_team{i}_params_{i}.pkl', 'wb') as f:
                    model_params = {'params': train_state.params, 'train_key': train_key}
                    pickle.dump(model_params, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved to: {checkpoint_dir}/{filename}_team{i}_params_{i}.pkl")
                with open(f'{checkpoint_dir}/{filename}_team{i}_params_{i}_ckpt{config["CHECKPOINT_ID"]}.pkl', 'wb') as f:
                    model_params = {'params': train_state.params, 'train_key': train_key}
                    pickle.dump(model_params, f, protocol=pickle.HIGHEST_PROTOCOL)

                train_state_other = jax.tree_map(lambda x: x[i], out["runner_state"][1])  # agent 1, team i, should be garbage if sharing weights
                with open(f'{checkpoint_dir}/{filename}_other_team{i}_params_{i}.pkl', 'wb') as f:
                    model_params = {'params': train_state_other.params, 'train_key': train_key}
                    pickle.dump(model_params, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f'{checkpoint_dir}/{filename}_other_team{i}_params_{i}_ckpt{config["CHECKPOINT_ID"]}.pkl', 'wb') as f:
                    model_params = {'params': train_state_other.params, 'train_key': train_key}
                    pickle.dump(model_params, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            
            print("Saved all models for all agents on teams 0 and 1 for one checkpoint")

            tmp_config = pickle.loads(pickle.dumps(config))
            tmp_config["LOAD_MODEL"] = True  # after a checkpoint is done we want to load it again for the next iteration
            tmp_config["TOTAL_TIMESTEPS"] = checkpoint_step
            del train_jit
        print("Training complete")

    # load data
    with open(f'{checkpoint_dir}/{filename}.pkl', 'rb') as f:  
        saved_data = pickle.load(f)
    rewards = saved_data["rewards"]
    entropy = saved_data["entropy"]
    shaped_rewards = saved_data["shaped_rewards"]

    
    print('** Saving Results **')
    fig, axs = plt.subplots(2, 2, figsize=(15, 7))

    unshaped_reward = rewards - shaped_rewards
    unshaped_reward_mean = unshaped_reward.mean(0)  # mean
    unshaped_reward_std = unshaped_reward.std(0) / np.sqrt(num_seeds)

    reward_mean = rewards.mean(0)  # mean 
    reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error
    shaped_reward_mean = shaped_rewards.mean(0)  # mean
    shaped_reward_std = shaped_rewards.std(0) / np.sqrt(num_seeds)

    print(f"Mean Reward: {reward_mean[-1]}, Std Error: {reward_std[-1]}")
    reward_mean = moving_average(reward_mean, int(1e4))
    reward_std = moving_average(reward_std, int(1e4))
    axs[0][0].plot(reward_mean)
    axs[0][0].fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    axs[0][0].set_xlabel("Update Step /  1e4")
    axs[0][0].set_ylabel("Return")

    entropy_mean = entropy.mean(0)  # mean
    entropy_std = entropy.std(0) / np.sqrt(num_seeds)
    print(f"Mean Entropy: {entropy_mean[-1]}, Std Error: {entropy_std[-1]}")
    entropy_mean = moving_average(entropy_mean, int(1e4))
    entropy_std = moving_average(entropy_std, int(1e4))

    axs[0][1].plot(entropy_mean)
    axs[0][1].fill_between(range(len(entropy_mean)), entropy_mean - entropy_std, entropy_mean + entropy_std, alpha=0.2)
    axs[0][1].set_xlabel("Update Step / 1e4")
    axs[0][1].set_ylabel("Policy Entropy")


    shaped_reward_mean = moving_average(shaped_reward_mean, int(1e4))
    shaped_reward_std = moving_average(shaped_reward_std, int(1e4))
    axs[1][0].plot(shaped_reward_mean)
    axs[1][0].fill_between(range(len(shaped_reward_mean)), shaped_reward_mean - shaped_reward_std, shaped_reward_mean + shaped_reward_std, alpha=0.2)
    axs[1][0].set_xlabel("Update Step / 1e4")
    axs[1][0].set_ylabel("Shaped Reward")

    # Plot reward - shaped_reward
    unshaped_reward_mean = moving_average(unshaped_reward_mean, int(1e4))
    unshaped_reward_std = moving_average(unshaped_reward_std, int(1e4))
    axs[1][1].plot(unshaped_reward_mean)
    axs[1][1].fill_between(range(len(unshaped_reward_mean)), unshaped_reward_mean - unshaped_reward_std, unshaped_reward_mean + unshaped_reward_std, alpha=0.2)
    axs[1][1].set_xlabel("Update Step / 1e4")
    axs[1][1].set_ylabel("Unshaped Reward")

    plt.savefig(f'training_pngs/{filename}.png')
    plt.close()
    print(f"Total time taken: {time.time() - start}")

    if config["TRAINING"]:
        exit(0)
    else:
        print("Training not enabled, exiting")
        exit(0)





if __name__ == "__main__":
    main()