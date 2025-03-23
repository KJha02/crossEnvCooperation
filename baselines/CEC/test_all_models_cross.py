import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl import make
from jax_tqdm import scan_tqdm
from tqdm import tqdm

from actor_networks import GraphActor, GraphLstmActor, MlpActor, MlpLstmActor

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pickle
from pdb import set_trace as T
from pathlib import Path

from actor_networks import GraphActor, GraphLstmActor, MlpActor, MlpLstmActor

def initialize_environment(config):
    env = make('overcooked', **config["ENV_KWARGS"])
    env.training = False  # always on held out set


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
    config["obs_dim"] = env.observation_space(env.agents[0]).shape
    return env

def load_models():
    model_dict = {
        'fixedPlay': {'agent_0': [], 'agent_1': []},
        'fixedPlayE3T': {'agent_0': [], 'agent_1': []},
        'procPlay': {'agent_0': [], 'agent_1': []},
        'procPlayE3T': {'agent_0': [], 'agent_1': []}
    }
    checkpoint_dir = "/mmfs1/gscratch/socialrl/kjha/psiphi-jepa/save/checkpoints"
    for ckpt_id in [5]:
        for useE3T in [False, True]:
            for model_type in ['graph', 'mlp']:
                for useLSTM in [False, True]:
                    if model_type == 'graph' and useLSTM == True:
                        model = GraphLstmActor
                    elif model_type == 'graph' and useLSTM == False:
                        model = GraphActor
                    elif model_type == 'mlp' and useLSTM == True:
                        model = MlpLstmActor
                    elif model_type == 'mlp' and useLSTM == False:
                        model = MlpActor
                    else:
                        raise ValueError(f'Invalid model type: {model_type} and useLSTM: {useLSTM}')

                    for seed in [69, 70, 71]:
                        for layout_name in ['cramped_room_padded', 'counter_circuit_padded', 'forced_coord_padded', 'asymm_advantages_padded', 'coord_ring_padded']:
                            generation_type = f"{layout_name}_Fixed"
                            filename = f'overcooked_{generation_type}_{model_type}{seed}'
                            lstm_appendage = '_lstm' if useLSTM else ''
                            filename += lstm_appendage
                            e3t_appendage = '_e3t' if useE3T else ''
                            filename += e3t_appendage

                            for i in range(2):
                                with open(f'{checkpoint_dir}/{filename}_team{i}_params_{i}_ckpt{ckpt_id}.pkl', 'rb') as f:
                                    loaded_model_params = pickle.load(f)['params']
                                    if useE3T:
                                        model_dict['fixedPlayE3T']['agent_0'].append((loaded_model_params, model, useE3T, useLSTM, layout_name, model_type, ckpt_id))
                                    else:
                                        model_dict['fixedPlay']['agent_0'].append((loaded_model_params, model, useE3T, useLSTM, layout_name, model_type, ckpt_id))
                                with open(f'{checkpoint_dir}/{filename}_other_team{i}_params_{i}_ckpt{ckpt_id}.pkl', 'rb') as f:
                                    loaded_model_other_params = pickle.load(f)['params']
                                    if useE3T:
                                        model_dict['fixedPlayE3T']['agent_1'].append((loaded_model_other_params, model, useE3T, useLSTM, layout_name, model_type, ckpt_id))
                                    else:
                                        model_dict['fixedPlay']['agent_1'].append((loaded_model_other_params, model, useE3T, useLSTM, layout_name, model_type, ckpt_id))
                                    
                        generation_type = 'proc_gen'
                        filename = f'overcooked_{generation_type}_{model_type}{seed}'
                        lstm_appendage = '_lstm' if useLSTM else ''
                        filename += lstm_appendage
                        share_weights = '_share_weights'
                        filename += share_weights
                        e3t_appendage = '_e3t' if useE3T else ''
                        filename += e3t_appendage
                        for i in range(2):
                            with open(f'{checkpoint_dir}/{filename}_team{i}_params_{i}_ckpt{ckpt_id}.pkl', 'rb') as f:
                                loaded_model_params = pickle.load(f)['params'] # since we're sharing weights, we only need to load the params for one agent
                                if useE3T:
                                    model_dict['procPlayE3T']['agent_0'].append((loaded_model_params, model, useE3T, useLSTM, 'procGen', model_type, ckpt_id))
                                    model_dict['procPlayE3T']['agent_1'].append((loaded_model_params, model, useE3T, useLSTM, 'procGen', model_type, ckpt_id))
                                else:  
                                    model_dict['procPlay']['agent_0'].append((loaded_model_params, model, useE3T, useLSTM, 'procGen', model_type, ckpt_id))
                                    model_dict['procPlay']['agent_1'].append((loaded_model_params, model, useE3T, useLSTM, 'procGen', model_type, ckpt_id))
        
    return model_dict


def get_test_rollouts(train_state_params, other_train_state_params, network, other_network, config,  env,seed=0):

    n_actions = env.num_actions

    key = jax.random.PRNGKey(seed)
    key_r, key_a, key_b = jax.random.split(key, 3)
        
    network = model_init(n_actions, observation_shape=env.observation_space().shape, activation=config["ACTIVATION"], model_other_agent=config["MODEL_OTHER_AGENT"], e3t_baseline=config["E3T_0"])
    other_network = model_init(n_actions, observation_shape=env.observation_space().shape, activation=config["ACTIVATION"], model_other_agent=config["MODEL_OTHER_AGENT"], e3t_baseline=config["E3T_1"])

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    past_5_sa_pairs = {
        'agent_0': {'obs': jnp.zeros((1, 5, *env.observation_space().shape)), 'action': jnp.ones((1, 5)) * 4},
        'agent_1': {'obs': jnp.zeros((1, 5, *env.observation_space().shape)), 'action': jnp.ones((1, 5)) * 4}
    }

    network_params = network.init(key_a, init_x, previous_state_action=past_5_sa_pairs['agent_1'])
    other_network_params = other_network.init(key_b, init_x, previous_state_action=past_5_sa_pairs['agent_0'])

    network_params = train_state_params
    other_network_params = other_train_state_params


    def get_single_trajectory(k, network_params, other_network_params):
        # @scan_tqdm(int(config["NUM_STEPS"]))
        def env_step(carry, timestep):
            network_params, other_network_params, prior_state, prior_obs, k, foundReward, reward_timestep, actor_hidden_1, critic_hidden_1, actor_hidden_2, critic_hidden_2, past_5_sa_pairs = carry
            k, key_a0, key_a1, key_s = jax.random.split(k, 4)
            obs_0 = prior_obs['agent_0'].reshape(1, -1)
            obs_1 = prior_obs['agent_1'].reshape(1, -1)
            pi_0, _, actor_hidden_1, critic_hidden_1, pred_pi_1 = network.apply(network_params, obs_0, actor_hidden_1, critic_hidden_1, past_5_sa_pairs['agent_1'])
            actions = {}
            actions['agent_0'] = pi_0.sample(seed=key_a0)[0]

            single_agent_action = lambda args: (jnp.array(4), actor_hidden_2, critic_hidden_2)
            def multi_agent_action(args):
                o, k, actor_hidden_2, critic_hidden_2 = args
                pi_1, _, actor_hidden_2, critic_hidden_2, pred_pi_0 = network.apply(other_network_params, o, actor_hidden_2, critic_hidden_2, past_5_sa_pairs['agent_0'])
                return pi_1.sample(seed=k)[0], actor_hidden_2, critic_hidden_2
            actions['agent_1'], actor_hidden_2, critic_hidden_2 = jax.lax.cond(config["SINGLE_AGENT"], single_agent_action, multi_agent_action, (obs_1, key_a1, actor_hidden_2, critic_hidden_2))
            # STEP ENV
            obs, state, reward, done, info = env.step(key_s, prior_state, actions)

            reward_timestep = jnp.where(jnp.logical_and(reward['agent_0'] != 0, foundReward == 0), timestep + 1, reward_timestep)  # if this is our first time finding a reward, mark it
            # foundReward = jnp.logical_or(foundReward, reward['agent_0'] != 0)  # if we already found it keep it
            foundReward = jnp.where(reward['agent_0'] != 0, foundReward + 1, foundReward)
            

            # store transition
            joint_actions = jax.nn.one_hot(jnp.array([actions['agent_0'], actions['agent_1']]), n_actions)

            # Update past_5_sa_pairs
            past_5_sa_pairs['agent_0']['obs'] = past_5_sa_pairs['agent_0']['obs'].at[:, :-1, :, :, :].set(past_5_sa_pairs['agent_0']['obs'][:, 1:, :, :, :])
            past_5_sa_pairs['agent_0']['obs'] = past_5_sa_pairs['agent_0']['obs'].at[:, -1, :, :, :].set(prior_obs['agent_0'])
            past_5_sa_pairs['agent_0']['action'] = past_5_sa_pairs['agent_0']['action'].at[:, :-1].set(past_5_sa_pairs['agent_0']['action'][:, 1:])
            past_5_sa_pairs['agent_0']['action'] = past_5_sa_pairs['agent_0']['action'].at[:, -1].set(actions['agent_0'])
            
            past_5_sa_pairs['agent_1']['obs'] = past_5_sa_pairs['agent_1']['obs'].at[:, :-1, :, :, :].set(past_5_sa_pairs['agent_1']['obs'][:, 1:, :, :, :])
            past_5_sa_pairs['agent_1']['obs'] = past_5_sa_pairs['agent_1']['obs'].at[:, -1, :, :, :].set(prior_obs['agent_1'])
            past_5_sa_pairs['agent_1']['action'] = past_5_sa_pairs['agent_1']['action'].at[:, :-1].set(past_5_sa_pairs['agent_1']['action'][:, 1:])
            past_5_sa_pairs['agent_1']['action'] = past_5_sa_pairs['agent_1']['action'].at[:, -1].set(actions['agent_1'])

            carry = (network_params, other_network_params, state, obs, k, foundReward, reward_timestep, actor_hidden_1, critic_hidden_1, actor_hidden_2, critic_hidden_2, past_5_sa_pairs)
            res = (obs_0, obs_1, joint_actions, actions, reward, prior_state)
            return carry, res
        
        obs, init_state = env.reset(k)
        k = jax.random.split(k)[0]
        actor_hidden_1 = (jnp.zeros((1, 128)), jnp.zeros((1, 128)))
        critic_hidden_1 = (jnp.zeros((1, 128)), jnp.zeros((1, 128)))
        actor_hidden_2 = (jnp.zeros((1, 128)), jnp.zeros((1, 128)))
        critic_hidden_2 = (jnp.zeros((1, 128)), jnp.zeros((1, 128)))
        past_5_sa_pairs = {
            'agent_0': {'obs': None, 'action': jnp.ones((1, 5)) * 4},
            'agent_1': {'obs': None, 'action': jnp.ones((1, 5)) * 4}
        }
        past_5_sa_pairs['agent_0']['obs'] = obs['agent_0'][None][:, None, :, :, :].repeat(5, axis=1)
        past_5_sa_pairs['agent_1']['obs'] = obs['agent_1'][None][:, None, :, :, :].repeat(5, axis=1)
        
        carry = (network_params, other_network_params, init_state, obs, k, 0, int(config["NUM_STEPS"]), actor_hidden_1, critic_hidden_1, actor_hidden_2, critic_hidden_2, past_5_sa_pairs)
        carry, res = jax.lax.scan(env_step, carry, jnp.arange(int(config["NUM_STEPS"])), int(config["NUM_STEPS"]))
        (network_params, other_network_params, state, _, k, got_reward, reward_timestep, actor_hidden_1, critic_hidden_1, actor_hidden_2, critic_hidden_2, past_5_sa_pairs) = carry
        (obs_0_seq, obs_1_seq, joint_action_seq, action_seq, reward_seq, prior_states) = res
        num_walls = state.wall_map.sum()
        meta_res = (obs_0_seq, obs_1_seq, joint_action_seq, action_seq, reward_seq, num_walls, got_reward, reward_timestep, prior_states)

        return meta_res
    many_keys = jax.vmap(lambda x: jax.random.PRNGKey(x))(jnp.arange(int(config["NUM_TEST_EPOCHS"])))
    # many_keys = jax.random.split(key_r, int(int(config["NUM_TEST_EPOCHS"])))
    get_multi_trajectories = lambda k: get_single_trajectory(k, network_params, other_network_params)
    meta_res = jax.jit(jax.vmap(get_multi_trajectories))(many_keys)
    key = jax.random.split(key)[0]
    return key, meta_res, env, network, other_network



def test_models(model_dict):
    num_passes = []
    num_timesteps = []
    num_walls = []
    model_1_name = []
    model_2_name = []
    layout_namess = []
    use_lstm_1 = []
    use_e3t_1 = []
    use_lstm_2 = []
    use_e3t_2 = []
    trained_layout_name_1 = []
    trained_layout_name_2 = []
    model_type_1 = []
    model_type_2 = []
    ckpt_id_1 = []
    ckpt_id_2 = []
    for layout in ['cramped_room_padded', 'counter_circuit_padded', 'forced_coord_padded', 'asymm_advantages_padded', 'coord_ring_padded']:
        config = {
            "ENV_KWARGS": {
                "layout": layout,
                "random_reset": False,
                "max_steps": 256,
                "single_agent": False,
                "training": False
            },
            "ACTIVATION": "relu",
            "MODEL_OTHER_AGENT": False,
            "NUM_TEST_EPOCHS": 5,
            "NUM_STEPS": 256,
            "SINGLE_AGENT": False
        }
        env = initialize_environment(config)
        for model_1 in model_dict.keys():
            for model_2 in model_dict.keys():
                agent_0_models = model_dict[model_1]['agent_0']
                agent_1_models = model_dict[model_2]['agent_1']
                for agent_0_model in agent_0_models:
                    for agent_1_model in agent_1_models:
                        (loaded_model_other_params, other_model, other_useE3T, other_useLSTM, other_trained_layout_name, other_model_type, other_ckpt_id) = agent_1_model
                        (loaded_model_params, model, useE3T, useLSTM, trained_layout_name, model_type, ckpt_id) = agent_0_model
                        
                        config['E3T_0'] = useE3T
                        config['E3T_1'] = other_useE3T
                        key, meta_res, env, _, _ = get_test_rollouts(loaded_model_params, loaded_model_other_params, model, other_model, config, env, seed=0)
                        (obs_0_seq, obs_1_seq, joint_action_seq, action_seq, reward_seq, num_walls, got_reward, reward_timestep, state_seq) = meta_res
                        
                        num_passes.extend(got_reward)
                        num_timesteps.extend(reward_timestep)
                        num_walls.extend(num_walls)

                        use_lstm_1.extend([useLSTM] * len(got_reward))
                        use_e3t_1.extend([useE3T] * len(got_reward))
                        trained_layout_name_1.extend([trained_layout_name] * len(got_reward))
                        model_type_1.extend([model_type] * len(got_reward))
                        ckpt_id_1.extend([ckpt_id] * len(got_reward))

                        use_lstm_2.extend([other_useLSTM] * len(got_reward))
                        use_e3t_2.extend([other_useE3T] * len(got_reward))
                        trained_layout_name_2.extend([other_trained_layout_name] * len(got_reward))
                        model_type_2.extend([other_model_type] * len(got_reward))
                        ckpt_id_2.extend([other_ckpt_id] * len(got_reward))

                        model_1_name.extend([model_1] * len(got_reward))
                        model_2_name.extend([model_2] * len(got_reward))
                        layout_namess.extend([layout] * len(got_reward))
    data = {'num_passes': num_passes, 'num_timesteps': num_timesteps, 'num_walls': num_walls, 'model_1_name': model_1_name, 'model_2_name': model_2_name, 'trained_layout_name_1': trained_layout_name_1, 'trained_layout_name_2': trained_layout_name_2, 'use_lstm_1': use_lstm_1, 'use_lstm_2': use_lstm_2, 'use_e3t_1': use_e3t_1, 'use_e3t_2': use_e3t_2, 'model_type_1': model_type_1, 'model_type_2': model_type_2, 'ckpt_id_1': ckpt_id_1, 'ckpt_id_2': ckpt_id_2}
    df = pd.DataFrame(data)
    savepath = f"analysis_results/csvs"
    Path(savepath).mkdir(parents=True, exist_ok=True)
    save_name = f"{savepath}/test_all_models_cross.csv"
    df.to_csv(save_name)


if __name__ == "__main__":
    model_dict = load_models()
    test_models(model_dict)

                            
                            

                            
                
