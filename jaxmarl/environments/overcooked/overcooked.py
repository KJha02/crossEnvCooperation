from collections import OrderedDict
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict
import pdb

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts
from jaxmarl.environments.overcooked.layouts import make_counter_circuit_9x9, make_forced_coord_9x9, make_coord_ring_9x9, make_asymm_advantages_9x9, make_cramped_room_9x9
from jaxmarl.environments.overcooked.layouts import single_cramped_room


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 10, # reward for putting ingredients 
    "PLATE_PICKUP_REWARD": 3, # reward for picking up a plate
    "SOUP_PICKUP_REWARD": 15, # reward for picking up a ready soup
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
    done = 6


@struct.dataclass
class State:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23 # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20 # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3 # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40 # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""
    def __init__(
            self,
            layout = FrozenDict(layouts["cramped_room_padded"]),
            random_reset: bool = True,
            max_steps: int = 256,
            single_agent: bool = False,
            check_held_out: bool = False,
            shuffle_inv_and_pot: bool = False
    ):
        # Sets self.num_agents to 2
        super().__init__(num_agents=2)

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        # self.height = layout["height"]
        # self.width = layout["width"]
        self.height = 9
        self.width = 9

        self.corners = jnp.array([0, self.width-1, self.width*(self.height-1), (self.width*self.height)-1], dtype=jnp.uint32)
        top_row = jnp.arange(1, self.width - 1)
        bottom_row = jnp.arange(self.width * (self.height - 1) + 1, self.width * self.height - 1)
        left_column = jnp.arange(self.width, self.width * (self.height - 1), self.width)
        right_column = jnp.arange(2 * self.width - 1, self.width * self.height - 1, self.width)

        # excludes corners
        self.border_indices = jnp.concatenate((top_row, bottom_row, left_column, right_column)).astype(jnp.uint32)

        # gets potential t junctions or corners
        unflattened_height = jnp.repeat(jnp.array([0, self.height // 2, self.height - 1]), 3)
        unflattened_width = jnp.array([0, self.width // 2, self.width - 1] * 3)
        self.corner_indices = jnp.ravel_multi_index((unflattened_height, unflattened_width), (self.height, self.width))

        # goal_idx = layout.get("goal_idx")
        # plate_pile_idx = layout.get("plate_pile_idx")
        # all_invalid_locs = jnp.concatenate([goal_idx, plate_pile_idx, self.corner_indices])
        # self.valid_locs = jnp.array([i for i in self.border_indices if i not in all_invalid_locs])
        self.valid_locs = jnp.setdiff1d(self.border_indices, self.corner_indices)  # what's in broder that's not in corners

        # Define the quadrants excluding the median lines, used for sampling agent locations
        quadrant_1 = [(1, 1), (1, 2), (2,1)]
        quadrant_2 = [(1, 4), (1, 5), (2, 5)]
        quadrant_3 = [(4, 1), (5, 1), (5, 2)]
        quadrant_4 = [(5, 4), (5, 5), (4, 5)]

        middle_square = [(i, j) for i in range(2, 5) for j in range(2, 5)]

        flatten_q = lambda q: [i * self.width + j for i, j in q]
        self.q1 = jnp.array(flatten_q(quadrant_1))
        self.q2 = jnp.array(flatten_q(quadrant_2))
        self.q3 = jnp.array(flatten_q(quadrant_3))
        self.q4 = jnp.array(flatten_q(quadrant_4))
        self.middle_square = jnp.array(flatten_q(middle_square))

        
        self.middle_cross_indices = jnp.array([[self.height // 2, i] for i in range(self.width)] + [[i, self.width // 2] for i in range(self.height)])
        self.flattened_middle_cross_indices = jnp.ravel_multi_index(self.middle_cross_indices.T, (self.height, self.width))
        self.potential_valid_wall_locs = jnp.concatenate([self.flattened_middle_cross_indices, self.valid_locs, self.middle_square, self.corner_indices])
        self.potential_valid_wall_locs = jnp.setdiff1d(self.potential_valid_wall_locs, self.valid_locs)  # there's no guarantee these will be solvable or walls


        self.obs_shape = (self.width, self.height, 26)

        self.agent_view_size = 5  # Hard coded. Only affects map padding -- not observations.
        self.layout = layout
        self.agents = ["agent_0", "agent_1"]
        self.single_agent = single_agent

        self.action_set = jnp.array([
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
            Actions.interact,
        ])

        self.random_reset = random_reset
        self.shuffle_inv_and_pot = shuffle_inv_and_pot
        self.max_steps = max_steps

        self.check_held_out = check_held_out

        self.held_out_goal = None
        self.held_out_wall = None
        self.held_out_pot = None
    
    def action_to_string(self, action: int) -> str:
        return Actions(action).name

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array([actions["agent_0"], actions["agent_1"]]))

        state, reward, shaped_reward = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {"agent_0": reward, "agent_1": reward}
        shaped_rewards = {"agent_0": shaped_reward[0], "agent_1": shaped_reward[1]}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards},
        )

    def reset(self,
              key: chex.PRNGKey, params={'random_reset_fn': 'reset_all'}):

        jitted_reset = lambda k: self.custom_reset(k, layout=self.layout, random_reset=False, shuffle_inv_and_pot=self.shuffle_inv_and_pot)
        jitted_reset = jax.jit(jitted_reset)

        @jax.jit
        def random_og_5(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = self.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=self.shuffle_inv_and_pot)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
            
            def sampled_0(key, sampled_num):
                return reset_sub_dict(key, make_asymm_advantages_9x9)
            def sampled_1(key, sampled_num):
                def reset_coord_ring(key, sampled_num):
                    return reset_sub_dict(key, make_coord_ring_9x9)
                return jax.lax.cond(sampled_num==1, reset_coord_ring, sampled_2, key, sampled_num)
            def sampled_2(key, sampled_num):
                def reset_counter_circuit(key, sampled_num):
                    return reset_sub_dict(key, make_counter_circuit_9x9)
                return jax.lax.cond(sampled_num==2, reset_counter_circuit, sampled_3, key, sampled_num)
            def sampled_3(key, sampled_num):
                def reset_forced_coord(key, sampled_num):
                    return reset_sub_dict(key, make_forced_coord_9x9)
                return jax.lax.cond(sampled_num==3, reset_forced_coord, sampled_4, key, sampled_num)
            def sampled_4(key, sampled_num):
                return reset_sub_dict(key, make_cramped_room_9x9)

            # sample an index from 0 to 4
            index = jax.random.randint(key, (), minval=0, maxval=5)
            sampled_reset, key = jax.lax.cond(index == 0, sampled_0, sampled_1, key, index)
            return sampled_reset

        @jax.jit
        def random_counter_circuit(key):
            def reset_sub_dict(key, fn):
                key, subkey = jax.random.split(key)
                sampled_layout_dict = fn(subkey, ik=True)
                temp_o, temp_s = self.custom_reset(key, layout=sampled_layout_dict, random_reset=False, shuffle_inv_and_pot=self.shuffle_inv_and_pot)
                key, subkey = jax.random.split(key)
                return (temp_o, temp_s), key
            
            counter_circuit_reset, key = reset_sub_dict(key, make_counter_circuit_9x9)
            return counter_circuit_reset
        
        
        def check_match(state_):  # says whether or not the observation is in the held out set
            goal_match = lambda g_pos: jnp.all(g_pos == state_.goal_pos)
            pot_match = lambda p_pos: jnp.all(p_pos == state_.pot_pos)
            wall_match = lambda w_map: jnp.all(w_map == state_.wall_map)
            some_goal_match = jax.vmap(goal_match)(self.held_out_goal)
            some_pot_match = jax.vmap(pot_match)(self.held_out_pot)
            some_wall_match = jax.vmap(wall_match)(self.held_out_wall)
            temp = jnp.stack([some_goal_match, some_pot_match, some_wall_match], axis=0) # 3 x 100
            some_match = jnp.all(temp, axis=0).any()
            return some_match

        random_reset_fn = lambda k: jax.lax.cond(params['random_reset_fn'] == 'reset_all', random_og_5, random_counter_circuit, k)
        obs, state = jax.lax.cond(self.random_reset, random_reset_fn, jitted_reset, key)
        key = jax.random.split(key)[0]
        (obs, state) = jax.lax.cond(jnp.logical_and(check_match(state), self.check_held_out), random_og_5, lambda k: (obs, state), key)
        
        return lax.stop_gradient(obs), lax.stop_gradient(state)
    
    def custom_reset(
            self,
            key: chex.PRNGKey,
            random_reset,
            shuffle_inv_and_pot,
            layout,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state based on `self.random_reset`

        If True, everything is randomized, including agent inventories and positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by `self.layout`
        """
        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")
        
        # height and width are same
        wall_boundaries = jnp.expand_dims(jnp.arange(self.width), 1)
        repeated_middle = jnp.repeat(jnp.array([self.height // 2]), self.width)
        repeated_middle = jnp.expand_dims(repeated_middle, 1)
        horizontal_wall = jnp.concatenate([wall_boundaries, repeated_middle], axis=1)
        vertical_wall = jnp.concatenate([repeated_middle, wall_boundaries], axis=1)

        # given a 7x2 list, where each row has the x,y coordinate, get the flattened index
        def _flatten_idx(entry):
            return entry[1] * w + entry[0]
        vmapped_flatten = jax.vmap(_flatten_idx)
        horizontal_wall_idx = vmapped_flatten(horizontal_wall)
        vertical_wall_idx = vmapped_flatten(vertical_wall)

        def no_additional_wall_mask(args):
            wall_idx, vertical_wall_idx, horizontal_wall_idx, to_poke, subkey = args
            occupied_mask = jnp.zeros_like(all_pos)
            occupied_mask = occupied_mask.at[wall_idx].set(1)
            return occupied_mask
        def random_middle_wall_mask(args):  # middle block with poked holes
            subkey = args[-1]
            occupied_mask = no_additional_wall_mask(args)
            random_middle_mask = jax.random.randint(subkey, self.middle_square.shape, minval=0, maxval=2, dtype=jnp.uint32)
            occupied_mask = occupied_mask.at[self.middle_square].set(random_middle_mask)
            return occupied_mask
        def vertical_wall_mask(args):
            wall_idx, vertical_wall_idx, horizontal_wall_idx, to_poke, subkey = args
            occupied_mask = jnp.zeros_like(all_pos)
            occupied_mask = occupied_mask.at[wall_idx].set(1)
            occupied_unpoked_mask = occupied_mask.at[vertical_wall_idx].set(1)
            # only poke between the first and last items  non inclusive
            poked_holes = jax.random.choice(subkey, vertical_wall_idx[1:-1], shape=(2,), replace=False)
            occupied_poked_mask = occupied_unpoked_mask.at[poked_holes].set(0)
            occupied_mask = jnp.where(to_poke, occupied_poked_mask, occupied_unpoked_mask)
            return occupied_mask
        def horizontal_wall_mask(args):
            wall_idx, vertical_wall_idx, horizontal_wall_idx, to_poke, subkey = args
            occupied_mask = jnp.zeros_like(all_pos)
            occupied_mask = occupied_mask.at[wall_idx].set(1)
            occupied_unpoked_mask = occupied_mask.at[horizontal_wall_idx].set(1)
            # only poke between the first and last items  non inclusive
            poked_holes = jax.random.choice(subkey, horizontal_wall_idx[1:-1], shape=(2,), replace=False)
            occupied_poked_mask = occupied_unpoked_mask.at[poked_holes].set(0)
            occupied_mask = jnp.where(to_poke, occupied_poked_mask, occupied_unpoked_mask)
            return occupied_mask
        
        key, subkey = jax.random.split(key)
        to_poke = jax.random.choice(subkey, jnp.array([True, False]), shape=(1,), replace=False, p=jnp.array([0.75, 0.25]))
        wall_args = (wall_idx, vertical_wall_idx, horizontal_wall_idx, to_poke, subkey)

        random_mask_id = jax.random.choice(subkey, 4, p=jnp.array([1/10, 3/10, 3/10, 3/10]))

        def mask_id_zero(args):
            wall_args, mask_id = args
            return no_additional_wall_mask(wall_args)
        def mask_id_not_zero(args):
            wall_args, mask_id = args
            def mask_id_one(args):
                wall_args, mask_id = args
                return random_middle_wall_mask(wall_args)
            def mask_id_not_one(args):
                wall_args, mask_id = args
                return jax.lax.cond(mask_id == 2, vertical_wall_mask, horizontal_wall_mask, wall_args)
            return jax.lax.cond(mask_id == 1, mask_id_one, mask_id_not_one, (wall_args, mask_id))
        random_mask = jax.lax.cond(random_mask_id == 0, mask_id_zero, mask_id_not_zero, (wall_args, random_mask_id))

        default_wall_mask = no_additional_wall_mask(wall_args)

        # create mask
        occupied_mask = jnp.where(random_reset, random_mask, default_wall_mask)


        # Reset agent position + dir
        key, subkey = jax.random.split(key)

        '''
        FREEZING AGENT IN PLACE
        '''
        initial_quadrant_choice = jax.random.choice(key, 2)
        picked_0 = lambda x: (self.q1, self.q4)
        picked_1 = lambda x: (self.q2, self.q3)
        initial_quadrant, opposite_quadrant = jax.lax.cond(initial_quadrant_choice == 0, picked_0, picked_1, 0)

        # Randomly sample agents to be in separate quadrants
        key, subkey = jax.random.split(key)
        selected_index_1 = jax.random.choice(subkey, initial_quadrant)
        key, subkey = jax.random.split(key)
        selected_index_2 = jax.random.choice(subkey, opposite_quadrant)
        sampled_indices = jnp.array([selected_index_1, selected_index_2])
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.permutation(subkey, sampled_indices)  # ensure agents can be in either quadrant


        # Replace with fixed layout if applicable. Also randomize if agent position not provided
        agent_idx = random_reset*agent_idx + (1-random_reset)*layout.get("agent_idx", agent_idx)
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose() # dim = n_agents x 2

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get() # dim = n_agents x 2

        # Keep track of empty counter space (table)
        empty_table_mask = occupied_mask

        goal_idx = layout.get("goal_idx")
        onion_pile_idx = layout.get("onion_pile_idx")
        plate_pile_idx = layout.get("plate_pile_idx")
        pot_idx = layout.get("pot_idx")
        # Need to randomly shuffle the items along the border
        key, subkey = jax.random.split(key)

        item_idxes = jax.random.permutation(subkey, self.valid_locs)
        key, subkey = jax.random.split(key)
        
        questionable_probs = empty_table_mask[self.potential_valid_wall_locs]    
        questionable_probs = questionable_probs / jnp.sum(questionable_probs)
        questionable_idxes = jax.random.choice(subkey, self.potential_valid_wall_locs, shape=(4,), replace=False, p=questionable_probs).astype(jnp.uint32)
        sample_questionable_flag = jax.random.choice(subkey, jnp.array([True, False]), p=jnp.array([0.75, 0.25]), shape=(4,), replace=True)
        
        # First sample valid items
        start = 0
        end = len(onion_pile_idx)
        temp_onion_pile_idx = item_idxes[start:end]
        additional_onion = jnp.where(sample_questionable_flag[0], questionable_idxes[0], temp_onion_pile_idx[0])
        temp_onion_pile_idx = temp_onion_pile_idx.at[-1].set(additional_onion)

        start = end
        end += len(plate_pile_idx)
        temp_plate_pile_idx = item_idxes[start:end]
        additional_plate = jnp.where(sample_questionable_flag[1], questionable_idxes[1], temp_plate_pile_idx[0])
        temp_plate_pile_idx = temp_plate_pile_idx.at[-1].set(additional_plate)
        
        start = end 
        end += len(pot_idx)
        temp_pot_idx = item_idxes[start:end]
        additional_pot = jnp.where(sample_questionable_flag[2], questionable_idxes[2], temp_pot_idx[0])
        temp_pot_idx = temp_pot_idx.at[-1].set(additional_pot)

        start = end
        end += len(goal_idx)
        temp_goal_idx = item_idxes[start:end]
        additional_goal = jnp.where(sample_questionable_flag[3], questionable_idxes[3], temp_goal_idx[0])
        temp_goal_idx = temp_goal_idx.at[-1].set(additional_goal)

        # Now check if we are doing random reset
        onion_pile_idx = random_reset*temp_onion_pile_idx + (1-random_reset)*onion_pile_idx
        plate_pile_idx = random_reset*temp_plate_pile_idx + (1-random_reset)*plate_pile_idx
        pot_idx = random_reset*temp_pot_idx + (1-random_reset)*pot_idx
        goal_idx = random_reset*temp_goal_idx + (1-random_reset)*goal_idx

        occupied_mask = occupied_mask.at[onion_pile_idx].set(1)
        occupied_mask = occupied_mask.at[plate_pile_idx].set(1)
        occupied_mask = occupied_mask.at[pot_idx].set(1)
        occupied_mask = occupied_mask.at[goal_idx].set(1)

        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24 (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * shuffle_inv_and_pot + (1-shuffle_inv_and_pot) * jnp.ones((pot_idx.shape[0])) * 23

        key, subkey = jax.random.split(key)

        # get random permutation of indices
        temp_indices = jnp.arange(self.height * self.width)
        key, subkey = jax.random.split(key)

        # find first 3 indices that are walls
        wall_probs = empty_table_mask
        wall_probs = wall_probs / jnp.sum(wall_probs)
        sampled_flattened_coords = jax.random.choice(subkey, temp_indices, shape=(3,), p=wall_probs, replace=False)
        converted_to_2d = lambda x: jnp.array([x % w, x // w], dtype=jnp.uint32)
        converted_to_2d = jax.vmap(converted_to_2d)(sampled_flattened_coords)
        key, subkey = jax.random.split(key)

        sample_item_flag = jax.random.choice(subkey, jnp.array([True, False]), p=jnp.array([0.5, 0.5]), shape=(3,), replace=True)
        def get_sample(flag, item):
            passed_valid  = flag & random_reset
            return jnp.where(passed_valid, item, jnp.array([-1,-1]))
        sampled_coords = jax.vmap(get_sample)(sample_item_flag, converted_to_2d)
        onion_pos = sampled_coords[0][None]
        plate_pos = sampled_coords[1][None]
        dish_pos = sampled_coords[2][None]

        wall_map = occupied_mask.reshape(self.height, self.width)
        wall_map = wall_map.astype(jnp.bool_)

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                          OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        

        agent_inv = shuffle_inv_and_pot * random_agent_inv + \
                    (1-shuffle_inv_and_pot) * jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)
        
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation, of size (height x width x n_layers), where n_layers = 26.
        Layers are of shape (height x width) and  are binary (0/1) except where indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions (1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either a cooking/done pot or to a soup (dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2] ### TESTING ADDING CHANNEL INDICATING WHICH AGENT IS WHICH
        padding = (state.maze_map.shape[0]-height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)    # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc   # 0/3, as long as cooking or done
        pot_cooking_time_layer = pot_status * (pot_status < POT_FULL_STATUS)                           # Timer: 19 to 0
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc                 # Ready soups, plated or not
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0, state.agent_pos[0, 1], state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1, state.agent_pos[1, 1], state.agent_pos[1, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(state.agent_inv,(1,2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers,0), agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers,0)
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers,0)

        env_layers = [
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),       # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),        # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),                     # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes
            urgency_layer,                                                          # 25
        ]

        # Agent related layers
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        dir_layer_idx = state.agent_dir_idx+jnp.array([0,4])
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx,:,:].set(agent_pos_layers)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)

        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(agent_pos_layers[1]).at[1].set(agent_pos_layers[0])
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:]).at[6:10].set(agent_direction_layers[0:4])
        bob_obs = bob_obs.at[10:].set(jnp.stack(env_layers))

        alice_obs = jnp.transpose(alice_obs, (1, 2, 0))
        bob_obs = jnp.transpose(bob_obs, (1, 2, 0))

        return {"agent_0" : alice_obs, "agent_1" : bob_obs}

    def step_agents(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, float]:

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal : jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],                     # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],                     # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[0],
            alice_pos
        )
        bob_pos = jnp.where(
            ~collision * swap_places,
            state.agent_pos[1],
            bob_pos
        )

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward, alice_shaped_reward = self.process_interact(maze_map, state.wall_map, fwd_pos, state.agent_inv, 0)
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact,
                              candidate_maze_map,
                              maze_map)
        alice_inv = jax.lax.select(alice_interact,
                              alice_inv,
                              state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.)
        alice_shaped_reward = jax.lax.select(alice_interact, alice_shaped_reward, 0.)

        candidate_maze_map, bob_inv, bob_reward, bob_shaped_reward = self.process_interact(maze_map, state.wall_map, fwd_pos, state.agent_inv, 1)
        maze_map = jax.lax.select(bob_interact,
                              candidate_maze_map,
                              maze_map)
        bob_inv = jax.lax.select(bob_interact,
                              bob_inv,
                              state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.)
        bob_shaped_reward = jax.lax.select(bob_interact, bob_shaped_reward, 0.)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status-1) + (~is_cooking) * pot_status # defaults to zero if done
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        reward = alice_reward + bob_reward

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            reward,
            (alice_shaped_reward, bob_shaped_reward)
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""
        
        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]

        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"], object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"]
        )
        # Whether the object in front is counter space that the agent can drop on.
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)

        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"], object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # give reward for placing onion in pot, and for picking up soup
        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        # Update pot status and object in inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something unpickable (e.g. pot or goal)
        successful_pickup = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # Apply inventory update
        has_picked_up_plate = successful_pickup*(new_object_in_inv == OBJECT_TO_INDEX["plate"])
        
        # number of plates in player hands < number ready/cooking/partially full pot
        num_plates_in_inv = jnp.sum(inventory == OBJECT_TO_INDEX["plate"])
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2] 
        num_notempty_pots = jnp.sum((padded_map!=POT_EMPTY_STATUS)* pot_loc_layer)
        is_dish_picku_useful = num_plates_in_inv < num_notempty_pots

        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0
        
        shaped_reward += no_plates_on_counters*has_picked_up_plate*is_dish_picku_useful*BASE_REW_SHAPING_PARAMS["PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv
        
        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float)*DELIVERY_REWARD
        return maze_map, inventory, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return spaces.Discrete(
            len(self.action_set),
            dtype=jnp.uint32
        )

    def observation_space(self, agent_id="") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps


if __name__ == "__main__":
    env = Overcooked( 
            layout = None,
            random_reset= True,
            max_steps= 256,
            single_agent= False,
            check_held_out= False,
            shuffle_inv_and_pot= True)

    from jaxmarl.viz.overcooked_jitted_visualizer import render_fn
    import imageio


    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    def render_reset(key):
        obs, state = env.reset(key)
        return render_fn(state)
    images = jax.vmap(render_reset)(keys)
    # for each image, save it as a png
    for i, image in enumerate(images):
        imageio.imwrite(f"image_{i}.png", image)
        print(f"Saved image_{i}.png")
