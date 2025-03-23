import jax.numpy as jnp
import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict
import pdb
cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "goal_idx" : jnp.array([18, 18]),
    "plate_pile_idx" : jnp.array([16, 16]),
    "onion_pile_idx" : jnp.array([5,9]),
    "pot_idx" : jnp.array([2, 2]),
    "layout_name": "cramped_room"
}

asymm_advantages = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9,14]),
    "pot_idx" : jnp.array([22,31]),
    "layout_name": "asymm_advantages"
}
coord_ring = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,12,14,
                            15,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([7, 11]),
    "goal_idx" : jnp.array([22, 22]),
    "plate_pile_idx" : jnp.array([10, 10]),
    "onion_pile_idx" : jnp.array([15,21]),
    "pot_idx" : jnp.array([3,9]),
    "layout_name": "coord_ring"
}
forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,7,9,
                            10,12,14,
                            15,17,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([11,8]),
    "goal_idx" : jnp.array([23, 23]),
    "onion_pile_idx" : jnp.array([5,10]),
    "plate_pile_idx" : jnp.array([15, 15]),
    "pot_idx" : jnp.array([3,9]),
    "layout_name": "forced_coord"
}

# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWX
W A    W
B WWWW X
W     AW
BWWOOWWW
"""

# squeezed_room_drawn = {
#     "height" : 7,
#     "width" : 7,
#     "wall_idx": jnp.array([0,1,2,3,4,5,6,
#                             7,8,9,10,11,12,13,
#                             14,15,16,17,18,19,20,
#                             21,22,23,24,25,26,27,
#                             28,29,30,31,32,33,34,
#                             35,36,37,38,39,40,41,
#                             42,43,44,45,46,47,48]),
#     "agent_idx": jnp.array([8, 40]),
#     "goal_idx": jnp.array([1,7]),
#     "plate_pile_idx": jnp.array([5,13]),
#     "onion_pile_idx": jnp.array([35,43]),
#     "pot_idx": jnp.array([41,47]),
#     "layout_name": "squeezed_room_drawn"
# }
squeezed_room_manual = {
    "height" : 7,
    "width" : 7,
    "wall_idx": jnp.array([0,1,2,3,4,5,6,
                            7,13,
                            14,20,
                            21,27,
                            28,34,
                            35,41,
                            42,43,44,45,46,47,48]),
    "agent_idx": jnp.array([8, 40]),
    "goal_idx": jnp.array([1,7]),
    "plate_pile_idx": jnp.array([5,13]),
    "onion_pile_idx": jnp.array([35,43]),
    "pot_idx": jnp.array([41,47]),
    "layout_name": "squeezed_room_drawn"
}

squeezed_room_drawn = """
WWWWWWW
WWWWWWW
WWWWWWW
WXWWWPW
XA W  P
O  W AB
WWOWBWW
"""





# cramped_room_padded = """
# WXWWWPW
# XA    P
# W     W
# W     W
# W     W
# O    AB
# WOWWWBW
# """

# counter_circuit_padded = """
# BWPWPWX
# WA    W
# B     X
# W WWW W
# W    AW
# W     W
# WWOWOWW
# """

counter_circuit_padded = """
XWWPWWP
W  A  W
W     W
B WWW X
W     W
W  A  W
BWWOWWO
"""

harder_counter_circuit_padded = """
XWWPWWP
W  A  W
W W W W
B WWW X
W W W W
W  A  W
BWWOWWO
"""

coord_ring_padded = """
WWPWPWW
WA A  W
B WWW W
W WWW W
W WWW W
B     W
WOOWXXW
"""

forced_coord_padded = """
WWWWWWP
OA W  P
O  W  W
W  W  W
B  W AX
B  W  X
WWWWWWW
"""

asymm_advantages_padded = """
WXWWWWW
W   A P
B     O
WWWWWWW
B     X
W A   P
WOWWWWW
"""


cramped_room_padded = """
WWWPWWP
OA    O
W     W
W     W
W     W
W    AW
BBWWWXX
"""



coord_ring_padded = """
BWWWWPW
WA A  P
W WWW W
B WWW W
W WWW W
O     W
WOWXWWX
"""
# coord_ring_padded = """
# BWWWWPW
# WA A  P
# W WWW W
# W WWW W
# B WWW W
# O     W
# WOWXWWX
# """
# coord_ring_padded = """
# BWWWWPW
# WA A  P
# W WWW W
# W WWW W
# B WWW W
# O     W
# WOWWXWX
# """
# coord_ring_padded = """
# WWWXWWW
# W     W
# W WWW W
# W WXW W
# W W W W
# BA   AB
# WOPWPOW
# """

forced_coord_padded = """
WWWWWWW
OA W  P
O  W  P
W  W  W
B  W AW
B  W  W
WWWWXXW
"""

asymm_advantages_padded = """
WXWWWWW
W   A W
B     O
WWWPWPW
B     X
W A   W
WOWWWWW
"""

figure_ates = """
WWWWWWX
W     W
OAOWW W
P     X
BABWW W
P     W
WWWWWWW
"""

hallway_halting = """
WWWXWWW
W     W
W WWW W
W WXW W
W W W W
BA   AB
WOPWPOW
"""

column_control = """
WWWWWWW
W  W  X
W  W  X
P  W  O
P AW  B
W   A O
WWWWWBW
"""



def layout_grid_to_dict(grid, layout_name="counter_circuit_grid"):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    O: onion pile
    P: pot location
    ' ' (space) : empty cell
    """

    rows = grid.split('\n')

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    symbol_to_key = {"W" : "wall_idx",
                     "A" : "agent_idx",
                     "X" : "goal_idx",
                     "B" : "plate_pile_idx",
                     "O" : "onion_pile_idx",
                     "P" : "pot_idx"}

    layout_dict = {key : [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key.keys():
                # Add object
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "P"]:
                # These objects are also walls technically
                layout_dict["wall_idx"].append(idx)
            elif obj == " ":
                # Empty cell
                continue

    for key in symbol_to_key.values():
        # Transform lists to arrays
        layout_dict[key] = jnp.array(layout_dict[key])
    layout_dict["layout_name"] = layout_name

    return FrozenDict(layout_dict)


def initialize_border(height, width):
    '''
    makes a border around the grid
    returns list of lists of shape (height, width)
    '''
    grid = [[' ']*width for _ in range(height)]
    for h in range(height):
        grid[h][0] = 'W'
        grid[h][width-1] = 'W'
    for w in range(width):
        grid[0][w] = 'W'
        grid[height-1][w] = 'W'
    return grid

def initialize_items(grid, items):
    '''
    assumes grid is a list of lists of shape (height, width), with only the border covered in walls
    items is a dictionary mapping symbol --> 
        max num_items --> int
        item prob --> list
    '''
    total_to_sample = 0
    symbol_array = []
    for item_symbol in items:  # aggregate number of symbols
        max_num_item, item_prob = items[item_symbol]['max_items'], items[item_symbol]['item_prob']
        if item_prob is not None:
            assert len(item_prob) == max_num_item, f"Provided probabilities are not the same as maximum number of items for item {item_symbol}"
        num_items = np.random.choice(np.arange(1, max_num_item + 1), p=item_prob)
        total_to_sample += num_items
        symbol_array += [item_symbol] * num_items
    
    wall_idxes = get_walls(grid)
    assert total_to_sample <= len(wall_idxes), f"total to sample {total_to_sample}; num_walls {len(wall_idxes)}"
    sampled_ids = np.random.choice(len(wall_idxes), size=total_to_sample, replace=False)
    for i, list_id in enumerate(sampled_ids):  # substitute symbols
        h, w = wall_idxes[list_id]
        grid[h][w] = symbol_array[i]
    return grid

def initialize_agents(grid, one_frozen=False):
    height, width = len(grid), len(grid[0])
    if not one_frozen:
        sampled_heights = np.random.choice(np.arange(1, height-1), size=2, replace=False)
        sampled_widths = np.random.choice(np.arange(1, width-1), size=2, replace=False)
        for (h, w) in zip(sampled_heights, sampled_widths):
            grid[h][w] = 'A'
    else:
        # h, w = np.random.choice(np.arange(1, height-1)), np.random.choice(np.arange(1, width-1))
        # grid[h][w] = 'A'
        # grid[0][0] = 'A'  # stick one agent in a corner for now
        sampled_heights = np.random.choice(np.arange(1, height-1), size=2, replace=False)
        sampled_widths = np.random.choice(np.arange(1, width-1), size=2, replace=False)
        for (h, w) in zip(sampled_heights, sampled_widths):
            grid[h][w] = 'A'
    return grid

def is_wall(grid, i, j):
    '''
    Returns true if the grid location is a wall
    '''
    return grid[i][j] == 'W'


def get_walls(grid):
    '''
    returns a set of all (h,w) pairs that have a wall
    '''
    height, width = len(grid), len(grid[0])
    wall_idxes = {(h, w) for h, row in enumerate(grid) for w, cell in enumerate(row) if cell == 'W'}
    corners = {(0,0), (0, width-1), (height-1, 0), (height-1, width-1)}
    wall_idxes = wall_idxes - corners

    return list(wall_idxes)


def print_grid(grid):
    for row in grid:
        print(''.join(row))


def init_item_dict(max_num_items, item_prob=None):
    return {'max_items': max_num_items, 'item_prob': item_prob}


def sample_overcooked_grid(min_height=7, max_height=7, min_width=7, max_width=7, one_frozen=False):
    height = np.random.choice(np.arange(min_height, max_height+1))
    width = np.random.choice(np.arange(min_width, max_width+1))
    grid = initialize_border(height, width)

    if not one_frozen:
        items_to_init = {
            'P': init_item_dict(max_num_items=2, item_prob=[0.67, 0.33]),
            'X': init_item_dict(max_num_items=2, item_prob=[0.8, 0.2]),
            'O': init_item_dict(max_num_items=3, item_prob=[0.0, 0.8, 0.2]),
            'B': init_item_dict(max_num_items=2, item_prob=[0.67, 0.33]),
        }
    else:
        items_to_init = {
            'P': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
            'X': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
            'O': init_item_dict(max_num_items=3, item_prob=[0.0, 1.0, 0.0]),
            'B': init_item_dict(max_num_items=2, item_prob=[1.0, 0.0]),
        }

    grid = initialize_items(grid, items_to_init)
    grid = initialize_agents(grid, one_frozen=False)
    
    grid_string = '\n'.join([''.join(row) for row in grid])

    return grid_string

def layout_array_to_dict(grid, layout_name="array_layout", num_pots=2, num_plates=2, num_onions=2, num_goals=2, num_agents=2, num_base_walls=None):
    """Converts a jax.numpy array representation of a layout to dictionary format.
    Assumes the following encoding:
    0: free space
    1: wall
    2: agent
    3: goal
    4: plate (bowl) pile
    5: onion pile
    6: pot location
    
    Args:
        grid: jnp.array of shape (height, width) containing integers 0-5
        layout_name: string name for the layout
    """
    height, width = grid.shape
    
    # Create indices array
    idx_grid = jnp.arange(height * width).reshape(height, width)
    
    # Get indices for each object type using jnp.where
    if num_base_walls is None:
        num_base_walls = (2*height + 2*width - 4) - num_pots - num_plates - num_onions - num_goals
    else:
        num_base_walls = num_base_walls
    
        
    wall_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 1, size=num_base_walls)[0]]
    agent_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 2, size=num_agents)[0]]
    goal_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 3, size=num_goals)[0]]
    plate_pile_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 4, size=num_plates)[0]]
    onion_pile_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 5, size=num_onions)[0]]
    pot_idx = jnp.ravel(idx_grid)[jnp.where(jnp.ravel(grid) == 6, size=num_pots)[0]]
    
    # Add additional wall indices for objects that are also walls
    wall_idx = jnp.concatenate([wall_idx, goal_idx, plate_pile_idx, onion_pile_idx, pot_idx])

    corner_indices = jnp.array([
        [0, 0],
        [0, width-1],
        [height-1, 0],
        [height-1, width-1]
    ])
    
    layout_dict = {
        "height": height,
        "width": width,
        "wall_idx": wall_idx,
        "agent_idx": agent_idx,
        "goal_idx": goal_idx,
        "plate_pile_idx": plate_pile_idx,
        "onion_pile_idx": onion_pile_idx,
        "pot_idx": pot_idx
    }
    
    return FrozenDict(layout_dict)



def single_cramped_room():
    return layout_grid_to_dict(sample_overcooked_grid(one_frozen=True), layout_name="single_cramped_room")


def make_9x9_layout(rng, layout_grid, rotate=False, num_base_walls=None):
    base_layout = jnp.ones((9, 9))
    def sub_in_default_cramped_room(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        layout_height, layout_width = target_layout.shape
        def flip_zero(x):
            return jnp.flip(x, axis=0)
        def flip_one(x):
            return jnp.flip(x, axis=1)
        flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
        target_layout = jnp.where(to_flip==1, flip_func(target_layout, flip_axis), target_layout)
        updated_layout = base_layout.at[:layout_height, :layout_width].set(target_layout)
        frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
        return updated_layout, frozen_dict
    def sub_in_90_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        def get_effect(target_layout, base_layout, num_rotations, to_flip, flip_axis):
            rot_layout = jnp.rot90(target_layout)
            def flip_zero(x):
                return jnp.flip(x, axis=0)
            def flip_one(x):
                return jnp.flip(x, axis=1)
            flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
            rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
            layout_height, layout_width = rot_layout.shape
            updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
            frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
            return updated_layout, frozen_dict
        updated_layout, frozen_dict = jax.lax.cond(num_rotations == 1, get_effect, sub_in_180_degree_rotation, target_layout, base_layout, num_rotations, to_flip, flip_axis)
        return updated_layout, frozen_dict
    def sub_in_180_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        def get_effect(target_layout, base_layout, num_rotations, to_flip, flip_axis):
            rot_layout = jnp.rot90(target_layout, k=2)
            def flip_zero(x):
                return jnp.flip(x, axis=0)
            def flip_one(x):
                return jnp.flip(x, axis=1)
            flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
            rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
            layout_height, layout_width = rot_layout.shape
            updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
            frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
            return updated_layout, frozen_dict
        updated_layout, frozen_dict = jax.lax.cond(num_rotations == 2, get_effect, sub_in_270_degree_rotation, target_layout, base_layout, num_rotations, to_flip, flip_axis)
        return updated_layout, frozen_dict
    def sub_in_270_degree_rotation(target_layout, base_layout, num_rotations, to_flip, flip_axis):
        rot_layout = jnp.rot90(target_layout, k=3)
        def flip_zero(x):
            return jnp.flip(x, axis=0)
        def flip_one(x):
            return jnp.flip(x, axis=1)
        flip_func = lambda x, y: jnp.where(y==0, flip_zero(x), flip_one(x))
        rot_layout = jnp.where(to_flip==1, flip_func(rot_layout, flip_axis), rot_layout)
        layout_height, layout_width = rot_layout.shape
        updated_layout = base_layout.at[:layout_height, :layout_width].set(rot_layout)
        frozen_dict = layout_array_to_dict(updated_layout, num_base_walls=num_base_walls)
        return updated_layout, frozen_dict

    num_rotations = jax.random.randint(rng, (), 0, 4)
    num_rotations = jnp.where(rotate, num_rotations, 0)
    rng, rng_sub = jax.random.split(rng)
    to_flip = jax.random.randint(rng_sub, (), 0, 2)
    to_flip = jnp.where(rotate, to_flip, 0)
    rng, rng_sub = jax.random.split(rng)
    flip_axis = jax.random.randint(rng_sub, (), 0, 2)
    flip_axis = jnp.where(rotate, flip_axis, 0)

    updated_layout, layout_dict = jax.lax.cond(num_rotations == 0, sub_in_default_cramped_room, sub_in_90_degree_rotation, layout_grid, base_layout, num_rotations, to_flip, flip_axis)
    return layout_dict

@jax.jit
def make_cramped_room_9x9(rng, ik=False, num_default_walls=67):
    cramped_room_array = jnp.array([
        [6, 1, 6, 1, 1],
        [5, 2, 0, 2, 5],
        [1, 0, 0, 0, 1],
        [4, 4, 1, 3, 3],
    ])


    def default_cramped_room(rng, layout=cramped_room_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_cramped_room(rng, layout=cramped_room_array):
        height, width = layout.shape
        all_walls = jnp.array([0,1,2,3,4,5,9,10,14,15,16,17,18,19])
        need_one = jnp.array([1,2,3,5,9,10,14,16,17,18])
        # get a random permutation of need_one, and take the first 4
        need_one_permutation = jax.random.permutation(rng, need_one)[:4]
        rng, rng_sub = jax.random.split(rng)

        # Create mask where 1 indicates wall not in need_one_permutation
        wall_mask = jnp.ones(len(all_walls))
        sorted_all_walls = jnp.sort(all_walls)
        wall_mask = wall_mask.at[jnp.searchsorted(sorted_all_walls, need_one_permutation)].set(0)
        wall_probs = wall_mask.astype(float) / jnp.sum(wall_mask)
        # sample 4 walls from sorted_all_walls
        additional = jax.random.choice(rng_sub, sorted_all_walls, shape=(4,), replace=False, p=wall_probs)
        rng, rng_sub = jax.random.split(rng)

        # sample agent positions
        valid_agent_positions = jnp.array([6,7,8,11,12,13])
        agent_idx = jax.random.choice(rng_sub, valid_agent_positions, shape=(2,), replace=False)

        item_indices = jnp.concatenate([need_one_permutation, additional, agent_idx])
        # convert to 2d coordinates
        x_coords, y_coords = jnp.unravel_index(item_indices, (height, width))
        stacked_coords = jnp.stack([x_coords, y_coords], axis=1)

        plate_idx = jnp.array([stacked_coords[0], stacked_coords[4]])
        onion_idx = jnp.array([stacked_coords[1], stacked_coords[5]])
        pot_idx = jnp.array([stacked_coords[2], stacked_coords[6]])
        goal_idx = jnp.array([stacked_coords[3], stacked_coords[7]])
        agent_idx = jnp.array([stacked_coords[8], stacked_coords[9]])

        def update_map(layout, item_indices, value):
            def scan_body(carry, idx):
                layout, item_indices, value = carry
                layout = layout.at[item_indices[idx][0], item_indices[idx][1]].set(value)
                return (layout, item_indices, value), idx
            carry, _ = jax.lax.scan(scan_body, (layout, item_indices, value), jnp.arange(len(item_indices)))
            return carry[0]
        modified_layout = jnp.zeros_like(layout)
        # create the basic outline of the ring
        wall_coords_x, wall_coords_y = jnp.unravel_index(all_walls, (height, width))
        wall_coords = jnp.stack([wall_coords_x, wall_coords_y], axis=1)
        modified_layout = update_map(modified_layout, wall_coords, 1)

        modified_layout = update_map(modified_layout, plate_idx, 4)
        modified_layout = update_map(modified_layout, onion_idx, 5)
        modified_layout = update_map(modified_layout, pot_idx, 6)
        modified_layout = update_map(modified_layout, goal_idx, 3)
        modified_layout = update_map(modified_layout, agent_idx, 2)

        rng, rng_sub = jax.random.split(rng)

        return make_9x9_layout(rng, modified_layout, rotate=True, num_base_walls=num_default_walls)


    return jax.lax.cond(ik, ik_cramped_room, default_cramped_room, rng, cramped_room_array)

def calc_num_walls(layout):
    num_walls =jnp.where(layout == 1, 1, 0).sum()
    return 81 - (layout.shape[0] * layout.shape[1] - num_walls)

@jax.jit
def make_asymm_advantages_9x9(rng, ik=False, num_default_walls=59):
    # 14 walls by default means num of walls in 9x9 is 81 - (36-14) = 51
    asymm_advantages_array = jnp.array([
        [5, 0, 1, 3, 1, 5, 1, 0, 3],
        [1, 0, 2, 0, 6, 0, 2, 0, 1],
        [1, 0, 0, 0, 6, 0, 0, 0, 1],
        [1, 1, 1, 4, 1, 4, 1, 1, 1],
    ])

    height, width = asymm_advantages_array.shape

    def default_asymm_advantages(rng, layout=asymm_advantages_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_asymm_advantages(rng, layout=asymm_advantages_array):
        height, width = layout.shape
        all_walls = jnp.array([0,1,2,3,4,5,6,7,8,9,13,17,18,22,26,27,28,29,30,31,32,33,34,35,])
        need_one = jnp.array([1,2,3,4,5,6,7,9,17,18,26,28,29,30,31,32,33,34,13,22])
        # get a random permutation of need_one, and take the first 4
        need_one_permutation = jax.random.permutation(rng, need_one)[:4]
        rng, rng_sub = jax.random.split(rng)

        # Create mask where 1 indicates wall not in need_one_permutation
        wall_mask = jnp.ones(len(all_walls))
        sorted_all_walls = jnp.sort(all_walls)
        wall_mask = wall_mask.at[jnp.searchsorted(sorted_all_walls, need_one_permutation)].set(0)
        wall_probs = wall_mask.astype(float) / jnp.sum(wall_mask)
        # sample 6 walls from sorted_all_walls
        additional = jax.random.choice(rng_sub, sorted_all_walls, shape=(6,), replace=False, p=wall_probs)
        rng, rng_sub = jax.random.split(rng)

        # sample agent positions
        valid_agent_0_positions = jnp.array([10, 11, 12, 19, 20, 21])
        valid_agent_1_positions = jnp.array([14,15,16,23,24,25])
        agent_0_idx = jax.random.choice(rng_sub, valid_agent_0_positions, shape=(1,), replace=False)
        rng, rng_sub = jax.random.split(rng)
        agent_1_idx = jax.random.choice(rng_sub, valid_agent_1_positions, shape=(1,), replace=False)
        agent_idx = jnp.concatenate([agent_0_idx, agent_1_idx])

        item_indices = jnp.concatenate([need_one_permutation, additional, agent_idx])
        # convert to 2d coordinates
        x_coords, y_coords = jnp.unravel_index(item_indices, (height, width))
        stacked_coords = jnp.stack([x_coords, y_coords], axis=1)

        plate_idx = jnp.array([stacked_coords[0], stacked_coords[4]])
        onion_idx = jnp.array([stacked_coords[1], stacked_coords[5]])
        pot_idx = jnp.array([stacked_coords[2], stacked_coords[6]])
        goal_idx = jnp.array([stacked_coords[3], stacked_coords[7]])
        wall_holes = jnp.array([stacked_coords[8], stacked_coords[9]])
        agent_idx = jnp.array([stacked_coords[10], stacked_coords[11]])


        def update_map(layout, item_indices, value):
            def scan_body(carry, idx):
                layout, item_indices, value = carry
                layout = layout.at[item_indices[idx][0], item_indices[idx][1]].set(value)
                return (layout, item_indices, value), idx
            carry, _ = jax.lax.scan(scan_body, (layout, item_indices, value), jnp.arange(len(item_indices)))
            return carry[0]
        modified_layout = jnp.zeros_like(layout)
        # create the basic outline of the ring
        wall_coords_x, wall_coords_y = jnp.unravel_index(all_walls, (height, width))
        wall_coords = jnp.stack([wall_coords_x, wall_coords_y], axis=1)
        modified_layout = update_map(modified_layout, wall_coords, 1)


        modified_layout = update_map(modified_layout, plate_idx, 4)
        modified_layout = update_map(modified_layout, onion_idx, 5)
        modified_layout = update_map(modified_layout, pot_idx, 6)
        modified_layout = update_map(modified_layout, goal_idx, 3)
        modified_layout = update_map(modified_layout, wall_holes, 0)  # free space for agents to go to
        modified_layout = update_map(modified_layout, agent_idx, 2)

        rng, rng_sub = jax.random.split(rng)
        return make_9x9_layout(rng, modified_layout, rotate=True, num_base_walls=num_default_walls)
    
    return jax.lax.cond(ik, ik_asymm_advantages, default_asymm_advantages, rng, asymm_advantages_array)

@jax.jit
def make_coord_ring_9x9(rng, ik=False, num_default_walls=65):
    coord_ring_array = jnp.array([
        [4,1,1,6,1],
        [1,0,2,0,6],
        [4,0,1,0,1],
        [5,0,2,0,1],
        [1,5,3,1,3],
    ])

    def default_coord_ring(rng, layout=coord_ring_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_coord_ring(rng, layout=coord_ring_array):
        height, width = layout.shape
        all_walls = jnp.array([0,1,2,3,4,5,9,10,12,14,15,19,20,21,22,23,24])
        need_one = jnp.array([1,2,3,5,9,10,14,15,19,21,22,23,12])
        # get a random permutation of need_one, and take the first 4
        need_one_permutation = jax.random.permutation(rng, need_one)[:4]
        rng, rng_sub = jax.random.split(rng)

        # Create mask where 1 indicates wall not in need_one_permutation
        wall_mask = jnp.ones(len(all_walls))
        sorted_all_walls = jnp.sort(all_walls)
        wall_mask = wall_mask.at[jnp.searchsorted(sorted_all_walls, need_one_permutation)].set(0)
        wall_probs = wall_mask.astype(float) / jnp.sum(wall_mask)
        # sample 4 walls from sorted_all_walls
        additional = jax.random.choice(rng_sub, sorted_all_walls, shape=(4,), replace=False, p=wall_probs)
        rng, rng_sub = jax.random.split(rng)

        # sample agent positions
        valid_agent_positions = jnp.array([6,7,8,11,13,16,17,18])
        agent_idx = jax.random.choice(rng_sub, valid_agent_positions, shape=(2,), replace=False)

        item_indices = jnp.concatenate([need_one_permutation, additional, agent_idx])
        # convert to 2d coordinates
        x_coords, y_coords = jnp.unravel_index(item_indices, (height, width))
        stacked_coords = jnp.stack([x_coords, y_coords], axis=1)

        plate_idx = jnp.array([stacked_coords[0], stacked_coords[4]])
        onion_idx = jnp.array([stacked_coords[1], stacked_coords[5]])
        pot_idx = jnp.array([stacked_coords[2], stacked_coords[6]])
        goal_idx = jnp.array([stacked_coords[3], stacked_coords[7]])
        agent_idx = jnp.array([stacked_coords[8], stacked_coords[9]])

        def update_map(layout, item_indices, value):
            def scan_body(carry, idx):
                layout, item_indices, value = carry
                layout = layout.at[item_indices[idx][0], item_indices[idx][1]].set(value)
                return (layout, item_indices, value), idx
            carry, _ = jax.lax.scan(scan_body, (layout, item_indices, value), jnp.arange(len(item_indices)))
            return carry[0]
        modified_layout = jnp.zeros_like(layout)
        # create the basic outline of the ring
        wall_coords_x, wall_coords_y = jnp.unravel_index(all_walls, (height, width))
        wall_coords = jnp.stack([wall_coords_x, wall_coords_y], axis=1)
        modified_layout = update_map(modified_layout, wall_coords, 1)

        modified_layout = update_map(modified_layout, plate_idx, 4)
        modified_layout = update_map(modified_layout, onion_idx, 5)
        modified_layout = update_map(modified_layout, pot_idx, 6)
        modified_layout = update_map(modified_layout, goal_idx, 3)
        modified_layout = update_map(modified_layout, agent_idx, 2)

        rng, rng_sub = jax.random.split(rng)
        return make_9x9_layout(rng, modified_layout, rotate=True, num_base_walls=num_default_walls)

    return jax.lax.cond(ik, ik_coord_ring, default_coord_ring, rng, coord_ring_array)

@jax.jit
def make_forced_coord_9x9(rng, ik=False, num_default_walls=67):
    forced_coord_array = jnp.array([
        [1,1,1,6,1],
        [5,0,1,0,6],
        [5,2,1,2,1],
        [4,0,1,0,1],
        [4,1,1,3,3],
    ])

    def default_forced_coord(rng, layout=forced_coord_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_forced_coord(rng, layout=forced_coord_array):
        height, width = layout.shape
        all_walls = jnp.array([0,1,2,3,4,5,7,9,10,12,14,15,17,19,20,21,22,23,24])
        need_one = jnp.array([1,5,10,15,21,17,12,7,3,9,14,19,23])
        # get a random permutation of need_one, and take the first 4
        need_one_permutation = jax.random.permutation(rng, need_one)[:4]
        rng, rng_sub = jax.random.split(rng)

        # Create mask where 1 indicates wall not in need_one_permutation
        wall_mask = jnp.ones(len(all_walls))
        sorted_all_walls = jnp.sort(all_walls)
        wall_mask = wall_mask.at[jnp.searchsorted(sorted_all_walls, need_one_permutation)].set(0)
        wall_probs = wall_mask.astype(float) / jnp.sum(wall_mask)
        # sample 4 walls from sorted_all_walls
        additional = jax.random.choice(rng_sub, sorted_all_walls, shape=(4,), replace=False, p=wall_probs)
        rng, rng_sub = jax.random.split(rng)

        # sample agent positions
        valid_agent_0_positions = jnp.array([6,11,16])
        valid_agent_1_positions = jnp.array([8,13,18])
        agent_0_idx = jax.random.choice(rng_sub, valid_agent_0_positions, shape=(1,), replace=False)
        rng, rng_sub = jax.random.split(rng)
        agent_1_idx = jax.random.choice(rng_sub, valid_agent_1_positions, shape=(1,), replace=False)
        agent_idx = jnp.concatenate([agent_0_idx, agent_1_idx])
        
        item_indices = jnp.concatenate([need_one_permutation, additional, agent_idx])
        # convert to 2d coordinates
        x_coords, y_coords = jnp.unravel_index(item_indices, (height, width))
        stacked_coords = jnp.stack([x_coords, y_coords], axis=1)

        plate_idx = jnp.array([stacked_coords[0], stacked_coords[4]])
        onion_idx = jnp.array([stacked_coords[1], stacked_coords[5]])
        pot_idx = jnp.array([stacked_coords[2], stacked_coords[6]])
        goal_idx = jnp.array([stacked_coords[3], stacked_coords[7]])
        agent_idx = jnp.array([stacked_coords[8], stacked_coords[9]])

        def update_map(layout, item_indices, value):
            def scan_body(carry, idx):
                layout, item_indices, value = carry
                layout = layout.at[item_indices[idx][0], item_indices[idx][1]].set(value)
                return (layout, item_indices, value), idx
            carry, _ = jax.lax.scan(scan_body, (layout, item_indices, value), jnp.arange(len(item_indices)))
            return carry[0]
        modified_layout = jnp.zeros_like(layout)
        # create the basic outline of the ring
        wall_coords_x, wall_coords_y = jnp.unravel_index(all_walls, (height, width))
        wall_coords = jnp.stack([wall_coords_x, wall_coords_y], axis=1)
        modified_layout = update_map(modified_layout, wall_coords, 1)

        modified_layout = update_map(modified_layout, plate_idx, 4)
        modified_layout = update_map(modified_layout, onion_idx, 5)
        modified_layout = update_map(modified_layout, pot_idx, 6)
        modified_layout = update_map(modified_layout, goal_idx, 3)
        modified_layout = update_map(modified_layout, agent_idx, 2)

        rng, rng_sub = jax.random.split(rng)
        return make_9x9_layout(rng, modified_layout, rotate=True, num_base_walls=num_default_walls)

    return jax.lax.cond(ik, ik_forced_coord, default_forced_coord, rng, forced_coord_array)

@jax.jit
def make_counter_circuit_9x9(rng, ik=False, num_default_walls=59):
    counter_circuit_array = jnp.array([
        [4,1,1,6,6,1,1,3],
        [1,0,0,0,0,0,0,1],
        [4,2,1,1,1,1,2,3],
        [1,0,0,0,0,0,0,1],
        [1,1,1,5,5,1,1,1],
    ])

    def default_counter_circuit(rng, layout=counter_circuit_array):
        return make_9x9_layout(rng, layout, rotate=False, num_base_walls=num_default_walls)

    def ik_counter_circuit(rng, layout=counter_circuit_array):
        height, width = layout.shape
        all_walls = jnp.array([0,1,2,3,4,5,6,7,8,15,16,18,19,20,21,23,24,31,32,33,34,35,36,37,38,39])
        need_one = jnp.array([1,2,3,4,5,6,8,15,16,18,19,20,21,23,24,31,33,34,35,36,37,38])
        # get a random permutation of need_one, and take the first 4
        need_one_permutation = jax.random.permutation(rng, need_one)[:4]
        rng, rng_sub = jax.random.split(rng)

        # Create mask where 1 indicates wall not in need_one_permutation
        wall_mask = jnp.ones(len(all_walls))
        sorted_all_walls = jnp.sort(all_walls)
        wall_mask = wall_mask.at[jnp.searchsorted(sorted_all_walls, need_one_permutation)].set(0)
        wall_probs = wall_mask.astype(float) / jnp.sum(wall_mask)
        # sample 4 walls from sorted_all_walls
        additional = jax.random.choice(rng_sub, sorted_all_walls, shape=(4,), replace=False, p=wall_probs)
        rng, rng_sub = jax.random.split(rng)

        # sample agent positions
        valid_agent_positions = jnp.array([9,10,11,12,13,14,17,22,25,26,27,28,29,30])
        agent_idx = jax.random.choice(rng_sub, valid_agent_positions, shape=(2,), replace=False)

        item_indices = jnp.concatenate([need_one_permutation, additional, agent_idx])
        # convert to 2d coordinates
        x_coords, y_coords = jnp.unravel_index(item_indices, (height, width))
        stacked_coords = jnp.stack([x_coords, y_coords], axis=1)

        plate_idx = jnp.array([stacked_coords[0], stacked_coords[4]])
        onion_idx = jnp.array([stacked_coords[1], stacked_coords[5]])
        pot_idx = jnp.array([stacked_coords[2], stacked_coords[6]])
        goal_idx = jnp.array([stacked_coords[3], stacked_coords[7]])
        agent_idx = jnp.array([stacked_coords[8], stacked_coords[9]])

        def update_map(layout, item_indices, value):
            def scan_body(carry, idx):
                layout, item_indices, value = carry
                layout = layout.at[item_indices[idx][0], item_indices[idx][1]].set(value)
                return (layout, item_indices, value), idx
            carry, _ = jax.lax.scan(scan_body, (layout, item_indices, value), jnp.arange(len(item_indices)))
            return carry[0]
        modified_layout = jnp.zeros_like(layout)
        # create the basic outline of the ring
        wall_coords_x, wall_coords_y = jnp.unravel_index(all_walls, (height, width))
        wall_coords = jnp.stack([wall_coords_x, wall_coords_y], axis=1)
        modified_layout = update_map(modified_layout, wall_coords, 1)

        modified_layout = update_map(modified_layout, plate_idx, 4)
        modified_layout = update_map(modified_layout, onion_idx, 5)
        modified_layout = update_map(modified_layout, pot_idx, 6)
        modified_layout = update_map(modified_layout, goal_idx, 3)
        modified_layout = update_map(modified_layout, agent_idx, 2)

        rng, rng_sub = jax.random.split(rng)
        return make_9x9_layout(rng, modified_layout, rotate=True, num_base_walls=num_default_walls)

    return jax.lax.cond(ik, ik_counter_circuit, default_counter_circuit, rng, counter_circuit_array)


overcooked_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "squeezed_room_drawn" : layout_grid_to_dict(squeezed_room_drawn),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    "forced_coord" : FrozenDict(forced_coord),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid),
    "cramped_room_padded" : layout_grid_to_dict(cramped_room_padded),
    "counter_circuit_padded" : layout_grid_to_dict(counter_circuit_padded),
    "forced_coord_padded" : layout_grid_to_dict(forced_coord_padded),
    "asymm_advantages_padded" : layout_grid_to_dict(asymm_advantages_padded),
    "coord_ring_padded" : layout_grid_to_dict(coord_ring_padded),
    "figure_ates" : layout_grid_to_dict(figure_ates),
    "hallway_halting" : layout_grid_to_dict(hallway_halting),
    "column_control" : layout_grid_to_dict(column_control),
    'harder_counter_circuit': layout_grid_to_dict(harder_counter_circuit_padded),
    'cramped_room_9': make_cramped_room_9x9(jax.random.PRNGKey(0), ik=False),
    'asymm_advantages_9': make_asymm_advantages_9x9(jax.random.PRNGKey(0), ik=False),
    'coord_ring_9': make_coord_ring_9x9(jax.random.PRNGKey(0), ik=False),
    'counter_circuit_9': make_counter_circuit_9x9(jax.random.PRNGKey(0), ik=False),
    'forced_coord_9': make_forced_coord_9x9(jax.random.PRNGKey(0), ik=False),
}


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    layout_dict = make_counter_circuit_9x9(rng, ik=True)
    print(layout_dict)

    # coord_ring_array = jnp.array([
    #     [4,1,1,6,1],
    #     [1,0,2,0,6],
    #     [4,0,1,0,1],
    #     [5,0,2,0,1],
    #     [1,5,3,1,3],
    # ])
    # new_layout, new_layout_dict=make_9x9_layout(rng, coord_ring_array, rotate=False)
    # print(new_layout_dict)
    # print(new_layout)