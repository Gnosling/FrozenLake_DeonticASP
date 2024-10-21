from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

SLIDING_PROB = 0.1

MAPS = {
    "3x3_A": [
        "SFF",
        "FHF",
        "FFG"
    ],
    "3x3_B": [
        "SFC",
        "FHC",
        "CCG"
    ],
    "4x4_A": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "4x4_B": [
        "SFFF",
        "FHFH",
        "FFFH",
        "FFFG"
    ],
    "4x4_T": [
        "SFFF",
        "FHFH",
        "FPFP",
        "FFPG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

TRAVERSER_PATHS = {
    "3x3_Aa": [8,5,2,1,0],
    "3x3_A": [6,1],
    "4x4_A": [13,9,13,9,13,9,13,9,13,9,13,9,13,9],
    "4x4_B": [13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9,13,9],
    "4x4_T": [6,2,6,2,6,2,6,2,6,2,6],
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, p: float = 0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class FrozenLakeEnv(Env):
    """
    Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H)
    by walking over the Frozen(F) lake.
    The agent may not always move in the intended direction due to the slippery nature of the frozen lake.


    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:

    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Rewards

    Reward schedule:
    - Reach goal(G): +1
    - Reach hole(H): 0
    - Reach frozen(F): 0

    ### Arguments

    ```
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc`: Used to specify custom map for frozen lake. For example,

        desc=["SFFF", "FHFH", "FFFH", "HFFG"].

        A random generated map can be specified by calling the function `generate_random_map`. For example,

        ```
        from gym.envs.toy_text.frozen_lake import generate_random_map

        gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
        ```

    `map_name`: ID to use any of the preloaded maps.

        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]

    `is_slippery`: True/False. If True will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.

        For example, if action is left and is_slippery is True, then:
        - P(move left)=1/3
        - P(move up)=1/3
        - P(move down)=1/3

    ### Version History
    * v1: Bug fixes to rewards
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        traverser_path=None,
        is_slippery=True,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.map_name = map_name
        if traverser_path in TRAVERSER_PATHS.keys():
            self.traverser_path = TRAVERSER_PATHS[traverser_path]
            self.traverser_tracker = 0
        else:
            self.traverser_path = None

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)


        def update_probability_matrix(row, col, action):
            """
            defines new state, terminated and reward of n action
            """
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            difference_to_goal = len(desc)-1-newrow + len(desc[0])-1-newcol

            terminated = bytes(newletter) in b"GH"

            # if newletter == b"G":
            #     reward = 10
            # elif newletter == b"H":
            #     reward = -10
            # else:
            #     reward = -1

            if newletter == b"G":
                reward = 1
            else:
                reward = 0

            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    # li contains (prop, newstate, rewards, terminated)
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            left_turn = (a+1)%4
                            right_turn = (a-1)%4
                            li.append((SLIDING_PROB, *update_probability_matrix(row, col, left_turn)))
                            li.append(( (1-2*SLIDING_PROB) , *update_probability_matrix(row, col, a)))
                            li.append((SLIDING_PROB, *update_probability_matrix(row, col, right_turn)))
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def get_number_of_tiles(self):
        return self.nrow * self.ncol

    def get_tile_symbol_of_state_number(self, state: int):
        row = int(state / self.nrow)
        col = state % self.ncol
        return self.desc[row][col]

    def get_goal_tile(self):
        counter = 0
        for row in range(len(self.desc)):
            for col in range(len(self.desc[row])):
                if self.desc[row][col] == b'G':
                    return counter
                counter += 1

    def get_current_traverser_position(self) -> int:
        if self.traverser_path:
            return self.traverser_path[self.traverser_tracker]
        return -1

    def get_tiles_with_holes(self):
        counter = 0
        ret = []
        for row in range(len(self.desc)):
            for col in range(len(self.desc[row])):
                if self.desc[row][col] == b'H':
                    ret.append(counter)
                counter += 1
        ret.sort()
        return ret

    def get_tiles_with_presents(self):
        self.remove_present_from_tile(self.get_current_traverser_position())
        counter = 0
        ret = []
        for row in range(len(self.desc)):
            for col in range(len(self.desc[row])):
                if self.desc[row][col] == b'P':
                    ret.append(counter)
                counter += 1
        ret.sort()
        return ret

    def remove_present_from_tile(self, tile: int):
        if tile is None or tile < 0:
            pass

        row = int(tile / self.nrow)
        col = tile % self.ncol
        if self.desc[row][col] == b'P':
            self.desc[row][col] = b'F'

    def get_layout(self):
        return self.desc, len(self.desc[0]), len(self.desc)

    def step(self, a):
        transitions = self.P[self.s][a] # transition = [prop, next_position, reward, terminated]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, current_position, reward, terminated = transitions[i]
        tile = self.get_tile_symbol_of_state_number(current_position)

        self.s = current_position
        self.lastaction = a
        presents = tuple(self.get_tiles_with_presents())
        ret_values = ( (int(current_position), -1, presents), reward, terminated, False, {"prob": prob})

        if tile in b"P":
            self.remove_present_from_tile(current_position)

        if self.traverser_path:
            # case traverser exist
            if self.traverser_tracker < len(self.traverser_path)-1:
                self.traverser_tracker += 1
            traverser_position = self.traverser_path[self.traverser_tracker]

            if self.get_tile_symbol_of_state_number(traverser_position) in b"P":
                self.remove_present_from_tile(traverser_position)
                presents = tuple(self.get_tiles_with_presents())

            ret_values = ((int(current_position), traverser_position, presents), reward, terminated, False, {"prob": prob})

            if current_position == traverser_position and tile in b"C":
                # terminate episode with reward 0
                ret_values = ((int(current_position), traverser_position, presents), 0, True, False, {"prob": prob})

        if self.render_mode == "human":
            self.render()
        return ret_values

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.traverser_tracker=0
        self.lastaction = None
        if self.map_name is None:
            desc = generate_random_map()
        else:
            desc = MAPS[self.map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape

        if self.render_mode == "human":
            self.render()

        if self.traverser_path:
            traverser_position = self.traverser_path[self.traverser_tracker]
        else:
            traverser_position = -1

        presents = tuple(self.get_tiles_with_presents())
        return (int(self.s), traverser_position, presents), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        if self.traverser_path:
            row, col = self.s // self.ncol, self.s % self.ncol
            ts = self.traverser_path[self.traverser_tracker]
            trow, tcol = ts // self.ncol, ts % self.ncol

            if trow == row and tcol == col:
                desc[row][col] = utils.colorize(desc[row][col], "yellow", highlight=True)
            else:
                desc[row][col] = utils.colorize(desc[row][col], "blue", highlight=True)
                desc[trow][tcol] = utils.colorize(desc[trow][tcol], "red", highlight=True)

        else:
            row, col = self.s // self.ncol, self.s % self.ncol
            desc[row][col] = utils.colorize(desc[row][col], "blue", highlight=True)

        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/
