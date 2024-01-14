import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
import logging
from datetime import datetime

np.set_printoptions(edgeitems=30, linewidth=1000000)


def arr_to_tuple(arr: np.array):
    return [tuple(l) for l in arr.tolist()]


class Catan:
    # resource_card = {
    #     # name, count, tile_count
    #     1: ["brick", 19, 3],
    #     2: ["lumber", 19, 4],
    #     3: ["ore", 19, 3],
    #     4: ["grain", 19, 4],
    #     5: ["wool", 19, 4],
    #     6: ["desert", 0, 1],
    # }

    def __init__(self, seed: int, player_type: list, player_count: int, mode=str):
        self.seed = seed
        self.mode = mode
        self.turn = 0
        self.knight_card = {
            "knight": 14,
            "victory_point": 5,
            "road_building": 2,
            "year_of_plenty": 2,
            "monopoly": 2,
        }
        self.max_properties = {
            "settlements": 5,
            "cities": 4,
            "roads": 15,
        }
        self.resource_card = {
            # name, count, tile_count
            1: ["brick", 19],
            2: ["lumber", 19],
            3: ["ore", 19],
            4: ["grain", 19],
            5: ["wool", 19],
            6: ["desert", 0],
        }
        self.resource_cards = {
            1: 3,  # brick
            2: 4,  # lumber
            3: 3,  # ore
            4: 4,  # grain
            5: 4,  # wool
            6: 1,  # desert
        }

        self.center_coords = [
            # fmt: off
                (5, 8),(5, 12),(5, 16),
                (9, 6),(9, 10),(9, 14),(9, 18),
                (13, 4),(13, 8),(13, 12),(13, 16),(13, 20),
                (17, 6),(17, 10),(17, 14),(17, 18),
                (21, 8),(21, 12),(21, 16)
            # fmt: on
            # # fmt: off
            #     (3, 6),(3, 10),(3, 14),
            #     (7, 4),(7, 8),(7, 12),(7, 16),
            #     (11, 2),(11, 6),(11, 10),(11, 14),(11, 18),
            #     (15, 4),(15, 8),(15, 12),(15, 16),
            #     (19, 6),(19, 10),(19, 14)
            # # fmt: on
        ]

        # self.building_spots = [
        #     (center[0] - 1, center[1]) for center in self.center_coords
        # ]

        self.board = self.generate_board()
        self.game_over = False
        self.winner = None
        self.all_settlement_spots = arr_to_tuple(np.argwhere(self.board == -1))
        self.all_road_spots = arr_to_tuple(np.argwhere(self.board == -2))
        self.all_trades = [(x, y) for x in range(1, 6) for y in range(1, 6) if x != y]

        self.base_action_space = self.get_all_action_space()

        self.players = self.generate_players(
            player_count=player_count, player_type=player_type, mode=mode
        )
        self.player_tags = list(self.players.keys()) + [
            x - 10 for x in self.players.keys()
        ]

    def get_all_action_space(self) -> list[int, tuple]:
        # 0, 0 :: Dim = 1
        # 1, {list of all settlements} :: Dim = 2*(3+4+4+5+5+6) = 54
        # 2, {list of all road} :: Dim = 72
        # 3, {list of all cities} :: Dim = 2*(3+4+4+5+5+6) = 54
        # 4, {list of all trades} :: Dim = (4 * 5) = 20
        # Total dim = 201

        action_space = {
            0: [None],  # nothing
            1: self.all_road_spots,  # road
            2: self.all_settlement_spots,  # settlement
            3: self.all_settlement_spots,  # city
            4: self.all_trades,  # trade
        }
        flatten_action_space = [
            (action_index, potential_action)
            for action_index, potential_actions in action_space.items()
            for potential_action in potential_actions
        ]

        return flatten_action_space

    def generate_players(self, player_count: int, player_type: str, mode: str):
        players = {}
        for i in range(player_count):
            player = Catan.Player(
                self, tag=i - 9, player_type=player_type[i], mode=mode
            )
            players[i - 9] = player
        return players

    def print_board(self, debug=True):
        board = self.board.copy()
        board = board[2 : board.shape[0] - 2, 2 : board.shape[1] - 2]
        zero_tmp = 97
        board[board == 0] = zero_tmp
        board[board == 50] = zero_tmp
        for i in range(12, 22 + 1):
            board[board == i] = i - 10

        board = np.insert(board, 0, np.arange(2, board.shape[1] + 2), axis=0)
        board = np.insert(board, 0, np.arange(1, board.shape[0] + 1), axis=1)

        board_string = np.array2string(board)
        board_string = board_string.replace(f"{zero_tmp}", "  ")
        logging.debug(board_string) if debug else logging.info(board_string)

    def board_turn(self):
        """
        roll dice and deliver resources
        """
        roll = self.roll()
        if roll == 17:
            # need to cut resources by half
            pass
        else:
            logging.debug(f"Rolled: {roll - 10}")
            self.deliver_resource(roll)

    def roll(self):
        return random.randint(1, 6) + random.randint(1, 6) + 10

    def deliver_resource(self, roll):
        resource_hit = np.argwhere(self.board == roll)

        for y, x in resource_hit:
            list_of_potential_owners = [
                [y - 4, x],
                [y + 2, x],
                [y, x + 2],
                [y, x - 2],
                [y - 2, x + 2],
                [y - 2, x - 2],
            ]
            resource_type = self.board[y - 2, x]
            logging.debug(f"Resource type: {resource_type}")

            if resource_type == 6:
                continue
            else:
                for potential_y, potential_x in list_of_potential_owners:
                    owner_tag = self.board[potential_y, potential_x]

                    if owner_tag != -1:
                        resource_num = 1 if owner_tag >= -10 else 2
                        owner_tag = owner_tag if owner_tag >= -10 else owner_tag + 10
                        self.add_resource(owner_tag, resource_type, resource_num)

    def add_resource(self, owner_tag, resource_type, resource_num: int):
        self.players[owner_tag].resources[resource_type] += resource_num
        logging.debug(
            f"Player {self.players[owner_tag].tag} received {resource_num} of {resource_type}({self.resource_card[resource_type][0]})"
        )

    def game_start(self):
        # Clockwise order
        for player_tag, player in self.players.items():
            player.player_start()
        # Anticlockwise order
        for player_tag, player in reversed(self.players.items()):
            player.player_start()
        logging.info("Game started!")

    def generate_board(self):
        resource_list = [
            key for key, value in self.resource_cards.items() for x in range(value)
        ]

        # Add 10 to each number to avoid clashing with resource numbers
        number_tokens = [
            # fmt: off
            12,  13, 13,  14, 14,  15,  15,  16,  16,
            18, 18,  19, 19,  20,  20,  21,  21,  22, 0,
            # fmt: on
        ]

        def pick_and_pop(__list: list):
            picked = 0
            value = __list[picked]
            __list.pop(picked)
            return value

        arr = np.zeros((10, 11))
        """
        Grid of 5 hexagons CENTER = resource number, CENTER+1 = resource type
        21 x 23

        """
        arr = np.array(
            [
                # The extra padding is to make sure the validation works for delivering resources.
                # fmt: off
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0],
            [0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0],
            [0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, -2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # fmt: on
            ]
        )

        # Shuffle based on seed
        random.Random(self.seed).shuffle(number_tokens)
        random.Random(self.seed).shuffle(resource_list)

        for y, x in self.center_coords:
            number = pick_and_pop(number_tokens)
            resource = pick_and_pop(resource_list)
            arr[y, x] = 50  # Knight reference
            arr[y - 1, x] = resource
            arr[y + 1, x] = number

        return arr

    class Player:
        def __init__(self, catan, tag: int, player_type: str, mode: str) -> None:
            self.catan = catan
            self.player_type = player_type
            self.tag = tag
            self.mode = mode
            self.resources = {
                # name, count, Start with 2 settlement + 2 roads
                1: 0,  # brick
                2: 0,  # lumber
                3: 0,  # ore
                4: 0,  # grain
                5: 0,  # wool
            }
            self.knight_cards = {
                "knight": 0,
                "victory_point": 0,
                "road_building": 0,
                "year_of_plenty": 0,
                "monopoly": 0,
            }
            self.settlements = []
            self.roads = []
            self.cities = []
            self.action_space = None
            self.potential_settlement = []
            self.potential_road = []
            self.potential_trade = []
            self.points = 0
            (
                self.x_s,
                self.z1_s,
                self.a1_s,
                self.z2_s,
                self.a2_s,
                self.z3_s,
                self.a3_s,
                self.y_s,
                self.r_s,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            self.actions_taken = []
            self.reward_sum = 0

        def recalc_points(self):
            self.longest_road = self.calculate_longest_road()
            self.points = (
                len(self.settlements)
                + 2 * len(self.cities)
                + self.knight_cards["victory_point"]
            )
            logging.debug(f"Player {self.tag} has {self.points} points")
            if self.points >= 10:
                self.r_s[-1] += self.reward_matrix("win")

                # give other players a negative reward
                for tag, player in catan.players.items():
                    if tag != self.tag:
                        player.r_s[-1] += self.reward_matrix("loss")

                logging.info(f"Player {self.tag} wins!")

                self.catan.game_over = True
                self.catan.winner = self.tag

        def calculate_longest_road(self):
            max_length_all = 0
            for road in self.roads:
                max_length_all = max(
                    self.depth_search_longest_road(road, existing_roads=[road]),
                    max_length_all,
                )

            return max_length_all

        def depth_search_longest_road(self, road, existing_roads: list):
            next_roads = [
                r
                for r in self.roads
                if r not in existing_roads
                and abs(r[0] - road[0]) + abs(r[1] - road[1]) <= 3
            ]
            if len(next_roads) == 0:
                return len(existing_roads)

            for next_road in next_roads:
                existing_roads.append(next_road)
                return self.depth_search_longest_road(road, existing_roads)

        def get_action_space(self, start=None, attributes=None):
            """
            return a list of actions that the player can take
            """
            action_space_list = []
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                # only calculate potential actions if debug mode is on
                self.potential_settlement = self.get_potential_settlement()
                self.potential_road = self.get_potential_road()
                self.potential_trade = self.get_potential_trade()

            if start == "settlement":
                self.potential_settlement = self.get_potential_settlement()
                for settlement in self.potential_settlement:
                    action_space_list.append((2, settlement))
            elif start == "road":
                self.potential_road = self.get_potential_road(attributes)
                for road in self.potential_road:
                    action_space_list.append((1, road))
            else:
                # leavingdevelopment card later
                action_space_list.append((0, None))

                if (self.resources[1] >= 1 and self.resources[2] >= 1) and len(
                    self.roads
                ) < catan.max_properties["roads"]:
                    self.potential_road = self.get_potential_road()
                    for road in self.potential_road:
                        action_space_list.append((1, road))

                if (
                    self.resources[1] >= 1
                    and self.resources[2] >= 1
                    and self.resources[4] >= 1
                    and self.resources[5] >= 1
                ) and len(self.settlements) < catan.max_properties["settlements"]:
                    self.potential_settlement = self.get_potential_settlement()
                    for settlement in self.potential_settlement:
                        action_space_list.append((2, settlement))

                if (self.resources[3] >= 3 and self.resources[4] >= 2) and len(
                    self.cities
                ) < catan.max_properties["cities"]:
                    self.potential_city = self.get_potential_city()
                    for city in self.potential_city:
                        action_space_list.append((3, city))

                if (
                    self.resources[1] >= 4
                    or self.resources[2] >= 4
                    or self.resources[3] >= 4
                    or self.resources[4] >= 4
                    or self.resources[5] >= 4
                ):
                    self.potential_trade = self.get_potential_trade()
                    for trade in self.potential_trade:
                        action_space_list.append((4, trade))

            action_space = self.action_space_list_to_action_arr(action_space_list)
            logging.debug(action_space)
            return action_space

        def action_space_list_to_action_arr(self, action_space: list) -> np.array:
            out_action_space = np.zeros(len(self.catan.base_action_space))
            for idx, base_action in enumerate(self.catan.base_action_space):
                if base_action in action_space:
                    out_action_space[idx] = 1
            return out_action_space

        def action_arr_to_action_space_list(self, action_rr: np.array) -> list:
            list_of_actions = np.argwhere(action_rr == 1).tolist()
            out_action_list = [
                self.catan.base_action_space[action_idx[0]]
                for action_idx in list_of_actions
            ]
            return out_action_list

        def action_idx_to_action_tuple(self, action_idx: int) -> tuple:
            return self.catan.base_action_space[action_idx]

        def action_filter(self, action_space: np.array, model_output: np.array):
            # filter action based on action space
            possible_actions_arr = np.multiply(action_space, model_output)

            # possible_actions_list = []
            # for i in range(len(possible_actions)):
            #     if possible_actions[i] != 0:
            #         possible_actions_list.append((i, possible_actions[i]))

            # rand = np.random.uniform()
            # closest_action = min(possible_actions_list, key=lambda x: abs(x[1] - rand))

            best_action = np.argmax(possible_actions_arr)
            return best_action

        def get_potential_trade(self):
            # returns a list of tuples of the form (resource, resource)
            list_of_potential_trades = []

            for i in range(1, 6):
                if self.resources[i] >= 4:
                    for j in range(1, 6):
                        if j != i:
                            list_of_potential_trades.append((i, j))

            return list_of_potential_trades

        def prepro(self):
            "preprocess inputs"
            """
                players_resources with You, next, next+1 ,next+2
                board positions.
            """
            resource_arr = np.array([])
            for player in self.catan.players.values():
                resource_arr = np.append(
                    resource_arr, [resource for resource in player.resources.values()]
                )
            # resource_arr = np.array(ordered_resource_list)
            board = self.catan.board.ravel().astype(np.float64)
            board = np.delete(board, np.where(board == 0))  # crop

            return np.append(resource_arr, board)

        def pick_action(self):
            if self.player_type == "random":
                # Prefer to build settlements and roads over trading and doing nothing
                action, attributes = random.choice(
                    self.action_arr_to_action_space_list(self.action_space)
                )
                reward = self.reward_matrix(action)
                self.r_s.append(reward)

                return action, attributes

            elif self.player_type == "model":
                # model design:
                # forward the policy network and sample an action from the returned probability

                x = self.prepro()  # appendboard state

                z1, a1, z2, a2, z3, a3 = policy_forward(x, self.action_space)

                self.x_s.append(x)
                self.z1_s.append(z1)
                self.a1_s.append(a1)
                self.z2_s.append(z2)
                self.a2_s.append(a2)
                self.z3_s.append(z3)
                self.a3_s.append(a3)

                action_idx = np.random.choice(np.arange(a3.size), p=a3)

                # Promote the action taken (which will be adjusted by the reward)
                y = np.zeros(len(self.catan.base_action_space))
                y[action_idx] = 1
                self.y_s.append(y)

                action, attributes = self.action_idx_to_action_tuple(action_idx)
                self.actions_taken.append(action)

                reward = self.reward_matrix(action)
                self.r_s.append(reward)

                return action, attributes

        def perform_action(self, action, attributes, start=False):
            if action == 0:
                pass
            elif action == 1:
                self.build_road(attributes, start)
            elif action == 2:
                self.build_settlement(attributes, start)
            elif action == 3:
                self.build_city(attributes)
            elif action == 4:
                self.trade_with_bank(attributes)

            else:
                raise ValueError("action not in action space")

        def build_settlement(self, coords: tuple, start=False):
            # examing where I can build a settlement
            assert coords not in self.settlements, "Building in the same spot!"

            logging.debug(f"Built settlement at : {coords}")

            self.catan.board[coords[0], coords[1]] = self.tag
            self.settlements.append(coords)

            if not start:
                # removing resources
                self.resources[1] -= 1
                self.resources[2] -= 1
                self.resources[4] -= 1
                self.resources[5] -= 1

        def build_road(self, coords: tuple, start=False):
            assert coords not in self.roads, "Building in the same spot!"

            logging.debug(f"Built road at : {coords}")

            self.catan.board[coords[0], coords[1]] = self.tag
            self.roads.append(coords)
            if not start:
                # removing resources
                self.resources[1] -= 1
                self.resources[2] -= 1

        def build_city(self, coords):
            assert coords not in self.cities, "Building in the same spot!"

            logging.debug(f"Built city at : {coords}")

            self.catan.board[coords[0], coords[1]] = self.tag - 10
            self.cities.append(coords)
            self.settlements.remove(coords)

            self.resources[3] -= 3
            self.resources[4] -= 2

        def trade_with_bank(self, resource_in_resource_out):
            resource_in = resource_in_resource_out[0]
            resource_out = resource_in_resource_out[1]
            self.resources[resource_in] -= 4
            self.resources[resource_out] += 1
            logging.debug(f"Trading {resource_in} for {resource_out}")

        def get_potential_settlement(self):
            """
            Returns a list of tuples of the form (y, x) of all potential settlements
            """
            if len(self.settlements) < 2:
                # All empty spaces are potential settlements at the start of the game
                return_list = arr_to_tuple(np.argwhere(self.catan.board == -1))
            else:
                # Find all potential settlements based on existing roads
                return_list = []
                for y, x in self.roads:
                    potential_list = [
                        (y + 1, x + 1),
                        (y - 1, x + 1),
                        (y + 1, x - 1),
                        (y - 1, x - 1),
                        (y - 1, x),
                        (y + 1, x),
                    ]
                    for _y, _x in potential_list:
                        if self.catan.board[_y, _x] == -1:
                            return_list.append((_y, _x))

            # Remove all settlements that are too close to other settlements
            remove_list = []
            for y, x in return_list:
                nearby_settlemts = [
                    (y + 2, x + 2),
                    (y - 2, x + 2),
                    (y + 2, x - 2),
                    (y - 2, x - 2),
                    (y - 2, x),
                    (y + 2, x),
                ]
                for _y, _x in nearby_settlemts:
                    if self.catan.board[_y, _x] in self.catan.player_tags:
                        remove_list.append((y, x))
                        break

            for y, x in remove_list:
                return_list.remove((y, x))

            return return_list

        def get_potential_road(self, coords=None):
            # This only applies to the start of the game where the road has to link to a settlement
            act_settlements = (
                [coords] if coords != None and len(self.roads) < 2 else self.settlements
            )

            # Find all potential roads based on settlements
            return_list_sett = []
            for y, x in act_settlements:
                potential_list = [
                    (y + 1, x),
                    (y - 1, x),
                    (y + 1, x + 1),
                    (y - 1, x - 1),
                    (y + 1, x - 1),
                    (y - 1, x + 1),
                ]
                for _y, _x in potential_list:
                    if self.catan.board[_y, _x] == -2:
                        return_list_sett.append((_y, _x))

            # Find all potential roads based on roads
            return_list_road = []
            if coords == None:
                for y, x in self.roads:
                    potential_list = [
                        [y, x + 2, [(y + 1, x + 1), (y - 1, x + 1)]],
                        [y, x - 2, [(y + 1, x - 1), (y - 1, x - 1)]],
                        [y + 2, x + 1, [(y + 1, x + 1), (y + 1, x)]],
                        [y - 2, x + 1, [(y - 1, x + 1), (y - 1, x)]],
                        [y + 2, x - 1, [(y + 1, x - 1), (y + 1, x)]],
                        [y - 2, x - 1, [(y - 1, x - 1), (y - 1, x)]],
                    ]

                    for _y, _x, blocking in potential_list:
                        if (
                            self.catan.board[_y, _x] == -2
                            and (_y, _x) not in return_list_sett
                        ):
                            for _y2, _x2 in blocking:
                                if (
                                    self.catan.board[_y2, _x2]
                                    not in self.catan.player_tags
                                ):
                                    return_list_road.append((_y, _x))

            return return_list_sett + return_list_road

        def get_potential_city(self):
            return self.settlements

        def reward_matrix(self, action: int):
            reward_matrix = {
                "solo": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    "win": (self.points**2 - 6**2) + 2000 * (1 / (catan.turn + 1)),
                    "loss": self.points**2 - 6**2,
                    # self.points
                    # self.points
                },
                "multi": {
                    0: 0,
                    1: 0.2,
                    2: 2.5,
                    3: 5,
                    4: 0,
                    "win": (self.points**2 - 6**2) + 2000 * (1 / (catan.turn + 1)),
                    "loss": self.points**2 - 6**2,
                },
            }

            return reward_matrix[self.mode][action]

        def player_turn(self):
            action = None
            while action != 0:
                # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
                self.action_space = self.get_action_space()
                action, attributes = self.pick_action()
                logging.debug(f"Action: {action}, Attributes: {attributes}")
                self.perform_action(action, attributes)

        def player_start(self):
            # for the start we need to enforce the action space to
            logging.debug(f"<-- Player {self.tag} -->")

            # 1. build settlement
            self.action_space = self.get_action_space(start="settlement")
            action, attributes = self.pick_action()
            self.perform_action(action, attributes, start=True)

            # 2. build road
            self.action_space = self.get_action_space(
                start="road", attributes=attributes
            )
            action, attributes = self.pick_action()
            self.perform_action(action, attributes, start=True)

        def player_preturn_debug(self):
            logging.debug(f"<-- Player {self.tag} -->")

        def player_posturn_debug(self):
            self.catan.print_board()
            # logging.debug(f"<-- Player {self.tag} -->")
            logging.debug(f"Resources: {self.resources}")
            logging.debug(f"Points: {self.points}")
            logging.debug(
                f"Potential: {self.potential_settlement} {self.potential_road} {self.potential_trade}"
            )

        def player_episode_audit(self):
            unique, counts = np.unique(self.actions_taken, return_counts=True)
            logging.info(f"<-- Player {self.tag} -->")
            logging.info(f"Actions taken: {dict(zip(unique, counts))}")
            logging.info(f"Settlements : {self.settlements}")
            logging.info(f"City : {self.cities}")
            logging.info(f"Roads : {self.roads}")
            logging.info(f"End Resources: {self.resources}")
            self.reward_sum = round(np.sum(np.vstack(self.r_s)))
            logging.info(f"Reward sum: {self.reward_sum}")

            logging.info(f"Points: {self.points} ")


if __name__ == "__main__":
    #
    logname = "catan_game.txt"
    logging.basicConfig(
        format="%(message)s",
        level=20,
        # filename=logname,
        # filemode="w",
    )

    # Model_base
    # Hyperparameters
    H = 2048  # number of hidden layer 1 neurons
    W = 1024  # number of hidden layer 2 neurons
    batch_size = 5  # every how many episodes to do a param update?
    episodes = 1000
    learning_rate = 1e-5
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    max_turn = 500
    player_type = ["model", "model"]  # model | random
    player_count = 2  # 1 - 4

    resume = False  # resume from previous checkpoint?
    render = False

    def softmax(x):
        zero_indices = np.where(x == 0)

        shiftx = x - np.max(x, axis=0)
        exps = np.exp(shiftx)

        exps[zero_indices] = 0
        return exps / np.sum(exps, axis=0)

    def discount_rewards(r):
        """take 1D float array of rewards and compute discounted reward"""
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def relu(x):
        x[x < 0] = 0
        return x

    def d_relu(x):
        return x >= 0

    def policy_forward(x, mask):
        # forward pass: Take in board state, return probability of taking action [0,1,2,3]
        # Ne = Neurons in hidden state. As = Action Space
        z1 = model["W1"] @ x  # Ne x 483 * 483 x M = Ne x M
        a1 = relu(z1)  # Ne x M

        z2 = model["W2"] @ a1  # Ne x 483 * 483 x M = Ne x M
        a2 = relu(z2)  # Ne x M

        z3 = model["W3"] @ a2  # As x Ne * Ne x M = As x M
        m3 = np.multiply(z3, mask)
        a3 = softmax(m3)  # As x M

        return (
            z1,
            a1,
            z2,
            a2,
            z3,
            a3,
        )  # return probability of taking action [0,1,2,3], and hidden state

    def policy_backward(x_s, z1_s, a1_s, z2_s, a2_s, z3_s, a3_s, y_s, r_s):
        """backward pass. (eph is array of intermediate hidden states)"""
        dz3 = (a3_s - y_s) * discount_rewards(
            r_s
        )  # based on cross entropy + softmax regulated by rewards. Dim = M x As

        dW3 = dz3.T @ a2_s

        da2 = dz3 @ model["W3"]  # Dim = M x As * As x Ne = M x Ne
        dz2 = da2 * d_relu(z2_s)  # Dim = M x Ne ** M x Ne = M x Ne
        dW2 = dz2.T @ a1_s  # Dim = As x M * M x Ne = As x Ne

        da1 = dz2 @ model["W2"]  # Dim = M x As * As x Ne = M x Ne
        dz1 = da1 * d_relu(z1_s)  # Dim = M x Ne ** M x Ne = M x Ne
        dW1 = dz1.T @ x_s  # Dim = Ne x M * M x 483 = Ne x 483

        return {"W1": dW1, "W2": dW2, "W3": dW3}

    def plot_running_avg(y: list, window=10):
        average_y = []
        for ind in range(len(y) - window + 1):
            average_y.append(np.mean(y[ind : ind + window]))

        plt.plot(average_y)
        plt.show()

    # Stacking
    running_reward = None
    turn_list = []
    reward_list = []

    # create dummy catan game to get action spaces
    catan = Catan(seed=1, player_type=player_type, player_count=player_count)

    D = len(
        next(iter(catan.players.values())).prepro()
    )  # sum([len(player.prepro()) for key, player in catan.players.items()])
    # D = 23 * 21  # input dimensionality: 23 x 21 grid (483)
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(
        D
    )  # "Xavier" initialization. Dim = H x 483

    types_of_actions = len(catan.get_all_action_space())
    model["W2"] = np.random.randn(W, H) / np.sqrt(H)  # Dim = As x Ne

    model["W3"] = np.random.randn(types_of_actions, W) / np.sqrt(W)  # Dim = As x Ne

    grad_buffer = {
        k: np.zeros_like(v) for k, v in model.items()
    }  # update buffers that add up gradients over a batch
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

    start = datetime.now()
    # Run experiment
    for episode in range(episodes):
        logging.info(f"Episode {episode}")
        catan = Catan(
            seed=1, player_type=player_type, player_count=player_count, mode="multi"
        )

        catan.game_start()

        while catan.game_over == False and catan.turn < max_turn:
            catan.turn += 1
            logging.debug(f"Turn {catan.turn}")

            for player_tag, player in catan.players.items():
                # Phase 1: roll dice and get resources
                catan.board_turn()
                # Phase 2: player performs actions
                player.player_preturn_debug()
                player.player_turn()
                player.recalc_points()
                player.player_posturn_debug()

        logging.info(f"End board state:")
        catan.print_board(debug=False)

        # Episode Audit
        for player_tag, player in catan.players.items():
            if catan.turn == max_turn:
                player.r_s[-1] += player.reward_matrix("loss")

            player.player_episode_audit()

        logging.info(f"Game finished in {catan.turn} turns. Winner: {catan.winner}")
        turn_list.append(catan.turn)
        reward_list.append(player.reward_sum)

        for player_tag, player in catan.players.items():
            if player.player_type == "model":
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                ep_x_s = np.vstack(player.x_s)
                ep_z1_s = np.vstack(player.z1_s)
                ep_a1_s = np.vstack(player.a1_s)
                ep_z2_s = np.vstack(player.z2_s)
                ep_a2_s = np.vstack(player.a2_s)
                ep_z3_s = np.vstack(player.z3_s)
                ep_a3_s = np.vstack(player.a3_s)
                ep_y_s = np.vstack(player.y_s)
                ep_r_s = np.vstack(player.r_s)

                avg_ep_loss = (ep_a3_s - ep_y_s) / len(ep_y_s)

                grad = policy_backward(
                    ep_x_s,
                    ep_z1_s,
                    ep_a1_s,
                    ep_z2_s,
                    ep_a2_s,
                    ep_z3_s,
                    ep_a3_s,
                    ep_y_s,
                    ep_r_s,
                )
                for k in model:
                    grad_buffer[k] += grad[k]  # accumulate grad over batch

                # perform rmsprop parameter update every batch_size episodes
                if episode % batch_size == 0:
                    for k, v in model.items():
                        g = grad_buffer[k]  # gradient
                        rmsprop_cache[k] = (
                            decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                        )
                        model[k] -= (
                            learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-7)
                        )
                        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
    logging.info(f"Time taken: {datetime.now() - start}")

    # save model
    pickle.dump(model, open("catan_model.pickle", "wb"))

    plot_running_avg(turn_list)
    plot_running_avg(reward_list)
