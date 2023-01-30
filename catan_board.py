import random
import numpy as np
import pickle


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

    def __init__(self):
        self.turn = 0

        self.knight_card = {
            "knight": 14,
            "victory_point": 5,
            "road_building": 2,
            "year_of_plenty": 2,
            "monopoly": 2,
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
            1: 3,
            2: 4,
            3: 3,
            4: 4,
            5: 4,
            6: 1,  # desert
        }

        # Add 10 to each number to avoid clashing with resource numbers
        self.number_tokens = [
            12,
            13,
            13,
            14,
            14,
            15,
            15,
            16,
            16,
            18,
            18,
            19,
            19,
            20,
            20,
            21,
            21,
            22,
            0,
        ]
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
        self.players = self.generate_players(1)
        self.game_over = False

    def generate_players(self, player_count: int):
        players = {}
        for i in range(1, player_count + 1):
            players[i - 10] = Catan.Player(self, i - 10)
        return players

    def print_board(self):
        board = self.board.copy()
        print(board.shape)
        board = board[2 : board.shape[0] - 2, 2 : board.shape[1] - 2]
        zero_tmp = 97
        board[board == 0] = zero_tmp
        board_string = np.array2string(board)
        board_string = board_string.replace(f"{zero_tmp}", "  ")
        print(board_string)

    def board_turn(self):

        """
        roll dice and deliver resources
        """
        roll = self.roll()
        if roll == 17:
            # need to cut resources by half
            pass
        else:
            print(f"Rolled: {roll}")
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
            print(f"Resource type: {resource_type}")

            if resource_type == 6:
                continue
            else:
                for potential_y, potential_x in list_of_potential_owners:
                    player_tag = self.board[potential_y, potential_x]

                    if player_tag != -1:
                        self.add_resource(player_tag, resource_type, 1)
                        print(f"Player {player_tag} got resource {resource_type}")

    def add_resource(self, player_tag, resource_type, resource_num: int):
        self.players[player_tag].resources[resource_type] += 1

    def game_start(self):
        # Clockwise order
        for player_tag, player in self.players.items():
            player.player_start()
        # Anticlockwise order
        for player_tag, player in reversed(self.players.items()):
            player.player_start()
        self.print_board()
        print("Game started!")

    def generate_board(self):

        resource_list = [
            key for key, value in self.resource_cards.items() for x in range(value)
        ]

        def randpick_and_pop(__list: list):
            picked = random.randint(0, len(__list) - 1)
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
                # The extra padding is to make sure the validation works
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

        for (y, x) in self.center_coords:
            number = randpick_and_pop(self.number_tokens)
            resource = randpick_and_pop(resource_list)
            arr[y, x] = 50  # Knight reference
            arr[y - 1, x] = resource
            arr[y + 1, x] = number

        return arr

    class Player:
        def __init__(self, catan, tag: int) -> None:
            self.catan = catan
            self.player_type = "random_esc"
            self.tag = tag
            self.resources = {
                # name, count, Start with 2 settlement + 2 roads
                1: 4,  # brick
                2: 4,  # lumber
                3: 0,  # ore
                4: 2,  # grain
                5: 2,  # wool
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
            self.action_space = self.get_action_space()
            self.potential_settlement = []
            self.potential_road = []
            self.potential_trade = []
            self.points = 0

        def recalc_points(self):
            self.points = (
                len(self.settlements)
                + 2 * len(self.cities)
                + self.knight_cards["victory_point"]
            )
            print(f"Player {self.tag} has {self.points} points")
            if self.points >= 10:
                print(f"Player {self.tag} wins!")
                self.catan.game_over = True

        def get_action_space(self):
            # leaving city and development car later
            action_space = [
                1,  # nothing
                0,  # settlement
                0,  # road
                0,  # trade
            ]
            if (
                self.resources[1] >= 1
                and self.resources[2] >= 1
                and self.resources[4] >= 1
                and self.resources[5] >= 1
            ):
                self.potential_settlement = self.get_potential_settlement()
                if len(self.potential_settlement) > 0:
                    action_space[1] = 1

            if self.resources[1] >= 1 and self.resources[2] >= 1:
                self.potential_road = self.get_potential_road()
                if len(self.potential_road) > 0:
                    action_space[2] = 1
            if (
                self.resources[1] >= 4
                or self.resources[2] >= 4
                or self.resources[3] >= 4
                or self.resources[4] >= 4
                or self.resources[5] >= 4
            ):
                self.potential_trade = self.get_potential_trade()
                if len(self.potential_trade) > 0:
                    action_space[3] = 1

            return action_space

        def get_potential_trade(self):
            # returns a list of tuples of the form (resource, resource)
            list_of_potential_trades = []
            for i in range(5):
                if self.resources[i + 1] >= 4:
                    for j in range(5):
                        if j != i:
                            list_of_potential_trades.append((i + 1, j + 1))

            return list_of_potential_trades

        def pick_action(self):
            if self.player_type == "random":
                # Pure random actions
                potential_actions = []
                for i in range(len(self.action_space)):
                    if self.action_space[i] == 1:
                        potential_actions.append(i)
                action = random.choice(potential_actions)
                if action == 0:
                    attributes = None
                elif action == 1:
                    attributes = random.choice(self.potential_settlement)
                elif action == 2:
                    attributes = random.choice(self.potential_road)
                elif action == 3:
                    attributes = random.choice(self.potential_trade)
                return action, attributes

            elif self.player_type == "random_esc":
                # Prefer to build settlements and roads over trading and doing nothing
                if self.action_space[1] == 1:
                    action = 1
                    attributes = random.choice(self.potential_settlement)
                elif self.action_space[2] == 1:
                    action = 2
                    attributes = random.choice(self.potential_road)
                elif self.action_space[3] == 1:
                    action = 3
                    attributes = random.choice(self.potential_trade)
                elif self.action_space[0] == 1:
                    action = 0
                    attributes = 1

                return action, attributes

            elif self.player_type == "model":
                # model design:
                # forward the policy network and sample an action from the returned probability
                aprob, h = self.policy_forward(self.board)
                action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        def perform_action(self, action, attributes):

            if action == 0:
                pass
            elif action == 1:
                self.build_settlement(attributes)
            elif action == 2:
                self.build_road(attributes)
            elif action == 3:
                self.trade_with_bank(attributes)
            else:
                raise ValueError("action not in action space")

        def build_settlement(self, coords: tuple):

            # examing where I can build a settlement
            print(f"Built settlement at : {coords}")
            self.catan.board[coords[0], coords[1]] = self.tag
            self.settlements.append(coords)

            # removing resources
            self.resources[1] -= 1
            self.resources[2] -= 1
            self.resources[4] -= 1
            self.resources[5] -= 1

        def build_road(self, coords: tuple):
            self.catan.board[coords[0], coords[1]] = self.tag
            self.roads.append(coords)

            # removing resources
            self.resources[1] -= 1
            self.resources[2] -= 1

        def build_city(self, coords):
            self.catan.board[coords[0], coords[1]] = self.tag

        def trade_with_bank(self, resource_in_resource_out):
            resource_in = resource_in_resource_out[0]
            resource_out = resource_in_resource_out[1]
            self.resources[resource_in] -= 4
            self.resources[resource_out] += 1

        def get_potential_settlement(self):
            if len(self.settlements) < 2:
                return np.argwhere(self.catan.board == -1).tolist()

            potential_list = []
            for y, x in self.roads:
                potential_list.append((y + 1, x + 1))
                potential_list.append((y - 1, x + 1))
                potential_list.append((y + 1, x - 1))
                potential_list.append((y - 1, x - 1))
                potential_list.append((y - 1, x))
                potential_list.append((y + 1, x))
            return_list = []
            for y, x in potential_list:
                if self.catan.board[y, x] == -1:
                    return_list.append((y, x))

            return return_list

        def get_potential_road(self, coords=None):
            if coords != None:
                # This only applies to the start of the game where the road has to link to a settlement
                act_settlements = [coords]
            else:
                act_settlements = self.settlements

            potential_list = []
            for y, x in act_settlements:
                potential_list.append((y + 1, x))
                potential_list.append((y - 1, x))
                potential_list.append((y + 1, x + 1))
                potential_list.append((y - 1, x - 1))
                potential_list.append((y + 1, x - 1))
                potential_list.append((y - 1, x + 1))

            return_list = []
            for y, x in potential_list:
                if self.catan.board[y, x] == -2:
                    return_list.append((y, x))

            return return_list

        def player_turn(self):
            action = None

            while action != 0:
                # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
                self.action_space = self.get_action_space()
                action, attributes = self.pick_action()
                print(f"Action: {action}, Attributes: {attributes}")
                self.perform_action(action, attributes)

            self.recalc_points()

        def player_start(self):
            # for the start we enforce the action space
            # build settlement
            self.action_space = [0, 1, 0, 0, 0, 0]
            self.potential_settlement = self.get_potential_settlement()
            action, attributes = self.pick_action()
            self.perform_action(action, attributes)

            # build road
            self.action_space = [0, 0, 1, 0, 0, 0]
            self.potential_road = self.get_potential_road(attributes)
            action, attributes = self.pick_action()
            self.perform_action(action, attributes)


if __name__ == "__main__":
    # hyperparameters
    H = 200  # number of hidden layer neurons
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    resume = False  # resume from previous checkpoint?
    render = False

    D = 23 * 21  # input dimensionality: 23 x 21 grid
    model = {}
    model["W1"] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization

    types_of_actions = 4
    model["W2"] = np.random.randn(4, H) / np.sqrt(H)
    grad_buffer = {
        k: np.zeros_like(v) for k, v in model.items()
    }  # update buffers that add up gradients over a batch
    rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

    def softmax(arr: list):
        # softmax probabilities
        return [np.exp(-y) / sum([np.exp(-x) for x in arr]) for y in arr]

    def prepro(I):
        "preprocess inputs"
        I = I[2:25, 2:23]  # crop
        return I.astype(np.float64).ravel()

    def discount_rewards(r):
        """take 1D float array of rewards and compute discounted reward"""
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(x):
        # forward pass: Take in board state, return probability of taking action [0,1,2,3]
        h = np.dot(model["W1"], x)
        h[h < 0] = 0  # ReLU nonlinearity
        arr_logp = np.dot(model["W2"], h)
        p = softmax(arr_logp)

        return (
            p,
            h,
        )  # return probability of taking action [0,1,2,3], and hidden state

    def policy_backward(ep_hs, ep_xs, ep_dlogp):
        """backward pass. (eph is array of intermediate hidden states)"""

        dW2 = np.dot(ep_hs.T, ep_dlogp).ravel()
        dh = np.outer(ep_dlogp, model["W2"])
        dh[ep_hs <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, ep_xs)
        return {"W1": dW1, "W2": dW2}

    def action_filter(action_space, action):
        # filter action based on action space
        possible_actions = np.multiply(action_space, action)

        possible_actions_list = []
        for i in range(len(possible_actions)):
            if possible_actions[i] != 0:
                possible_actions_list.append((i, possible_actions[i]))

        rand = np.random.uniform()
        closest_action = min(possible_actions_list, key=lambda x: abs(x[1] - rand))
        return closest_action[0]

    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    # Run experiment for 10 rounds

    for j in range(10):
        catan = Catan()

        # Game start
        catan.print_board()
