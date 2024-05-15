from __future__ import annotations

import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
import logging
from enum import Enum

from catan_game import Player, Catan


class PlayerType(Enum):
    RANDOM = "random"
    MODEL = "model"


class PlayerAI(Player):
    def __init__(self, catan: CatanAI, tag: int, player_type: PlayerType) -> None:
        self.player_type = player_type
        self.tag = tag
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
        super().__init__(catan, tag)
        self.catan = catan

    def recalc_points(self):
        self.points = (
            len(self.settlements)
            + 2 * len(self.cities)
            + self.knight_cards["victory_point"]
        )
        logging.debug(f"Player {self.tag} has {self.points} points")
        if self.points >= 10:
            self.r_s[-1] += self.reward_matrix("win")

            # give other players a negative reward
            for tag, player in self.catan.players.items():
                if tag != self.tag:
                    player.r_s[-1] += self.reward_matrix("loss")

            logging.info(f"Player {self.tag} wins!")

            self.catan.game_over = True
            self.catan.winner = self.tag

    def discard_resources_turn(self):
        if sum(self.resources.values()) > 7:
            if self.catan.player_type == PlayerType.RANDOM:
                self.discard_resources_random()
            elif self.player_type == PlayerType.MODEL:
                self.discard_resources_model()

    def discard_resources_random(self):
        while sum(self.resources.values()) > 7:
            self.resources[random.randint(1, 5)] -= 1

    def discard_resources_model(self):
        while sum(self.resources.values()) > 7:
            # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
            self.action_space = self.get_action_space(start="discard")
            action, attributes = self.pick_action()
            logging.debug(f"Action: {action}, Attributes: {attributes}")
            self.perform_action(action, attributes)

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
        # board = np.delete(board, np.where(board == 0))  # crop

        return np.append(resource_arr, board)

    def pick_action(self):
        if self.player_type == PlayerType.RANDOM:
            # Prefer to build settlements and roads over trading and doing nothing
            action, attributes = random.choice(
                self.action_arr_to_action_space_list(self.action_space)
            )
            reward = self.reward_matrix(action)
            self.r_s.append(reward)

            return action, attributes

        elif self.player_type == PlayerType.MODEL:
            # model design:
            # forward the policy network and sample an action from the returned probability

            x = self.prepro()  # append board state

            z1, a1, z2, a2, z3, a3 = self.catan.catan_training.policy_forward(
                x, self.action_space
            )

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

    def reward_matrix(self, action: int):
        reward_matrix = {
            "solo": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                "win": (self.points**2 - 6**2) + 2000 * (1 / (self.catan.turn + 1)),
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
                5: 0,
                "win": (self.points**2 - 6**2) + 2000 * (1 / (self.catan.turn + 1)),
                "loss": self.points**2 - 6**2,
            },
        }

        return reward_matrix[self.catan.mode][action]

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
        self.action_space = self.get_action_space(start="road", attributes=attributes)
        action, attributes = self.pick_action()
        self.perform_action(action, attributes, start=True)


class CatanAI(Catan):
    def __init__(
        self,
        seed,
        player_count: int,
        player_type: list[PlayerType],
        mode: str,
        catan_training: CatanAITraining,
    ) -> None:
        self.seed = seed
        self.player_count = player_count
        self.player_type = player_type
        self.mode = mode
        self.catan_training: CatanAITraining = catan_training

        self.players: dict[str, PlayerAI]
        super().__init__()

    def generate_players(
        self,
    ) -> dict[str, PlayerAI]:
        players: dict[str, PlayerAI] = {}
        for i in range(self.player_count):
            player = PlayerAI(
                catan=self,
                tag=i - 9,
                player_type=self.player_type[i],
            )
            players[i - 9] = player
        return players

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
        # fmt: off
        number_tokens = [
            12,13,13,14,14,15,15,16,16,
            18,18,19,19,20,20,21,21,22,0,
        ]
        # fmt: on

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
        # fmt: off
        arr = np.array(
            [
                # The extra padding is to make sure the validation works for delivering resources.
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

    def board_turn(self):
        """
        roll dice and deliver resources
        """
        roll = self.roll()
        logging.debug(f"Rolled: {roll - 10}")

        if roll == 17:
            # need to cut resources by half
            for player in self.players.values():
                player.discard_resources_turn()
            pass
        else:
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


def softmax(x):
    zero_indices = np.where(x == 0)

    shiftx = x - np.max(x, axis=0)
    exps = np.exp(shiftx)

    exps[zero_indices] = 0
    return exps / np.sum(exps, axis=0)


def relu(x):
    x[x < 0] = 0
    return x


class CatanAITraining:
    # Model_base
    # Hyperparameters

    H = 2048  # number of hidden layer 1 neurons
    W = 1024  # number of hidden layer 2 neurons
    batch_size = 5  # every how many episodes to do a param update?
    episodes = 10
    learning_rate = 1e-5
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    max_turn = 500
    player_type = [PlayerType.MODEL, PlayerType.MODEL]  # model | random
    player_count = 2  # 1 - 4
    # Stacking
    running_reward = None
    turn_list = []
    reward_list = []

    def __init__(self):
        # create dummy catan game to get action spaces
        self.catan = CatanAI(
            seed=1,
            player_type=self.player_type,
            player_count=self.player_count,
            mode="multi",
            catan_training=self,
        )

        self.D = len(
            next(iter(self.catan.players.values())).prepro()
        )  # sum([len(player.prepro()) for key, player in catan.players.items()])
        # D = 23 * 21  # input dimensionality: 23 x 21 grid (483)
        self.model = {}
        self.model["W1"] = np.random.randn(self.H, self.D) / np.sqrt(
            self.D
        )  # "Xavier" initialization. Dim = H x 483

        self.types_of_actions = len(self.catan.generate_all_action_space())
        self.model["W2"] = np.random.randn(self.W, self.H) / np.sqrt(
            self.H
        )  # Dim = As x Ne

        self.model["W3"] = np.random.randn(self.types_of_actions, self.W) / np.sqrt(
            self.W
        )  # Dim = As x Ne

        self.grad_buffer = {
            k: np.zeros_like(v) for k, v in self.model.items()
        }  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {
            k: np.zeros_like(v) for k, v in self.model.items()
        }  # rmsprop memory

    def policy_forward(self, x, mask):
        # forward pass: Take in board state, return probability of taking action [0,1,2,3]
        # Ne = Neurons in hidden state. As = Action Space
        z1 = self.model["W1"] @ x  # Ne x 483 * 483 x M = Ne x M
        a1 = relu(z1)  # Ne x M

        z2 = self.model["W2"] @ a1  # Ne x 483 * 483 x M = Ne x M
        a2 = relu(z2)  # Ne x M

        z3 = self.model["W3"] @ a2  # As x Ne * Ne x M = As x M
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

    def discount_rewards(self, r):
        """take 1D float array of rewards and compute discounted reward"""
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def d_relu(self, x):
        return x >= 0

    def policy_backward(self, x_s, z1_s, a1_s, z2_s, a2_s, z3_s, a3_s, y_s, r_s):
        """backward pass. (eph is array of intermediate hidden states)"""
        dz3 = (a3_s - y_s) * self.discount_rewards(
            r_s
        )  # based on cross entropy + softmax regulated by rewards. Dim = M x As

        dW3 = dz3.T @ a2_s

        da2 = dz3 @ self.model["W3"]  # Dim = M x As * As x Ne = M x Ne
        dz2 = da2 * self.d_relu(z2_s)  # Dim = M x Ne ** M x Ne = M x Ne
        dW2 = dz2.T @ a1_s  # Dim = As x M * M x Ne = As x Ne

        da1 = dz2 @ self.model["W2"]  # Dim = M x As * As x Ne = M x Ne
        dz1 = da1 * self.d_relu(z1_s)  # Dim = M x Ne ** M x Ne = M x Ne
        dW1 = dz1.T @ x_s  # Dim = Ne x M * M x 483 = Ne x 483

        return {"W1": dW1, "W2": dW2, "W3": dW3}

    def plot_running_avg(self, y: list, window=10):
        average_y = []
        for ind in range(len(y) - window + 1):
            average_y.append(np.mean(y[ind : ind + window]))

        plt.plot(average_y)
        plt.show()

    def training(self):
        # Run experiment
        for episode in range(self.episodes):
            logging.info(f"Episode {episode}")
            catan = CatanAI(
                seed=1,
                player_type=self.player_type,
                player_count=self.player_count,
                mode="multi",
                catan_training=self,
            )

            catan.game_start()

            while catan.game_over is False and catan.turn < self.max_turn:
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

            logging.info("End board state:")
            catan.print_board(debug=False)

            # Episode Audit
            for player_tag, player in catan.players.items():
                if catan.turn == self.max_turn:
                    player.r_s[-1] += player.reward_matrix("loss")

                player.player_episode_audit()

            logging.info(f"Game finished in {catan.turn} turns. Winner: {catan.winner}")
            self.turn_list.append(catan.turn)
            self.reward_list.append(player.reward_sum)

            for player_tag, player in catan.players.items():
                if player.player_type == PlayerType.MODEL:
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

                    grad = self.policy_backward(
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
                    for k in self.model:
                        self.grad_buffer[k] += grad[k]  # accumulate grad over batch

                    # perform rmsprop parameter update every batch_size episodes
                    if episode % self.batch_size == 0:
                        for k, v in self.model.items():
                            g = self.grad_buffer[k]  # gradient
                            self.rmsprop_cache[k] = (
                                self.decay_rate * self.rmsprop_cache[k]
                                + (1 - self.decay_rate) * g**2
                            )
                            self.model[k] -= (
                                self.learning_rate
                                * g
                                / (np.sqrt(self.rmsprop_cache[k]) + 1e-7)
                            )
                            self.grad_buffer[k] = np.zeros_like(
                                v
                            )  # reset batch gradient buffer

        # save model
        pickle.dump(self.model, open("catan_model.pickle", "wb"))
        pickle.dump(self.turn_list, open("turn_list.pickle", "wb"))
        pickle.dump(self.reward_list, open("reward_list.pickle", "wb"))

        self.plot_running_avg(self.turn_list)
        self.plot_running_avg(self.reward_list)


if __name__ == "__main__":
    #
    logname = "catan_game.txt"
    logging.basicConfig(
        format="%(message)s",
        level=20,
        # filename=logname,
        # filemode="w",
    )
    CatanAITraining().training()
