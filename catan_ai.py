from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
import logging
from enum import Enum
from catan_game import Player, Catan, Action
from collections import OrderedDict


class PlayerType(Enum):
    RANDOM = "random"
    MODEL = "model"


class PlayerAI(Player):
    def __init__(
        self,
        catan: CatanAI,
        tag: int,
        player_type: PlayerType,
        reward_matrix_dict: dict = {"win": 0, "loss": 0},
    ) -> None:
        self.reward_matrix_dict: dict = reward_matrix_dict
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
        super().recalc_points()
        if self.points >= self.catan.max_points:
            self.r_s[-1] += self.reward_matrix("win") - self.catan.turn

            # give other players a negative reward
            for tag, player in self.catan.players.items():
                if tag != self.tag:
                    player.r_s[-1] += self.reward_matrix("loss")

            logging.debug(f"Player {self.tag} wins!")

    def discard_resources_turn(self):
        if sum(self.resources.values()) > 7:
            if self.player_type == PlayerType.RANDOM:
                self.discard_resources_random()
            elif self.player_type == PlayerType.MODEL:
                self.discard_resources_model()

    def discard_resources_random(self):
        while sum(self.resources.values()) > 7:
            self.resources[random.choice(self.available_resources())] -= 1

    def discard_resources_model(self):
        half_resources = sum(self.resources.values()) // 2 + 1
        while sum(self.resources.values()) > half_resources:
            # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
            self.action_space = self.get_action_space(situation=Action.DISCARD)
            action, attributes = self.pick_action()
            logging.debug(f"Disard Action: {action}, Attributes: {attributes}")
            self.perform_action(action, attributes)

    def prepro(self):
        "preprocess inputs"
        """
            players_resources with You, next, next+1 ,next+2
            board positions.

            Appending opponent's dev card count.
        """

        resource_arr = np.array([])
        dev_cards_arr = np.array([])

        players = self.catan.players.copy()

        first_player = next(iter(players))
        while first_player != self.tag:
            players.move_to_end(first_player)
            first_player = next(iter(players))

        for player in players.values():
            resource_arr = np.append(resource_arr, list(player.resources.values()))
            dev_cards_arr = np.append(dev_cards_arr, [player.dev_card_count])
            dev_cards_arr = np.append(
                dev_cards_arr, list(player.dev_card_used.values())
            )

        board = self.catan.board.ravel().astype(np.int8)
        board = np.delete(board, np.where(board == 0))  # crop
        board = self.one_hot_encoder_board(board)

        assert len(board) == 192
        concentated_arr = np.concatenate(
            (self.catan.turn, resource_arr, dev_cards_arr, board), axis=None
        )

        return concentated_arr

    def one_hot_encoder_board(self, board: np.ndarray):
        assert set(board).issubset(set(self.catan.unique_tags))
        board = np.append(board, self.catan.unique_tags)  # board + all unique tags

        unique, inverse = np.unique(board, return_inverse=True)

        onehot = np.eye(unique.shape[0])[inverse]
        onehot = onehot[: len(onehot) - len(self.catan.unique_tags)]

        return onehot

    def pick_action(self):
        if self.player_type == PlayerType.RANDOM:
            # Prefer to build settlements and roads over trading and doing nothing
            action, attributes = random.choice(
                self.action_arr_to_action_space_list(self.action_space)
            )
            self.r_s.append(0)

            return action, attributes

        elif self.player_type == PlayerType.MODEL:
            # model design:
            # forward the policy network and sample an action from the returned probability

            x = self.prepro()  # append board state

            z1, a1, z2, a2, z3, a3 = policy_forward(
                x, self.action_space, self.catan.catan_training.model
            )

            self.x_s.append(x)
            self.z1_s.append(z1)
            self.a1_s.append(a1)
            self.z2_s.append(z2)
            self.a2_s.append(a2)
            self.z3_s.append(z3)
            self.a3_s.append(a3)

            # choose a random action based on the action probabilities returned.
            action_idx = np.random.choice(np.arange(a3.size), p=a3)

            # Promote the action taken (which will be adjusted by the reward)
            y = np.zeros(len(self.catan.base_action_space))
            y[action_idx] = 1
            self.y_s.append(y)

            action, attributes = self.action_idx_to_action_tuple(action_idx)
            self.actions_taken.append(action)

            self.r_s.append(0)

            return action, attributes

    def reward_matrix(self, action: str) -> int:
        # """Scaling reward by points and winning"""

        return self.reward_matrix_dict[action]

    def player_turn(self):
        action = None
        self.dev_cards_turn = self.base_dev_cards()
        while action is not Action.PASS:
            # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
            action = self.player_subturn()
        self.player_turn_action_list = []

    def player_subturn(self):
        self.action_space = self.get_action_space()
        action, attributes = self.pick_action()
        logging.debug(f"Action: {action}, Attributes: {attributes}")
        self.perform_action(action, attributes)
        self.player_turn_action_list.append(action)
        return action

    def player_start(self):
        # for the start we need to enforce the action space to
        logging.debug(f"<-- Player {self.tag} -->")

        # 1. build settlement
        self.action_space = self.get_action_space(situation=Action.SETTLEMENT)
        action, attributes = self.pick_action()
        self.perform_action(action, attributes, remove_resources=False)

        # 2. build road
        self.action_space = self.get_action_space(
            situation=Action.ROAD, attributes=attributes
        )
        action, attributes = self.pick_action()
        self.perform_action(action, attributes, remove_resources=False)


class CatanAI(Catan):
    def __init__(
        self,
        seed,
        player_count: int,
        player_type: list[PlayerType],
        player_matrices: list[dict],
        mode: str,
        catan_training: CatanAITraining,
    ) -> None:
        self.seed = seed
        self.player_count = player_count
        self.player_type = player_type
        self.player_matrices = player_matrices
        self.mode = mode
        self.catan_training: CatanAITraining = catan_training

        super().__init__()
        self.players: OrderedDict[str, PlayerAI]

    def generate_players(
        self,
    ) -> dict[str, PlayerAI]:
        players: dict[str, PlayerAI] = OrderedDict()
        for i in range(self.player_count):
            player = PlayerAI(
                catan=self,
                tag=i - 9,
                player_type=self.player_type[i],
                reward_matrix_dict=self.player_matrices[i],
            )
            players[i - 9] = player
        return players

    def generate_board(self):
        resource_list = [
            resource.value
            for resource, count in self.resource_tokens.items()
            for x in range(count)
        ]
        random.Random(self.seed).shuffle(resource_list)

        number_list = [
            key + 10 for key, value in self.number_tokens.items() for x in range(value)
        ]
        random.Random(self.seed + 1).shuffle(number_list)

        """
        Grid of 5 hexagons CENTER = resource number, CENTER+1 = resource type
        21 x 23

        """
        # fmt: off
        arr = self.empty_board

        # Shuffle based on seed, off by one

        for y, x in self.center_coords:
            resource = resource_list.pop()

            if resource == 6:
                number = 1 # this is for desert as a dummy holder
            else:
                number =number_list.pop()
            
            
            arr[y - 1, x] = resource
            arr[y + 1, x] = number

            if resource == 6:
                arr[y, x] = self.robber_tag  # robber reference
            else:
                arr[y, x] = self.dummy_robber_tag

        harbor_tokens_list = [harbor.tag for harbor, count in self.harbor_tokens.items() for i in range(count) ]
        random.Random(self.seed+2).shuffle(harbor_tokens_list)

        for y,x in self.harbor_coords:
            harbor_tag = harbor_tokens_list.pop()
            arr[y,x] = harbor_tag


        # Generate coords of ownership
        return arr

    def game_start(self):
        # Clockwise order
        for player_tag, player in self.players.items():
            player.player_start()
        # Anticlockwise order
        for player_tag, player in reversed(self.players.items()):
            player.player_start()
        logging.debug("Game started!")

    def board_turn(self, player_tag=int):
        """
        roll dice and deliver resources.
        On roll 7,  discard then place robber
        """
        roll = self.roll()
        self.turn += 1
        current_player = self.players[player_tag]

        logging.debug("\n")
        logging.debug("<--Board-->")
        logging.debug(f"Rolled: {roll - 10}")

        if roll == 17:
            for player in self.players.values():
                player.discard_resources_turn()

            self.robber_action(current_player=current_player)

        else:
            self.deliver_resource(roll)

    def roll(self):
        return random.randint(1, 6) + random.randint(1, 6) + 10

    def robber_action(self, current_player: PlayerAI):
        current_player.action_space = current_player.get_action_space(
            situation=Action.ROBBER
        )
        action, attributes = current_player.pick_action()
        logging.debug(f"Robber Action: {action}, Attributes: {attributes}")
        current_player.perform_action(action, attributes, remove_resources=False)


def softmax(x):
    zero_indices = np.where(x == 0)

    shiftx = x - np.max(x, axis=0)
    exps = np.exp(shiftx)

    exps[zero_indices] = 0
    return exps / np.sum(exps, axis=0)


def relu(x):
    x[x < 0] = 0
    return x


def policy_forward(x, action_space, model):
    # forward pass: Take in board state, return probability of taking action [0,1,2,3]
    # Ne = Neurons in hidden state. As = Action Space
    z1 = model["W1"] @ x  # Ne x 483 * 483 x M = Ne x M
    a1 = relu(z1)  # Ne x M

    z2 = model["W2"] @ a1  # Ne x 483 * 483 x M = Ne x M
    a2 = relu(z2)  # Ne x M

    z3 = model["W3"] @ a2  # As x Ne * Ne x M = As x M
    m3 = np.multiply(z3, action_space)
    a3 = softmax(m3)  # As x M

    return (
        z1,
        a1,
        z2,
        a2,
        z3,
        a3,
    )  # return probability of taking action, and hidden state


class CatanAITraining:
    # Model_base
    # Hyperparameters

    H = 256  # number of hidden layer 1 neurons
    W = 256  # number of hidden layer 2 neurons
    batch_size = 100  # every how many episodes to do a param update?
    episodes = 100000
    learning_rate = 1e-5
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.999  # decay factor for RMSProp leaky sum of grad^2
    max_turn = 350
    player_type = [PlayerType.MODEL, PlayerType.MODEL]  # model | random
    player_count = 2  # 1 - 4
    # Stacking
    running_reward = None
    turn_list = []
    reward_list = []

    reward_matrices = {
        # "win_10": {"win": 10, "loss": 0},
        # "win_10_loss_10": {"win": 10, "loss": -10},
        # "win_50": {"win": 50, "loss": 0},
        # "win_50_loss_10": {"win": 50, "loss": -10},
        # "win_100": {"win": 100, "loss": 0},
        "test3": {"win": 350, "loss": 0},
        # "win_100_loss_100_big": {"win": 100, "loss": -100},
    }

    def __init__(self):
        # create dummy catan game to get action spaces
        catan = CatanAI(
            seed=1,
            player_type=self.player_type,
            player_matrices=[{"win": 10, "loss": 0}, {"win": 10, "loss": 0}],
            player_count=self.player_count,
            mode="multi",
            catan_training=self,
        )
        catan.board = catan.generate_board()

        self.D = len(
            catan.players[-9].prepro()
        )  # sum([len(player.prepro()) for key, player in catan.players.items()])
        # D = 23 * 21  # input dimensionality: 23 x 21 grid (483)
        self.model = {}
        self.model["W1"] = np.random.randn(self.H, self.D) / np.sqrt(
            self.D
        )  # "Xavier" initialization. Dim = H x 483

        self.types_of_actions = len(catan.generate_all_action_space())
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

    def plot_running_avg(
        self,
        y: list,
        fig_path: str,
        window=10,
    ):
        average_y = []
        for ind in range(len(y) - window + 1):
            average_y.append(np.mean(y[ind : ind + window]))

        plt.plot(average_y)
        plt.savefig(fig_path)
        plt.close()

    def play_game_existing_model(self, model_path):
        logging.basicConfig(
            format="%(message)s",
            level=logging.INFO,
            filename="running_2.txt",
            filemode="w",
            force=True,
        )
        parent = Path(__file__).parent.resolve()
        model_path = parent.joinpath(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.catan = CatanAI(
            seed=1,
            player_type=self.player_type,
            player_matrices=[{"win": 0, "loss": 0}, {"win": 0, "loss": 0}],
            player_count=self.player_count,
            mode="multi",
            catan_training=self,
        )

        self.catan.game_start()

        while self.catan.game_over is False and self.catan.turn < self.max_turn:
            logging.debug(f"Turn {self.catan.turn}")

            for player_tag, player in self.catan.players.items():
                # Phase 1: roll dice and get resources
                self.catan.board_turn(player_tag)
                # Phase 2: player performs actions
                player.player_preturn_debug()
                player.player_turn()
                player.recalc_points()
                player.player_posturn_debug()

    def training(self):
        for name, matrix in self.reward_matrices.items():
            # Run experiment
            Path.mkdir(Path(f"models/{name}"), exist_ok=True)
            logname = f"models/{name}/catan_game.txt"
            logging.basicConfig(
                format="%(message)s",
                level=logging.INFO,
                filename=logname,
                filemode="w",
                force=True,
            )
            for episode in range(self.episodes):
                logging.info(f"Episode {episode}")
                self.catan = CatanAI(
                    seed=episode,
                    player_type=self.player_type,
                    player_matrices=[matrix, matrix],
                    player_count=self.player_count,
                    mode="multi",
                    catan_training=self,
                )

                self.catan.game_start()

                while self.catan.game_over is False and self.catan.turn < self.max_turn:
                    logging.debug(f"Turn {self.catan.turn}")

                    for player_tag, player in self.catan.players.items():
                        # Phase 1: roll dice and get resources
                        self.catan.board_turn(player_tag)
                        # Phase 2: player performs actions
                        player.player_preturn_debug()
                        player.player_turn()
                        player.recalc_points()
                        player.player_posturn_debug()

                logging.info("End board state:")
                self.catan.print_board()

                # Episode Audit
                max_points = max(
                    [player.points for player in self.catan.players.values()]
                )
                for player_tag, player in self.catan.players.items():
                    if self.catan.turn == self.max_turn and player.points != max_points:
                        player.r_s[-1] += player.reward_matrix("loss")

                    player.player_episode_audit()

                logging.debug(
                    f"Game finished in {self.catan.turn} turns. Winner: {self.catan.winner}"
                )
                self.turn_list.append(self.catan.turn)
                self.reward_list.append(player.reward_sum)

                for player_tag, player in self.catan.players.items():
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
                if episode % 100 == 0:
                    Path.open(f"models/{name}/catan_model.pickle", "wb")
                    pickle.dump(
                        self.model, Path.open(f"models/{name}/catan_model.pickle", "wb")
                    )
                    pickle.dump(
                        self.turn_list,
                        Path.open(f"models/{name}/turn_list.pickle", "wb"),
                    )
                    pickle.dump(
                        self.reward_list,
                        Path.open(f"models/{name}/reward_list.pickle", "wb"),
                    )

                    self.plot_running_avg(
                        self.turn_list, Path(f"models/{name}/turn_list.jpg")
                    )
                    self.plot_running_avg(
                        self.reward_list, Path(f"models/{name}/reward_list.jpg")
                    )


if __name__ == "__main__":
    #
    CatanAITraining().training()
    # CatanAITraining().play_game_existing_model(
    #     "models/win_100_loss_50_100kep/catan_model.pickle"
    # )
