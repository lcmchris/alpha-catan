from __future__ import annotations
import random
import numpy as np
import logging
from enum import Enum
from collections import Counter


np.set_printoptions(edgeitems=30, linewidth=1000000)


def arr_to_tuple(arr):
    return [tuple(list) for list in arr.tolist()]


class Action(Enum):
    PASS: int = 0
    ROAD: int = 1
    SETTLEMENT: int = 2
    CITY: int = 3
    TRADE: int = 4
    DISCARD: int = 5
    ROBBER: int = 6


class Resource(Enum):
    BRICK = 1
    LUMBER = 2
    ORE = 3
    GRAIN = 4
    WOOL = 5
    DESERT = 6


class HARBOR(Enum):
    THREETOONE = -20
    BRICK = -21
    LUMBER = 22
    ORE = -23
    GRAIN = -24
    WOOL = -25


class Player:
    def __init__(
        self,
        catan: Catan,
        tag: int,
    ) -> None:
        self.catan = catan
        self.tag = tag
        self.resources = {
            # name, count, Start with 2 settlement + 2 roads
            Resource.BRICK: 0,  # brick
            Resource.LUMBER: 0,  # lumber
            Resource.ORE: 0,  # ore
            Resource.GRAIN: 0,  # grain
            Resource.WOOL: 0,  # wool
        }
        self.knight_cards = {
            "knight": 0,
            "victory_point": 0,
            "road_building": 0,
            "year_of_plenty": 0,
            "monopoly": 0,
        }
        self.settlements: list[tuple] = []
        self.roads: list[tuple] = []
        self.cities: list[tuple] = []
        self.action_space = None
        self.potential_settlement = []
        self.potential_road = []
        self.potential_trade = []
        self.points = 0
        self.actions_taken = [Action]
        self.reward_sum = 0

    def recalc_points(self):
        self.points = (
            len(self.settlements)
            + 2 * len(self.cities)
            + self.knight_cards["victory_point"]
        )
        logging.debug(f"Player {self.tag} has {self.points} points")

    def calculate_longest_road(self):
        max_length_all = 0
        for road in self.roads:
            max_length_all = max(
                self.depth_search_longest_road(
                    road, existing_roads=[road], backward_roads=[]
                ),
                max_length_all,
            )
        return max_length_all

    def depth_search_longest_road(
        self, road, existing_roads: list, backward_roads: list
    ):
        next_roads = [
            r
            for r in self.roads
            if r not in backward_roads
            and r not in existing_roads
            and r != road
            and abs(r[0] - road[0]) + abs(r[1] - road[1]) <= 3
        ]
        if len(next_roads) == 0:
            return len(existing_roads)

        assert len(next_roads) <= 4
        for next_road in next_roads:
            existing_roads.append(next_road)
            return self.depth_search_longest_road(road, existing_roads, next_roads)

    def get_action_space(self, situation: Action = None, attributes=None):
        """
        return a list of actions that the player can take
        """
        action_space_list = []
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            # only calculate potential actions if debug mode is on
            self.potential_settlement = self.get_potential_settlement()
            self.potential_road = self.get_potential_road()
            self.potential_trade = self.get_potential_trade()

        if situation == Action.SETTLEMENT:
            self.potential_settlement = self.get_potential_settlement()
            for settlement in self.potential_settlement:
                action_space_list.append((Action.SETTLEMENT, settlement))
        elif situation == Action.ROAD:
            self.potential_road = self.get_potential_road(attributes)
            for road in self.potential_road:
                action_space_list.append((Action.ROAD, road))
        elif situation == Action.DISCARD:
            for resource, count in self.resources.items():
                if count > 0:
                    action_space_list.append((Action.DISCARD, resource.value))
        elif situation == Action.ROBBER:
            for center_coord in self.catan.center_coords:
                action_space_list.append((Action.ROBBER, center_coord))
        elif situation is not None and not isinstance(situation, Action):
            raise Exception("Invalid situation")
        else:
            action_space_list.append((Action.PASS, None))

            if (
                self.resources[Resource.BRICK] >= 1
                and self.resources[Resource.LUMBER] >= 1
            ) and len(self.roads) < self.catan.max_properties["roads"]:
                self.potential_road = self.get_potential_road()
                for road in self.potential_road:
                    action_space_list.append((Action.ROAD, road))

            if (
                self.resources[Resource.BRICK] >= 1
                and self.resources[Resource.LUMBER] >= 1
                and self.resources[Resource.GRAIN] >= 1
                and self.resources[Resource.WOOL] >= 1
            ) and len(self.settlements) < self.catan.max_properties["settlements"]:
                self.potential_settlement = self.get_potential_settlement()
                for settlement in self.potential_settlement:
                    action_space_list.append((Action.SETTLEMENT, settlement))

            if (
                self.resources[Resource.ORE] >= 3
                and self.resources[Resource.GRAIN] >= 2
            ) and len(self.cities) < self.catan.max_properties["cities"]:
                self.potential_city = self.get_potential_city()
                for city in self.potential_city:
                    action_space_list.append((Action.CITY, city))

            self.potential_trade = self.get_potential_trade()
            for trade in self.potential_trade:
                action_space_list.append((Action.TRADE, trade))

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

        best_action = np.argmax(possible_actions_arr)
        return best_action

    def discard_resources(self, attributes):
        self.resources[Resource(attributes)] -= 1

    def discard_resources_turn(self):
        pass

    def owned_harbors(self) -> list:
        owned_harbors = set()
        for spot in self.settlements + self.cities:
            if spot in self.catan.harbor_ownership:
                owned_harbors.add(HARBOR(self.catan.harbor_ownership[spot]))
        return list(owned_harbors)

    def add_all_other_resource(self, resource: Resource):
        pot_trades_for_resource = set()
        for j in range(1, 6):
            if j != resource.value:
                pot_trades_for_resource.add((resource.value, j))
        return pot_trades_for_resource

    def get_potential_trade(self) -> list[tuple[int, int]]:
        # returns a list of tuples of the form (resource, resource)
        set_of_potential_trades = set()

        owned_harbors = self.owned_harbors()
        for harbor in owned_harbors:
            if harbor == HARBOR.THREETOONE:
                for resource, count in self.resources.items():
                    if count >= 3:
                        for j in range(1, 6):
                            if j != resource.value:
                                set_of_potential_trades.add((resource.value, j))

            elif harbor == HARBOR.BRICK:
                if self.resources[Resource.BRICK] >= 2:
                    set_of_potential_trades.add(
                        self.add_all_other_resource(Resource.BRICK)
                    )
            elif harbor == HARBOR.LUMBER:
                if self.resources[Resource.LUMBER] >= 2:
                    set_of_potential_trades.add(
                        self.add_all_other_resource(Resource.LUMBER)
                    )
            elif harbor == HARBOR.ORE:
                if self.resources[Resource.ORE] >= 2:
                    set_of_potential_trades.add(
                        self.add_all_other_resource(Resource.ORE)
                    )
            elif harbor == HARBOR.GRAIN:
                if self.resources[Resource.GRAIN] >= 2:
                    set_of_potential_trades.add(
                        self.add_all_other_resource(Resource.GRAIN)
                    )
            elif harbor == HARBOR.WOOL:
                if self.resources[Resource.WOOL] >= 2:
                    set_of_potential_trades.add(
                        self.add_all_other_resource(Resource.WOOL)
                    )
            else:
                raise Exception("Invalid harbor")

        for resource, count in self.resources.items():
            if count >= 4:
                for j in range(1, 6):
                    if j != resource.value:
                        set_of_potential_trades.add((resource.value, j, 4))

        list_of_potential_trades = list(set_of_potential_trades)
        return list_of_potential_trades

    def perform_action(self, action: Action, attributes, remove_resources=True):
        if action == Action.PASS:
            pass
        elif action == Action.ROAD:
            self.build_road(attributes, remove_resources)
        elif action == Action.SETTLEMENT:
            self.build_settlement(attributes, remove_resources)
        elif action == Action.CITY:
            self.build_city(attributes)
        elif action == Action.TRADE:
            self.trade_with_bank(attributes)
        elif action == Action.DISCARD:
            self.discard_resources(attributes)
        elif action == Action.ROBBER:
            self.place_robber(attributes)
        else:
            raise ValueError("action not in action space")

    def build_settlement(self, coords: tuple, remove_resources=True):
        # examing where I can build a settlement
        assert coords not in self.settlements, "Building in the same spot!"

        logging.debug(f"Built settlement at : {coords}")

        self.catan.board[coords[0], coords[1]] = self.tag
        self.settlements.append(coords)

        if remove_resources:
            # removing resources
            self.update_resources_settlement()

    def update_resources_settlement(self):
        self.resources[Resource.BRICK] -= 1
        self.resources[Resource.WOOL] -= 1
        self.resources[Resource.LUMBER] -= 1
        self.resources[Resource.WOOL] -= 1

    def build_road(self, coords: tuple, remove_resources=True):
        assert coords not in self.roads, "Building in the same spot!"

        logging.debug(f"Built road at : {coords}")

        self.catan.board[coords[0], coords[1]] = self.tag
        self.roads.append(coords)
        if remove_resources:
            self.update_resources_settlement()

            if len(self.roads) >= 5:
                self.longest_road = self.calculate_longest_road()

    def update_resources_road(self):
        self.resources[Resource.BRICK] -= 1
        self.resources[Resource.LUMBER] -= 1

    def build_city(self, coords):
        assert coords not in self.cities, "Building in the same spot!"

        logging.debug(f"Built city at : {coords}")

        self.catan.board[coords[0], coords[1]] = self.tag - 10
        self.cities.append(coords)
        self.settlements.remove(coords)

        self.update_resources_city()

    def update_resources_city(self):
        self.resources[Resource.ORE] -= 3
        self.resources[Resource.GRAIN] -= 2

    def trade_with_bank(self, resource_in_resource_out: tuple):
        resource_in = Resource(resource_in_resource_out[0])
        resource_out = Resource(resource_in_resource_out[1])


        self.resources[resource_in] -= count
        self.resources[resource_out] += 1
        logging.debug(f"Trading {resource_in} for {resource_out} at {count}")

    def place_robber(self, coords: tuple):
        """Place robber at coords and steal resources from opponent"""
        self.catan.board[self.catan.board == self.catan.robber_tag] = (
            self.catan.dummy_robber_tag
        )

        self.catan.board[coords] = self.catan.robber_tag
        logging.debug(f"Placed robber at : {coords}")            
        (y, x) = coords

        list_of_potential_owners = [
            [y - 4, x],
            [y + 2, x],
            [y, x + 2],
            [y, x - 2],
            [y - 2, x + 2],
            [y - 2, x - 2],
        ]
        for y, x in list_of_potential_owners:
            potential_opponent = self.catan.board[y, x]
            if (
                potential_opponent in self.catan.players
                and potential_opponent != self.tag
            ):
                opponent = potential_opponent
                if any(
                    [
                        player_resoruce > 0
                        for player_resoruce in self.catan.players[
                            opponent
                        ].resources.values()
                    ]
                ):
                    robbed_resource = random.choice(
                        self.catan.players[opponent].available_resources()
                    )
                    self.catan.players[opponent].resources[robbed_resource] -= 1
                    self.catan.players[self.tag].resources[robbed_resource] += 1
                    logging.debug(f"Robber stole {robbed_resource} from {opponent}")

        assert len(np.where(self.catan.board == self.catan.robber_tag)[0]) == 1

    def available_resources(self) -> list[Resource]:
        available_resources = []
        for resource, count in self.resources.items():
            if count > 0:
                available_resources.append(resource)

        return available_resources

    def update_resources(self, resource_tag: int, number: int):
        self.resources[Resource(resource_tag)] += number

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
                            if self.catan.board[_y2, _x2] not in self.catan.player_tags:
                                return_list_road.append((_y, _x))

        return list(set(return_list_sett + return_list_road))

    def get_potential_city(self):
        return self.settlements

    def player_turn(self):
        pass

    def player_start(self):
        pass

    def player_preturn_debug(self):
        logging.debug(f"<-- Player {self.tag} -->")

    def player_posturn_debug(self):
        logging.debug(f"<-- Player {self.tag} -->")
        logging.debug(f"Resources: {self.resources}")
        logging.debug(f"Points: {self.points}")
        logging.debug(f"Potential sett: {self.potential_settlement}")
        logging.debug(f"Potential road: {self.potential_road}")
        logging.debug(f"Potential trade: {self.potential_trade}")

    def player_episode_audit(self):
        logging.info(f"<-- Player {self.tag} -->")

        unique_actions = Counter(self.actions_taken)
        logging.info(
            f"Actions taken: {dict(zip(unique_actions.keys(), unique_actions.values()))}"
        )
        logging.info(f"Settlements : {self.settlements}")
        logging.info(f"City : {self.cities}")
        logging.info(f"Roads : {self.roads}")
        logging.info(f"End Resources: {self.resources}")
        self.reward_sum = round(np.sum(np.vstack(self.r_s)))
        logging.info(f"Reward sum: {self.reward_sum}")
        logging.info(f"Points: {self.points} ")


class Catan:
    def __init__(self) -> None:
        self.seed = 0
        self.robber_tag = 50
        self.dummy_robber_tag = 52
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
        self.resources_tag = {"brick": 1, "lumber": 2, "ore": 3, "grain": 4, "wool": 5}
        self.resource_card_count = {
            # name, count, tile_count
            Resource.BRICK: 19,
            Resource.LUMBER: 19,
            Resource.ORE: 19,
            Resource.GRAIN: 19,
            Resource.WOOL: 19,
            Resource.DESERT: 0,
        }
        self.resource_tokens = {
            Resource.BRICK: 3,  # brick
            Resource.LUMBER: 4,  # lumber
            Resource.ORE: 3,  # ore
            Resource.GRAIN: 4,  # grain
            Resource.WOOL: 4,  # wool
            Resource.DESERT: 1,  # desert
        }
        self.number_tokens = {
            2: 1,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 1,
        }

        self.harbor_tokens = {
            HARBOR.THREETOONE.value: 4,  # 3 to 1
            HARBOR.BRICK.value: 1,  # brick
            HARBOR.LUMBER.value: 1,  # lumber
            HARBOR.ORE.value: 1,  # ore
            HARBOR.GRAIN.value: 1,  # grain
            HARBOR.WOOL.value: 1,  # wool
        }
        self.harbor_coords = {
            (2, 6): [(0, 2), (2, 0)],
            (2, 14): [(2, 0), (0, -2)],
            (6, 20): [(2, 0), (0, -2)],
            (9, 3): [(-1, 1), (1, 1)],
            (13, 23): [(-1, -1), (1, -1)],
            (17, 3): [(-1, 1), (1, 1)],
            (20, 20): [(-2, 0), (0, -2)],
            (24, 6): [(-2, 0), (0, 2)],
            (24, 14): [(-2, 0), (0, -2)],
        }
        self.harbor_ownership: dict[tuple, int]

        # fmt: off
        self.center_coords = [
            (5, 8),(5, 12),(5, 16),
            (9, 6),(9, 10),(9, 14),(9, 18), 
            (13, 4),(13, 8),(13, 12),(13, 16),(13, 20),
            (17, 6),(17, 10),(17, 14),(17, 18),
            (21, 8),(21, 12),(21, 16),
        ]
        # fmt: on
        self.game_over = False
        self.winner = None
        self.turn = 0
        # fmt: off
        self.empty_board =  np.array(
        [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,3,0,-1,0,0,0,-1,0,3,0,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0,0,0],
        [0,0,0,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,0,0,0],
        [0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,3,0,0,0,0],
        [0,0,0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0],
        [0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0],
        [0,0,0,3,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,0],
        [0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0],
        [0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0],
        [0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0],
        [0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,3,0],
        [0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0],
        [0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0],
        [0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0],
        [0,0,0,3,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,0],
        [0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0],
        [0,0,0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0],
        [0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,3,0,0,0,0],
        [0,0,0,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,-2,0,0,0,0,0,0],
        [0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,-1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,3,0,-1,0,0,0,-1,0,3,0,-1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ]
    )
        # fmt: on

        self.board = self.generate_board()
        self.players = self.generate_players()
        self.player_tags = [-9, -19, -8, -18]

        self.all_road_spots = self.get_road_spots()
        self.all_settlement_spots = self.get_settment_spots()
        self.base_action_space = self.generate_all_action_space()

        self.unique_tags = (
            [-2, -1]
            + self.player_tags
            + list(self.harbor_tokens)
            + list(self.resource_tokens.keys())
            + [key + 10 for key in self.number_tokens.keys()]
            + [self.robber_tag, self.dummy_robber_tag]
        )

    def get_settment_spots(self):
        return arr_to_tuple(np.argwhere(self.board == -1))

    def get_road_spots(self):
        return arr_to_tuple(np.argwhere(self.board == -2))

    def get_trades(self):
        return [(x, y) for x in range(1, 6) for y in range(1, 6) if x != y]

    def get_discards(self):
        return [1, 2, 3, 4, 5]

    def get_robber_spots(self):
        return self.center_coords

    def generate_all_action_space(self) -> list[int, tuple]:
        """
        0, 0 :: Dim = 1
        1, {list of all settlements} :: Dim = 2*(3+4+4+5+5+6) = 54
        2, {list of all road} :: Dim = 72
        3, {list of all cities} :: Dim = 2*(3+4+4+5+5+6) = 54
        4, {list of all trades} :: Dim = (4 * 5) = 20
        5, {list of discards} :: Dim = (5 * 1) = 5
        6, {list of robber places} :: Dim = (3+4+5+4+3) = 19
        """

        action_space = {
            Action.PASS: [None],  # nothing
            Action.ROAD: self.get_road_spots(),  # road
            Action.SETTLEMENT: self.get_settment_spots(),  # settlement
            Action.CITY: self.get_settment_spots(),  # city
            Action.TRADE: self.get_trades(),  # trade
            Action.DISCARD: self.get_discards(),  # discards
            Action.ROBBER: self.get_robber_spots(),  # robbers
        }
        flatten_action_space = [
            (action_index, potential_action)
            for action_index, potential_actions in action_space.items()
            for potential_action in potential_actions
        ]

        return flatten_action_space

    def generate_players(self, *args, **kwargs) -> dict[str, Player]:
        pass

    def print_board(self, debug=True):
        board = self.board.copy()
        board = board[2 : board.shape[0] - 2, 2 : board.shape[1] - 2]
        zero_tmp = 97
        board[board == 0] = zero_tmp
        # board[board == 50] = zero_tmp
        for i in range(12, 22 + 1):
            board[board == i] = i - 10

        board = np.insert(board, 0, np.arange(2, board.shape[1] + 2), axis=0)
        board = np.insert(board, 0, np.arange(1, board.shape[0] + 1), axis=1)

        board_string = np.array2string(board)
        board_string = board_string.replace(f"{zero_tmp}", "  ")
        print(board_string)

    def generate_board(self):
        resource_list = [
            resource.value
            for resource, count in self.resource_tokens.items()
            for x in range(count)
        ]
        number_list = [
            key + 10 for key, value in self.number_tokens.items() for x in range(value)
        ]

        def pick_and_pop(__list: list) -> int:
            picked = 0
            value = __list[picked]
            __list.pop(picked)
            return value

        """
        Grid of 5 hexagons CENTER = resource number, CENTER+1 = resource type
        21 x 23

        """
        # fmt: off
        arr = self.empty_board

        # Shuffle based on seed, off by one
        random.Random(self.seed).shuffle(resource_list)
        random.Random(self.seed+1).shuffle(number_list)

        for y, x in self.center_coords:
            resource = pick_and_pop(resource_list)

            if resource == 6:
                number = 0
            else:
                number = pick_and_pop(number_list)
            
            
            arr[y - 1, x] = resource
            arr[y + 1, x] = number

            if resource == 6:
                arr[y, x] = self.robber_tag  # robber reference
            else:
                arr[y, x] = self.dummy_robber_tag

        harbor_toknes_list = [token for token, count in self.harbor_tokens.items() for i in range(count) ]

        for y,x in self.harbor_coords:
            harbor = pick_and_pop(harbor_toknes_list)
            arr[y,x] = harbor


        # Generate coords of ownership
        self.harbor_ownership = {
            (harbor[0] + pos[0], harbor[1] + pos[1]): arr[harbor]
            for harbor, positions in self.harbor_coords.items()
            for pos in positions
        }
        return arr

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
            resource_type = Resource(self.board[y - 2, x])
            logging.info(f"Resource type: {resource_type}")

            if resource_type == Resource.DESERT:
                continue
            else:
                for potential_y, potential_x in list_of_potential_owners:
                    owner_tag = self.board[potential_y, potential_x]

                    if owner_tag != -1:
                        resource_num = 1 if owner_tag >= -10 else 2
                        owner_tag = owner_tag if owner_tag >= -10 else owner_tag + 10
                        self.add_resource(owner_tag, resource_type, resource_num)

    def add_resource(self, owner_tag, resource_type: Resource, resource_num: int):
        self.players[owner_tag].resources[resource_type] += resource_num
        logging.debug(
            f"Player {self.players[owner_tag].tag} received {resource_num} of {resource_type}({resource_type})"
        )
