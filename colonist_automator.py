import time
from playwright.sync_api import sync_playwright, Locator
import pathlib
import cv2 as cv2
import numpy as np
from catan_game import Catan, Player
from catan_ai import CatanAITraining, PlayerAI, PlayerType
import pickle
from catan_ai import relu, softmax
import logging

start_time = time.time()


class ColonistIOAutomator(Catan):
    """
    Colonist automator automates a game on the colonist.io platform using a pre-build model from catan_ai.

    """

    starting_coords = [80, 80]
    x_spacing = 50
    y_1_spacing = 30
    y_2_spacing = 60
    selection_spacing = 45
    nodes_count = [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]
    road = y_1_spacing / 2
    match_method = cv2.TM_CCOEFF_NORMED
    tile_count = 2  # 2 tiles per number
    box_size = (45, 85)
    # fmt: off
    box_coords = {
        ( 90,210): (5, 8),( 90,310): (5, 12),(90,410): (5, 16),
        ( 180,160): (9, 6),( 180,260): (9, 10),( 180,360): (9, 14),( 180,460): (9, 18),
        ( 270,110): (13, 4),(270,210): (13, 8),( 270,310): (13, 12),( 270,410): (13, 16),(270,510): (13, 20),
        ( 360,160): (17, 6),(360,260): (17, 10),( 360,360): (17, 14),( 360,460): (17, 18),
        ( 470,210): (21, 8),( 470,310): (21, 12),(470,410): (21, 16)
        }

    # fmt: on
    selection_spacing = 45

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.n_action = 0
        self.pr = sync_playwright().start()
        self.page = self.create_page()
        self.canvas, self.players, self.players_name = self.create_1v1_game()

        self.canvas_img: cv2.typing.MatLike

        self.take_canvas_img()
        self.set_board_numbers_ocr()
        self.set_board_tiles_ocr()

        self.player_automator: PlayerAutomator = self.players[-9]
        self.player_automator.embargo_opponent()

        self.model = self.load_model_pickle(model_path)

        self.settlement_pxl_to_coords, self.settlement_coords_to_pxl = (
            self.generate_settlement_pxl_mapping()
        )
        self.road_pxl_to_coords, self.road_coords_to_pxl = (
            self.generate_road_pxl_mapping()
        )

        self.last_messages = self.page.query_selector_all(".message-post")

        self.play_game()

    def load_model_pickle(self, rel_model_path: str):
        parent = pathlib.Path(__file__).parent.resolve()
        model_path = parent.joinpath(rel_model_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def generate_settlement_pxl_mapping(self):
        settlement_pxl_to_coords = {}
        settlement_coords_to_pxl = {}

        for spt_y, spt_x in self.all_settlement_spots:
            nomarlised_y = (spt_y - 2) / 2
            nomarlised_x = (spt_x - 2) / 2
            y_pxl = (
                nomarlised_y // 2 * (self.y_2_spacing + self.y_1_spacing)
                + (0 if nomarlised_y % 2 == 0 else self.y_1_spacing)
                + self.starting_coords[0]
            )
            x_pxl = nomarlised_x * (self.x_spacing) + self.starting_coords[1]
            settlement_pxl_to_coords[(y_pxl, x_pxl)] = (spt_y, spt_x)
            settlement_coords_to_pxl[(spt_y, spt_x)] = (y_pxl, x_pxl)

        return settlement_pxl_to_coords, settlement_coords_to_pxl

    def generate_road_pxl_mapping(self):
        road_pxl_to_coords = {}
        road_coords_to_pxl = {}

        for spt_y, spt_x in self.all_road_spots:
            nomarlised_y = (spt_y - 3) / 2
            nomarlised_x = (spt_x - 3) / 2
            y_pxl = (
                (
                    self.y_2_spacing / 2 + self.y_1_spacing / 2
                    if nomarlised_y % 2 == 1
                    else 0
                )
                + nomarlised_y // 2 * (self.y_2_spacing + self.y_1_spacing)
                + self.y_1_spacing / 2
                + self.starting_coords[0]
            )
            x_pxl = (
                nomarlised_x * (self.x_spacing) + self.x_spacing / 2
            ) + self.starting_coords[1]
            road_pxl_to_coords[(y_pxl, x_pxl)] = (spt_y, spt_x)
            road_coords_to_pxl[(spt_y, spt_x)] = (y_pxl, x_pxl)
        return road_pxl_to_coords, road_coords_to_pxl

    def create_page(self):
        browser = self.pr.chromium.launch(headless=False)
        context = browser.new_context()
        return context.new_page()

    def create_1v1_game(self):
        self.page.goto("https://colonist.io/")
        self.page.get_by_label("Consent", exact=True).click()
        self.page.click("text=Lobby")
        self.page.click("text=Create Room")

        self.page.locator("#add-bot-button").nth(1).click()
        self.page.locator("#botspeed_settings_right_arrow").click()
        self.page.wait_for_timeout(1000)

        players, players_name = self.set_players()

        self.page.click("text=Start Game")

        canvas = self.page.locator("canvas").nth(0)
        self.page.wait_for_url("https://colonist.io/#*")
        self.page.wait_for_load_state("domcontentloaded")
        self.page.wait_for_timeout(3000)

        return canvas, players, players_name

    def take_canvas_img(
        self,
    ) -> cv2.typing.MatLike:
        screenshot_bytes = self.canvas.screenshot(
            path=f"game_images/screenshot_{self.n_action}.png", animations="disabled"
        )
        self.canvas_img = cv2.imdecode(
            np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR
        )

    def set_players(self) -> tuple[dict[str, Player], dict[str, int]]:
        players_query = self.page.locator("span.room_player_username").all()

        players_dict = {}
        players_name_dict = {}

        for i in range(len(players_query)):
            player_tag = i - 9
            if players_query[i].inner_text() != "Player":
                if " (You)" in players_query[i].inner_text():
                    player_name = players_query[i].inner_text().replace(" (You)", "")
                    players_dict[player_tag] = PlayerAutomator(
                        catan=self, tag=player_tag, player_type=PlayerType.MODEL
                    )
                    players_name_dict[player_name] = player_tag
                else:
                    player_name = players_query[i].inner_text()
                    players_dict[player_tag] = Player(catan=self, tag=player_tag)
                    players_name_dict[player_name] = player_tag

        assert len(players_dict) == 2

        return players_dict, players_name_dict

    def play_game(self):
        """

        In vitro state vs actual state

        Which should I trust more?
        Actual state is more important as it is real

        > Is there anypoint updating the state then? Probably not


        Options:
        - Polling every 3 seconds of new state
            Check for:
            - My turn?

        """
        time_past = 0
        n_actions = 0
        while True:
            time_past += 1
            self.page.wait_for_timeout(2000)
            self.get_game_state()

            if self.is_my_turn():
                if len(self.player_automator.settlements) < 2:
                    self.player_automator.player_start()
                else:
                    self.roll_dice()
                    action = None
                    while action != 0:
                        # get action space, pick random action, perform action. Repeat until all actions are done or hits nothing action.
                        self.player_automator.action_space = (
                            self.player_automator.get_action_space()
                        )
                        action, attributes = self.player_automator.pick_action()
                        logging.debug(f"Action: {action}, Attributes: {attributes}")
                        self.player_automator.perform_action(action, attributes)
                        if action != 0:
                            self.get_game_state()

                    self.end_turn()

                n_actions += 1

    def get_game_state(self):
        self.tally_new_messages()
        self.take_canvas_img()
        self.loop_templates_ocr()

    def is_game_start(self, n_actions):
        return n_actions < 2

    def is_my_turn(self):
        pxl_at_coord = self.canvas_img[600, 680]
        if np.array_equal(pxl_at_coord, np.array([242, 248, 250])):
            return True
        else:
            return False

    def roll_dice(self):
        self.page.keyboard.press("Space")

    def end_turn(self):
        self.page.keyboard.press("Space")

    def tally_new_messages(
        self,
    ):
        messages = self.page.locator(".message-post").all()
        last_msg_cnt = len(self.last_messages)
        new_msg_cnt = len(messages)

        if new_msg_cnt > last_msg_cnt:
            new_messages = messages[last_msg_cnt:new_msg_cnt]
            for message in new_messages:
                self.parse_message(message)

        self.last_messages = messages

    def parse_message(self, message: Locator):
        player_name = self.player_from_msg(message=message)
        try:
            player_tag = self.players_name[player_name]
        except KeyError:
            pass

        if "rolled" in message.inner_text():
            dice_rolled = self.query_rolled(message)
            if dice_rolled == 7 and sum(self.player_automator.resources.values()) > 7:
                self.player_automator.discard_resources_model()

        elif "placed a" in message.inner_text():
            if (
                self.is_settlement(message)
                and len(self.players[player_tag].settlements) > 2
            ):
                self.players[player_tag].update_resources_settlement()
            elif self.is_road(message) and len(self.players[player_tag].roads) > 2:
                self.players[player_tag].update_resources_road()
            elif self.is_city(message):
                self.players[player_tag].update_resources_city()
            else:
                print(message)

        elif (
            "got" in message.inner_text()
            or "received starting resources" in message.inner_text()
        ):
            for resource, r_tag in self.resources_tag.items():
                num_resources = len(message.get_by_alt_text(resource).all())
                if num_resources > 0:
                    self.players[player_tag].update_resources(r_tag, num_resources)

        elif "gave bank" in message.inner_text() and "and took" in message.inner_text():
            for resource, r_tag in self.resources_tag.items():
                all = message.get_by_alt_text(resource).all()

    def query_rolled(self, message: Locator) -> list[int, int]:
        dices = ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "dice_6"]
        dice_rolled = []
        for dice in dices:
            if (matchs := message.get_by_alt_text(dice).all()) is not None:
                for _ in matchs:
                    dice_rolled.append(dice)
        assert len(dice_rolled) == 2
        return dice_rolled

    def player_from_msg(self, message: Locator):
        for player_name in self.players_name:
            if player_name in message.inner_text():
                return player_name

    def is_settlement(self, message: Locator):
        return message.get_by_alt_text("settlement").is_visible()

    def is_road(self, message: Locator):
        return message.get_by_alt_text("road").is_visible()

    def is_city(self, message: Locator):
        return message.get_by_alt_text("city").is_visible()

    def get_center(self, coords, h, w):
        return tuple(map(sum, zip(coords, (h / 2, w / 2))))

    def match_coords(self, pxl, mapping: dict, size=(25, 25)):
        for y, x in mapping.keys():
            if (
                x - size[0] < pxl[1] < x + size[0]
                and y - size[1] < pxl[0] < y + size[1]
            ):
                return mapping[(y, x)]

        raise KeyError

    def set_board_numbers_ocr(self):
        canvas_img = self.canvas_img.copy()

        for tile, tile_count in self.number_tokens.items():
            template = cv2.imread(
                f"image-matching/tile_{str(tile)}.png", cv2.IMREAD_COLOR
            )
            h, w = template.shape[:2]

            for count in range(tile_count):
                res = cv2.matchTemplate(canvas_img, template, self.match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                canvas_img[
                    max_loc[1] - 1 : max_loc[1] + h + 1,
                    max_loc[0] - 1 : max_loc[0] + w + 1,
                ] = 0

                for attempt in range(3):
                    try:
                        loc_yx = (max_loc[1], max_loc[0])
                        coord = self.match_coords(
                            self.get_center(loc_yx, h, w),
                            mapping=self.box_coords,
                            size=self.box_size,
                        )
                    except KeyError:
                        self.page.wait_for_timeout(3000)
                        self.take_canvas_img()
                    else:
                        break

                if coord is None:
                    raise Exception("No match found")

                else:
                    (y, x) = coord

                assert coord in self.center_coords
                self.board[y + 1, x] = tile

    def set_board_tiles_ocr(self):
        canvas_img = self.canvas_img.copy()

        for tile, tile_count in self.resource_tokens.items():
            template = cv2.imread(
                f"image-matching/type_{str(tile.value)}.png", cv2.IMREAD_COLOR
            )
            h, w = template.shape[:2]

            for count in range(tile_count):
                res = cv2.matchTemplate(canvas_img, template, self.match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                canvas_img[
                    max_loc[1] - 1 : max_loc[1] + h + 1,
                    max_loc[0] - 1 : max_loc[0] + w + 1,
                ] = 0
                loc_yx = (max_loc[1], max_loc[0])
                coord = self.match_coords(
                    self.get_center(loc_yx, h, w),
                    mapping=self.box_coords,
                    size=self.box_size,
                )
                (y, x) = coord
                assert coord in self.center_coords
                self.board[y - 1, x] = tile.value

    def loop_templates_ocr(
        self,
    ) -> list[int, int]:
        """Get the coordinates of the current board"""

        ## Red
        for player in ["red", "blue"]:
            for match_type in ["city", "settlement", "road"]:
                img = self.canvas_img.copy()
                if player == "red":
                    player_tag = -9
                elif player == "blue":
                    player_tag = -8

                if match_type == "city":
                    count = 1
                    player_tag -= 10
                    mapping = self.settlement_pxl_to_coords

                elif match_type == "settlement":
                    count = 1
                    mapping = self.settlement_pxl_to_coords
                elif match_type == "road":
                    count = 3
                    mapping = self.road_pxl_to_coords

                for i in range(count):
                    template = cv2.imread(
                        f"image-matching/{match_type}_{player}_{i}.png",
                        cv2.IMREAD_COLOR,
                    )
                    self.update_board_ocr(
                        img=img,
                        template=template,
                        player_tag=player_tag,
                        mapping=mapping,
                    )

    def update_board_ocr(
        self,
        img: cv2.typing.MatLike,
        template: cv2.typing.MatLike,
        player_tag: int,
        mapping: dict,
    ) -> list[int, int]:
        h, w = template.shape[:2]
        while True:
            result = cv2.matchTemplate(img, template, self.match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val < 0.95:
                break
            loc_yx = (max_loc[1], max_loc[0])

            logging.info(f"Found match {template} at {loc_yx} , player {player_tag}")

            coord = self.match_coords(self.get_center(loc_yx, h, w), mapping=mapping)
            (y, x) = coord
            self.board[y, x] = player_tag
            img[
                max_loc[1] - 15 : max_loc[1] + h + 15,
                max_loc[0] - 15 : max_loc[0] + w + 15,
            ] = 0

            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


class PlayerAutomator(PlayerAI):
    def __init__(self, catan: ColonistIOAutomator, tag, player_type):
        super().__init__(catan=catan, tag=tag, player_type=PlayerType.MODEL)
        self.catan: ColonistIOAutomator

    def policy_forward(self, x, mask):
        # forward pass: Take in board state, return probability of taking action [0,1,2,3]
        # Ne = Neurons in hidden state. As = Action Space
        z1 = self.catan.model["W1"] @ x  # Ne x 483 * 483 x M = Ne x M
        a1 = relu(z1)  # Ne x M

        z2 = self.catan.model["W2"] @ a1  # Ne x 483 * 483 x M = Ne x M
        a2 = relu(z2)  # Ne x M

        z3 = self.catan.model["W3"] @ a2  # As x Ne * Ne x M = As x M
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

    def pick_action(self):
        x = self.prepro()  # append board state
        z1, a1, z2, a2, z3, a3 = self.policy_forward(
            x=x, action_space=self.action_space
        )

        action_idx = np.argmax(a3)  # pick action with highest
        action, attributes = self.action_idx_to_action_tuple(action_idx)

        return action, attributes

    def perform_action(self, action, attributes, situation=False):
        if action == 0:
            pass
        elif action == 1:
            self.build_road(attributes)
        elif action == 2:
            self.build_settlement(attributes)
        elif action == 3:
            self.build_city(attributes)
        elif action == 4:
            self.trade_with_bank(attributes)
        elif action == 5:
            self.discard_resources(attributes)
        else:
            raise ValueError("action not in action space")

    def click_buy(
        self,
        mapping: dict,
        coords: tuple,
    ):
        self.catan.canvas.click(
            position={
                "y": mapping[coords][0],
                "x": mapping[coords][1],
            }
        )
        self.catan.page.wait_for_timeout(1000)
        self.catan.canvas.click(
            position={
                "y": mapping[coords][0] - self.catan.selection_spacing,
                "x": mapping[coords][1],
            }
        )

        self.catan.page.wait_for_timeout(1000)

    def build_settlement(
        self,
        coords: tuple,
    ):
        if coords not in self.catan.all_settlement_spots:
            raise ValueError(f"Incorrect buying coords: {coords}")
        mapping = self.catan.settlement_coords_to_pxl
        self.click_buy(mapping, coords)
        self.settlements.append(coords)

    def build_road(self, coords: tuple):
        if coords in self.catan.all_road_spots:
            mapping = self.catan.road_coords_to_pxl
            self.click_buy(mapping, coords)
        else:
            raise ValueError(f"Incorrect buying coords: {coords}")

        self.roads.append(coords)

    def embargo_opponent(self):
        opponent_icon_pxl = (560, 720)
        self.catan.canvas.click(
            position={
                "y": opponent_icon_pxl[0],
                "x": opponent_icon_pxl[1],
            }
        )
        self.catan.page.click("text=Embargo Player")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        # filename=logname,
        # filemode="w",
    )
    colonistUI = ColonistIOAutomator(model_path="catan_model.pickle")
