import re
import time
from playwright.sync_api import sync_playwright, Locator
import pathlib
import cv2 as cv2
import numpy as np
from catan_game import DevelopmentCard, Catan, Player, Resource, Action
from catan_ai import PlayerAI, PlayerType, policy_forward
import pickle
import logging
from dotenv import dotenv_values
import warnings

config = dotenv_values(".env")
start_time = time.time()


class ColonistIOAutomator(Catan):
    """
    Colonist automator automates a game on the colonist.io platform using a pre-build model from catan_ai.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.selection_spacing = 45
        self.starting_coords = [85, 85]
        self.x_spacing = 53.75
        self.y_1_spacing = 31
        self.y_2_spacing = 62
        self.selection_spacing = 45
        self.nodes_count = [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]
        self.road = self.y_1_spacing / 2
        self.match_method = cv2.TM_CCOEFF_NORMED
        self.tile_count = 2  # 2 tiles per number
        self.box_size = (90, 90)
        self.crop_top = (0, 660, None, None)
        self.crop_bottom = (660, 735, 0, 420)
        self.crop_dev_action_row = (560, 605, 0, 150)
        # fmt:

        self.center_pxl_to_coords, self.center_coords_to_pxl = (
            self.generate_center_pxl_mapping()
        )
        self.robber_coords_to_pxl = {
            coords: (pxl[0] - 30, pxl[1])
            for coords, pxl in self.center_coords_to_pxl.items()
        }
        self.settlement_pxl_to_coords, self.settlement_coords_to_pxl = (
            self.generate_settlement_pxl_mapping()
        )
        self.road_pxl_to_coords, self.road_coords_to_pxl = (
            self.generate_road_pxl_mapping()
        )

        self.confirm_action_pxl = (640, 345)

        self.harbor_coords_pxl = {
            # clockwise from top left
            (30, 175): (2, 6),
            (30, 380): (2, 14),
            (120, 530): (6, 20),
            (295, 635): (13, 23),
            (470, 533): (20, 20),
            (560, 380): (24, 14),
            (560, 175): (24, 6),
            (385, 73): (17, 3),
            (215, 73): (9, 3),
        }
        # fmt: on
        self.n_action = 0
        self.pr = sync_playwright().start()
        self.browser = self.create_browser()
        self.create_page()

        self.canvas, self.players, self.players_name = self.create_1v1_game()

        self.canvas_img: cv2.typing.MatLike
        self.take_canvas_img()
        self.set_board_numbers_ocr()
        self.set_board_tiles_ocr()
        self.set_harbor_ocr()

        self.player_automator: PlayerAutomator = self.players[-9]

        self.player_automator.embargo_opponent()
        self.last_messages = self.page.query_selector_all(".message-post")

        self.model = self.load_model_pickle(model_path)
        self.play_game()

    def create_browser(self):
        return self.pr.webkit.launch(
            headless=False,
        )

    def create_page(self):
        if pathlib.Path("state.json").exists():
            self.context = self.browser.new_context(storage_state="state.json")
            self.page = self.context.new_page()
            self.page.goto("https://colonist.io/")
            self.page.locator("body").click(position={"x": 10, "y": 10})
            self.page.wait_for_timeout(1000)
            self.page.locator("body").click(position={"x": 10, "y": 10})

        else:
            self.context = self.browser.new_context(
                viewport={"width": 1280, "height": 720}
            )
            self.page = self.context.new_page()
            self.page.goto("https://colonist.io/")
            self.new_game()

    def new_game(self):
        self.page.get_by_label("Consent", exact=True).click()
        self.page.click("#scene_landing_page_container", position={"x": 500, "y": 500})
        self.login_via_discord()

    def login_via_discord(self):
        self.page.click("id=header_profile_login_button")
        self.page.click("text=Login with Discord")

        # Interact with login form
        self.page.get_by_label("EMAIL OR PHONE NUMBER").fill(config["DISCORD_USR"])
        self.page.get_by_label("PASSWORD").fill(config["DISCORD_PWD"])
        self.page.get_by_role("button", name="Log in").click()
        self.page.click("text=Authorize")

        # Continue with the test
        self.context.storage_state(path="state.json")

    def create_1v1_game(self):
        self.context.storage_state(path="state.json")
        self.page.click("text=Lobby")
        self.page.click("text=Create Room")

        self.page.locator("body").click(position={"x": 10, "y": 100})
        self.page.locator("#add-bot-button").nth(1).click()
        self.page.locator("#botspeed_settings_right_arrow").click()
        self.page.wait_for_timeout(1000)

        players, players_name = self.set_players()

        self.page.click("text=Start Game")

        canvas = self.page.locator("canvas").nth(0)
        self.page.wait_for_url("https://colonist.io/#*")
        self.page.wait_for_load_state("domcontentloaded")
        self.page.wait_for_timeout(10000)

        return canvas, players, players_name

    def load_model_pickle(self, rel_model_path: str):
        parent = pathlib.Path(__file__).parent.resolve()
        model_path = parent.joinpath(rel_model_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def generate_center_pxl_mapping(self):
        center_pxl_to_coords = {}
        center_coords_to_pxl = {}

        for spt_y, spt_x in self.center_coords:
            nomarlised_y = (spt_y - 5) / 2
            nomarlised_x = (spt_x - 4) / 2
            y_pxl = (
                nomarlised_y // 2 * (self.y_2_spacing + self.y_1_spacing)
                + (0 if nomarlised_y % 2 == 0 else self.y_1_spacing)
                + (self.y_2_spacing / 2 + self.y_1_spacing)
                + self.starting_coords[0]
            )
            x_pxl = (
                nomarlised_x * (self.x_spacing)
                + self.x_spacing
                + self.starting_coords[1]
            )
            center_pxl_to_coords[(y_pxl, x_pxl)] = (spt_y, spt_x)
            center_coords_to_pxl[(spt_y, spt_x)] = (y_pxl, x_pxl)

        return center_pxl_to_coords, center_coords_to_pxl

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
            nomarlised_x = (spt_x - 2) / 2
            y_pxl = (
                +nomarlised_y * (self.y_2_spacing + self.y_1_spacing) / 2
                + self.y_1_spacing / 2
                + self.starting_coords[0]
            )
            x_pxl = (nomarlised_x * self.x_spacing) + self.starting_coords[1]
            road_pxl_to_coords[(y_pxl, x_pxl)] = (spt_y, spt_x)
            road_coords_to_pxl[(spt_y, spt_x)] = (y_pxl, x_pxl)
        return road_pxl_to_coords, road_coords_to_pxl

    def take_canvas_img(
        self,
    ) -> cv2.typing.MatLike:
        screenshot_bytes = self.canvas.screenshot(
            path=f"game_images/screenshot_{self.n_action}.png", animations="disabled"
        )
        self.canvas_img = cv2.imdecode(
            np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR
        )

    def set_players(self) -> tuple[dict[int, Player], dict[str, int]]:
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
        players_name_dict["You"] = -9
        players_name_dict["you"] = -9
        return players_dict, players_name_dict

    def play_game(self):
        """
        The game state is coming from messages + image.
        Playing the game is derived from that.

        """
        time_past = 0
        while True:
            time_past += 1
            self.page.wait_for_timeout(2000)
            self.get_game_state()

            if self.last_rolled() == 7:
                self.player_automator.discard_resources_turn()

            if self.is_my_turn():
                if len(self.player_automator.settlements) < 2:
                    self.player_automator.player_start()
                else:
                    self.player_automator.roll_dice()
                    self.get_game_state()

                    if self.last_rolled() == 7:
                        self.player_automator.discard_resources_turn()
                        self.robber_action(self.player_automator)

                    self.player_automator.player_turn()
                    self.player_automator.pass_turn()

                self.n_action += 1

            for player in self.players.values():
                player.recalc_points()

    def robber_action(self, current_player: PlayerAI):
        current_player.action_space = current_player.get_action_space(
            situation=Action.ROBBER
        )
        action, attributes = current_player.pick_action()
        logging.debug(f"Robber Action: {action}, Attributes: {attributes}")
        current_player.perform_action(action, attributes, remove_resources=False)

    def get_game_state(self, debug=False):
        self.take_canvas_img()
        self.tally_new_messages()
        if debug:
            for player in self.players.values():
                player.player_posturn_debug()

    def is_game_start(self):
        return self.n_action < 2

    def is_my_turn(self):
        pxl_at_coord = self.canvas_img[645, 740]
        if np.array_equal(pxl_at_coord, np.array([236, 244, 247])):
            return True
        else:
            return False

    def last_rolled(self):
        messages = self.page.locator(".message-post").all()
        for message in reversed(messages):
            message_text = message.inner_text()
            if "rolled" in message_text:
                dice_rolled = self.query_rolled(message)
                return dice_rolled

    def tally_new_messages(
        self,
    ):
        messages = self.page.locator(".message-post").all()
        last_msg_cnt = len(self.last_messages)
        new_msg_cnt = len(messages)

        if new_msg_cnt > last_msg_cnt:
            print("new message!")
            new_messages = messages[last_msg_cnt:new_msg_cnt]
            for message in new_messages:
                self.parse_message(message)

        self.last_messages = messages

    def parse_message(self, message: Locator):
        message_text = message.inner_text()
        print(message_text)

        if "rolled" in message_text:
            dice_rolled = self.query_rolled(message)
            print(f"Rolled  {dice_rolled}")
            # if dice_rolled == 7 and sum(self.player_automator.resources.values()) > 7:
            #     self.player_automator.discard_resources_model()

        elif "placed a" in message_text:
            print("Placing initial settlemets and roads.")
            self.loop_templates_ocr()
            pass
        elif "built a" in message_text:
            self.loop_templates_ocr()
            player = self.get_msg_players(message_text)
            if self.is_settlement(message):
                player.update_resources_settlement()
            elif self.is_road(message):
                player.update_resources_road()
            elif self.is_city(message):
                player.update_resources_city()
            else:
                print(message)

        elif "got" in message_text or "received starting resources" in message_text:
            player = self.get_msg_players(message_text)
            for resource in self.resources_list:
                num_resources = len(message.get_by_alt_text(resource.name).all())
                if num_resources > 0:
                    player.update_resources(resource.value, num_resources)

        elif "gave bank" in message_text and "and took" in message_text:
            player = self.get_msg_players(message_text)
            msg_resources = self.get_msg_resources(message)
            given_resources = msg_resources[:-1]
            received_resource = msg_resources[-1]
            assert 1 < len(given_resources) < 5
            assert all([resource == given_resources[0] for resource in given_resources])

            resource_in = Resource(given_resources[0])
            resource_out = Resource(received_resource)
            count = len(given_resources)
            player.resources[resource_in] -= count
            player.resources[resource_out] += 1
            logging.debug(f"Trading {resource_in} for {resource_out} at {count}")
        elif "stole" in message_text and "from" in message_text:
            # robber stealing
            msg_players = self.get_msg_players(message_text, expected=2)
            msg_resources = self.get_msg_resources(message)
            assert len(msg_resources) == 1
            resource_stolen = msg_resources[0]
            stealer = msg_players[0]
            giver = msg_players[1]

            stealer.resources[resource_stolen] += 1
            giver.resources[resource_stolen] -= 1

        elif "stole" in message_text and "from" not in message_text:  # Monopoly
            msg_players = self.get_msg_players(message_text, expected=1)
            msg_resources = self.get_msg_resources(message)
            assert len(msg_resources) == 1
            resource_stolen = msg_resources[0]
            stealer = msg_players
            stealer.monopoly_actions_resources(resource=resource_stolen)

        elif (
            "bought" in message_text
            and message.get_by_alt_text("development card").is_visible()
        ):
            """Bought dev card"""
            player = self.get_msg_players(message_text)
            if player != self.player_automator:
                player.dev_card_count += 1
            else:
                self.update_dev_cards(self.canvas_img)
            player.update_resources_dev_card()

        elif "used" in message_text:
            """using a dev card"""
            player = self.get_msg_players(message_text)
            if "Monopoly" in message_text:
                dev_card = DevelopmentCard.MONOPOLY
            elif "Knight" in message_text:
                dev_card = DevelopmentCard.KNIGHT
            elif "Road Building" in message_text:
                dev_card = DevelopmentCard.ROADBUILDING
            elif "Year of Plenty" in message_text:
                dev_card = DevelopmentCard.YEAROFPLENTY

            player.dev_card_count -= 1
            player.dev_card_used[dev_card] += 1
            player.dev_cards[dev_card] -= 1

        elif "took from bank" in message_text:
            """Took from bank"""
            player = self.get_msg_players(message_text)
            msg_resources = self.get_msg_resources(message)
            assert len(msg_resources) == 2
            player.resources[Resource(msg_resources[0])] += 1
            player.resources[Resource(msg_resources[1])] += 1

        elif "discarded" in message_text:
            player = self.get_msg_players(message_text)
            msg_resources = self.get_msg_resources(message)
            if player is self.player_automator:
                pass
            else:
                for resource in msg_resources:
                    player.resources[resource] -= 1

        elif "embargo" in message_text:
            pass
        elif "" in message_text:
            pass

        else:
            warnings.warn("Unparsed message " + message_text)

    def get_msg_resources(
        self,
        message: Locator,
    ) -> list[Resource]:
        resource_names = [resource.name for resource in Resource]
        resource_pattern = re.compile(rf'{"|".join(resource_names)}', re.IGNORECASE)
        msg_resources: list[Locator] = []
        # for resource_alt_text in renamed_resource:
        locators = message.get_by_alt_text(resource_pattern).all()

        msg_resources += [
            Resource[locator.get_attribute("alt").upper()] for locator in locators
        ]

        return msg_resources

    def get_msg_players(self, message_text: str, expected=1) -> list[Player] | Player:
        player_pattern = re.compile(
            rf'{"|".join(self.players_name.keys())}', re.IGNORECASE
        )

        msg_player_names = re.findall(player_pattern, message_text)
        msg_players = [
            self.players[self.players_name[msg_player_name]]
            for msg_player_name in msg_player_names
        ]

        assert len(msg_players) == expected
        if expected == 1:
            return msg_players[0]

        return msg_players

    def query_rolled(self, message: Locator) -> list[int, int]:
        dices = ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "dice_6"]
        dice_rolled: list[str] = []
        for dice in dices:
            if (matchs := message.get_by_alt_text(dice).all()) is not None:
                for _ in matchs:
                    dice_rolled.append(dice)
        assert len(dice_rolled) == 2

        sum_of_dice = sum([int(dice.split("_")[1]) for dice in dice_rolled])
        return sum_of_dice

    def is_settlement(self, message: Locator):
        return message.get_by_alt_text("settlement").is_visible()

    def is_road(self, message: Locator):
        return message.get_by_alt_text("road").is_visible()

    def is_city(self, message: Locator):
        return message.get_by_alt_text("city").is_visible()

    def get_center(self, coords, h, w):
        return tuple(map(sum, zip(coords, (h / 2, w / 2))))

    def match_coords(
        self, pxl, mapping: dict, canvas_img: cv2.typing.MatLike, size=(40, 40)
    ):
        for y, x in mapping.keys():
            min_x = int(x - size[1] / 2)
            max_x = int(x + size[1] / 2)
            min_y = int(y - size[0] / 2)
            max_y = int(y + size[0] / 2)

            if min_x < pxl[1] < max_x and min_y < pxl[0] < max_y:
                return mapping[(y, x)], (min_x, max_x, min_y, max_y)

        canvas_img[
            int(pxl[0]) - size[0] : int(pxl[0]) + size[1],
            int(pxl[1]) - size[0] : int(pxl[1]) + size[1],
        ] = 0
        cv2.imshow("Error", canvas_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise KeyError(f"There no matching {pxl} ")

    def generate_board(self) -> np.ndarray:
        return self.empty_board

    def crop(self, canvas_img: cv2.typing.MatLike, crop: tuple[int, int, int, int]):
        h, w, channel = canvas_img.shape
        y1, y2, x1, x2 = crop
        if not y1:
            y1 = 0
        if not y2:
            y2 = h
        if not x1:
            x1 = 0
        if not x2:
            x2 = w

        return canvas_img[y1:y2, x1:x2], y1, x1

    def ocr(
        self,
        canvas_img: cv2.typing.MatLike,
        template: cv2.typing.MatLike,
        mapping: dict,
        name: str = "",
        min_match_val: float = 0.85,
        box_size=[80, 80],
        num_matches=1,
        cover_pxl_instead=False,
        crop=(None, None, None, None),
        debug=False,
    ):
        h, w = template.shape[:2]

        if crop:
            cropped_canvas_img, y1, x1 = self.crop(canvas_img, crop)

        coords = []
        while True:
            if num_matches is not None and num_matches < 1:
                break

            res = cv2.matchTemplate(cropped_canvas_img, template, self.match_method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < min_match_val:
                break

            loc_yx = (max_loc[1] + y1, max_loc[0] + x1)

            if not mapping:
                return loc_yx

            coord, (min_x, max_x, min_y, max_y) = self.match_coords(
                self.get_center(loc_yx, h, w),
                mapping=mapping,
                canvas_img=canvas_img,
                size=box_size,
            )

            if not cover_pxl_instead:
                canvas_img[
                    loc_yx[0] - 1 : loc_yx[0] + h + 1,
                    loc_yx[1] - 1 : loc_yx[1] + w + 1,
                ] = 0

            else:
                canvas_img[
                    min_y:max_y,
                    min_x:max_x,
                ] = 0
            coords.append(coord)
            if num_matches is not None:
                num_matches -= 1
        if debug:
            self.debug_display_ocr(canvas_img)
        assert coords is not None
        return coords

    def debug_display_ocr(self, img):
        if logging.DEBUG == logging.root.level:
            cv2.imshow("cuurent", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def set_harbor_ocr(self):
        canvas_img = self.canvas_img.copy()

        for harbor, harbor_count in self.harbor_tokens.items():
            template_path = f"image-matching/harbor_{harbor.name.lower()}.png"
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)

            coords = self.ocr(
                canvas_img=canvas_img,
                template=template,
                mapping=self.harbor_coords_pxl,
                name=template_path,
                num_matches=harbor_count,
                min_match_val=0.7,
            )

            assert len(coords) == harbor_count and all(
                [coord in self.harbor_coords for coord in coords]
            )
            for coord in coords:
                (y, x) = coord
                self.board[y, x] = harbor.tag

        self.harbor_ownership: dict[tuple, int] = {
            (harbor[0] + pos[0], harbor[1] + pos[1]): self.empty_board[harbor]
            for harbor, positions in self.harbor_coords.items()
            for pos in positions
        }

    def set_board_numbers_ocr(self):
        canvas_img = self.canvas_img.copy()

        for tile, tile_count in self.number_tokens.items():
            template_path = f"image-matching/tile_{str(tile)}.png"
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            h, w = template.shape[:2]

            # for tile in tile_count:
            coords = self.ocr(
                canvas_img=canvas_img,
                template=template,
                mapping=self.center_pxl_to_coords,
                name=template_path,
                min_match_val=0.70,
                num_matches=tile_count,
            )

            assert len(coords) == tile_count and all(
                [coord in self.center_coords for coord in coords]
            )
            for coord in coords:
                (y, x) = coord
                self.board[y + 1, x] = tile + 10

    def set_board_tiles_ocr(self):
        canvas_img = self.canvas_img.copy()

        for tile, tile_count in self.resource_tokens.items():
            template_path = f"image-matching/type_{str(tile.name).lower()}.png"
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            h, w = template.shape[:2]

            coords = self.ocr(
                canvas_img=canvas_img,
                template=template,
                mapping=self.center_pxl_to_coords,
                name=template_path,
                num_matches=tile_count,
            )

            assert len(coords) == tile_count and all(
                [coord in self.center_coords for coord in coords]
            )
            for coord in coords:
                (y, x) = coord
                assert coord in self.center_coords
                self.board[y - 1, x] = tile.value
                if tile is Resource.DESERT:
                    self.board[y, x] = self.robber_tag
                    self.board[y + 1, x] = 1
                else:
                    self.board[y, x] = self.dummy_robber_tag

    def loop_templates_ocr(self, debug=False) -> list[int, int]:
        """Get the coordinates of the current board"""
        success = False
        for _ in range(3):
            try:
                self.take_canvas_img()
                img = self.canvas_img.copy()
                self.update_robber_coords(img)
                self.update_player_settlements(img, debug)

                success = True
            except BaseException as err:
                print(err)
                self.page.wait_for_timeout(3000)
            else:
                break

        if not success:
            raise Exception("Somthings gone wrong")

    def update_dev_cards(self, img):
        for dev_card in DevelopmentCard:
            dev_card_template = cv2.imread(
                f"image-matching/dev_card_{dev_card.name.lower()}.png",
                cv2.IMREAD_COLOR,
            )
            coords = self.ocr(
                canvas_img=img,
                template=dev_card_template,
                mapping=None,
                min_match_val=0.7,
                crop=self.crop_bottom,
            )
            if len(coords) > 0:
                if self.player_automator.dev_cards[dev_card] == 0:
                    self.player_automator.dev_cards[dev_card] = 1
                    self.player_automator.dev_cards_turn[dev_card] = 1
                else:
                    self.player_automator.dev_cards[dev_card] = 1

                self.player_automator.recalc_dev_card_count()

    def update_robber_coords(self, img):
        self.reset_robber_spots()
        robber_template = cv2.imread(
            "image-matching/robber.png",
            cv2.IMREAD_COLOR,
        )
        coords = self.ocr(
            canvas_img=img,
            template=robber_template,
            mapping=self.center_pxl_to_coords,
            min_match_val=0.7,
        )
        assert len(coords) == 1
        (y, x) = coords[0]
        self.reset_robber_spots()
        self.board[y, x] = self.robber_tag

    def update_player_settlements(self, img: cv2.typing.MatLike, debug=False):
        for player in ["red", "blue"]:
            for match_type in ["city", "settlement", "road"]:
                if player == "red":  # needs to be figured out but
                    player_tag = -9
                    match_tag = -9
                elif player == "blue":
                    player_tag = -8
                    match_tag = -8

                if match_type == "city":
                    count = 1
                    match_tag -= 10
                    mapping = self.settlement_pxl_to_coords
                elif match_type == "settlement":
                    count = 1
                    mapping = self.settlement_pxl_to_coords
                elif match_type == "road":
                    count = 3
                    mapping = self.road_pxl_to_coords

                canvas_img = img.copy()
                all_coords = []
                for i in range(count):
                    template = cv2.imread(
                        f"image-matching/{match_type}_{player}_{i}.png",
                        cv2.IMREAD_COLOR,
                    )
                    coords = self.ocr(
                        canvas_img=canvas_img,
                        template=template,
                        mapping=mapping,
                        num_matches=None,
                        min_match_val=0.90,
                        box_size=[40, 40],
                        debug=debug,
                        cover_pxl_instead=True,
                        crop=self.crop_top,
                    )
                    all_coords += coords

                for coord in all_coords:
                    (y, x) = coord
                    self.board[y, x] = match_tag

                if match_type == "settlement":
                    self.players[player_tag].settlements = all_coords
                elif match_type == "road":
                    self.players[player_tag].roads = all_coords
                elif match_type == "city":
                    self.players[player_tag].cities = all_coords


class PlayerAutomator(PlayerAI):
    def __init__(self, catan: ColonistIOAutomator, tag, player_type):
        super().__init__(catan=catan, tag=tag, player_type=PlayerType.MODEL)
        self.catan: ColonistIOAutomator

    def pick_action(self):
        x = self.prepro()  # append board state
        z1, a1, z2, a2, z3, a3 = policy_forward(
            x=x, action_space=self.action_space, model=self.catan.model
        )

        action_idx = np.argmax(a3)  # pick action with highest
        action, attributes = self.action_idx_to_action_tuple(action_idx)

        return action, attributes

    def click_pxl(self, pxl: tuple[int, int]):
        self.catan.page.wait_for_timeout(1200)
        self.catan.canvas.click(position={"y": pxl[0], "x": pxl[1]})
        logging.info(f"Clicked on pxl {pxl}")
        self.catan.page.wait_for_timeout(1200)

    def click_buy(
        self,
        mapping: dict,
        coords: tuple,
    ):
        buy_pxl = mapping[coords]
        self.click_pxl(pxl=buy_pxl)

        above_pxl = (buy_pxl[0] - self.catan.selection_spacing, buy_pxl[1])
        self.click_pxl(pxl=above_pxl)

    def read_template(self, path: str):
        return cv2.imread(
            f"image-matching/{path}",
            cv2.IMREAD_COLOR,
        )

    def recalc_points(self):
        Player.recalc_points(self)

    def click_template(
        self,
        template: cv2.typing.MatLike,
        min_max_val=0.75,
        crop=(None, None, None, None),
    ):
        canvas_img = self.catan.canvas_img
        if crop:
            canvas_img, y1, x1 = self.catan.crop(canvas_img, crop)

        res = cv2.matchTemplate(canvas_img, template, self.catan.match_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        h, w = template.shape[:2]
        loc_yx = (max_loc[1], max_loc[0])
        max_loc_center = self.catan.get_center(loc_yx, h, w)

        click_loc = (max_loc_center[0] + y1, max_loc_center[1] + x1)

        self.click_pxl(click_loc)

    def click_confirm(self):
        self.click_pxl(self.catan.confirm_action_pxl)

    def roll_dice(self):
        dice_pxl = (570, 630)
        self.click_pxl(dice_pxl)
        # self.catan.page.keyboard.press(key="Space")

    def pass_turn(self):
        # pass_turn = (700, 670)
        # self.click_pxl(pass_turn)
        self.catan.page.keyboard.press(key="Space")
        self.catan.page.wait_for_timeout(2000)

    def build_road(self, coords: tuple, remove_resources: bool):  # Verified
        logging.info(f"Buy road for {coords}")
        if remove_resources:
            road_buying_pxl = (695, 450)
            self.click_pxl(road_buying_pxl)

        if coords in self.catan.all_road_spots:
            mapping = self.catan.road_coords_to_pxl
            self.click_buy(mapping, coords)
        else:
            raise ValueError(f"Incorrect buying coords: {coords}")

    def build_settlement(  # Verified
        self,
        coords: tuple,
        remove_resources: bool,
    ):
        if remove_resources:
            settlement_buying_pxl = (695, 520)
            self.click_pxl(settlement_buying_pxl)

        if coords not in self.catan.all_settlement_spots:
            raise ValueError(f"Incorrect buying coords: {coords}")
        logging.info(f"Buy settlement for {coords}")
        mapping = self.catan.settlement_coords_to_pxl
        self.click_buy(mapping, coords)

    def build_city(  # Verified
        self, coords: tuple
    ):
        city_buying_pxl = (695, 595)
        self.click_pxl(city_buying_pxl)

        if coords not in self.catan.all_settlement_spots:
            raise ValueError(f"Incorrect buying coords city: {coords}")
        logging.info(f"Buy settlement for {coords}")
        logging.info(f"Buy settlement for {coords}")
        mapping = self.catan.settlement_coords_to_pxl
        self.click_buy(mapping, coords)

    def embargo_opponent(self):
        opponent_icon_pxl = (590, 760)
        self.click_pxl(opponent_icon_pxl)
        self.catan.page.click("text=Embargo Player")

    def trade_resources(self, resource_in_resource_out: tuple):
        resource_in = Resource(resource_in_resource_out[0])
        resource_out = Resource(resource_in_resource_out[1])
        count = resource_in_resource_out[2]

        logging.debug(f"Trading {resource_in} for {resource_out} at {count}")
        pxl_trade_resource_dict = {
            Resource.LUMBER: (480, 30),
            Resource.BRICK: (480, 65),
            Resource.WOOL: (480, 105),
            Resource.GRAIN: (480, 140),
            Resource.ORE: (480, 175),
        }
        trade_square_pxl = (695, 305)
        self.click_pxl(trade_square_pxl)
        template = cv2.imread(
            f"image-matching/resource_card_{resource_in.name.lower()}.png",
            cv2.IMREAD_COLOR,
        )
        h, w = template[:2]
        for _ in range(count):
            self.click_template(template, crop=self.catan.crop_bottom)
            self.catan.get_game_state()
        self.click_pxl(pxl_trade_resource_dict[resource_out])

        bank_trade_pxl = (550, 300)
        self.click_pxl(bank_trade_pxl)

        self.click_pxl(trade_square_pxl)

    def discard_resources(self, attributes):
        template = self.read_template(
            f"resource_card_{Resource(attributes).name.lower()}.png"
        )
        self.click_template(template, crop=self.catan.crop_bottom)
        self.resources[Resource(attributes)] -= 1
        self.catan.get_game_state()

    def discard_resources_model(self):
        super().discard_resources_model()
        self.click_confirm()

    def place_robber(self, coords: tuple, remove_resources=False):
        if remove_resources:
            knight_template = self.read_template(
                f"dev_card_{Action.KNIGHT.name.lower()}.png"
            )
            self.click_template(knight_template)
            self.click_pxl(self.catan.confirm_action_pxl)

        self.click_buy(self.catan.robber_coords_to_pxl, coords)

    def buy_dev_card(self) -> None:
        dev_card_square = (695, 375)
        self.click_pxl(dev_card_square)
        self.catan.take_canvas_img()

    def road_building_action(self, both_roads: frozenset[tuple]):
        road_build_template = self.read_template(
            f"dev_card_{Action.ROADBUILDING.name.lower()}.png"
        )
        self.click_template(road_build_template)
        self.click_pxl(self.catan.confirm_action_pxl)
        for road in both_roads:
            self.build_road(coords=road, remove_resources=False)

    def year_of_plenty_action(self, resources_pair: frozenset[Resource]) -> None:
        # Click card
        yop_template = self.read_template(
            f"dev_card_{Action.YEAROFPLENTY.name.lower()}.png"
        )
        self.click_template(yop_template)

        # Click thhee two resource
        for resource in resources_pair:
            resource_template = self.read_template(
                f"dev_action_resource_card_{Resource(resource).name.lower()}.png"
            )
            self.click_template(resource_template, crop=self.catan.crop_dev_action_row)

        # Click confirm
        self.click_confirm()

    def monopoly_action(self, resource: Resource):
        monopoly_template = self.read_template(
            f"dev_card_{Action.MONOPOLY.name.lower()}.png"
        )
        self.click_template(
            monopoly_template,
        )

        resource_to_monopolise = self.read_template(
            f"dev_action_resource_card_{Resource(resource).name.lower()}.png"
        )
        self.click_template(resource_to_monopolise, crop=self.catan.crop_dev_action_row)

        self.click_confirm()

    def player_subturn(self):
        action = super().player_subturn()
        self.catan.get_game_state()
        return action

    def perform_action(self, action: Action, attributes, remove_resources=True):
        self.catan.page.wait_for_timeout(1000)
        super().perform_action(action, attributes, remove_resources)
        self.catan.page.wait_for_timeout(1000)
        self.catan.get_game_state()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        filename="running.txt",
        filemode="w",
    )
    colonistUI = ColonistIOAutomator(
        model_path="models/win_100_loss_100_big/catan_model.pickle"
    )
