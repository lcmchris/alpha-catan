import time
from playwright.sync_api import sync_playwright, Locator

import cv2 as cv2
import numpy as np
from catan_game import Catan

# constants
starting_coords = [80, 80]
x_spacing = 50
y_1_spacing = 30
y_2_spacing = 60
selection_spacing = 45
nodes_count = [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]
road = y_1_spacing / 2


method = cv2.TM_CCOEFF_NORMED
tile_count = 2  # 2 tiles per number

start_time = time.time()
numbers = {
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
resource_cards = {
    1: 3,  # brick
    2: 4,  # lumber
    3: 3,  # ore
    4: 4,  # grain
    5: 4,  # wool
    6: 1,  # desert
}


box_size = (45, 85)
box_coords = {
    (210, 90): (5, 8),
    (310, 90): (5, 12),
    (410, 90): (5, 16),
    (160, 180): (9, 6),
    (260, 180): (9, 10),
    (360, 180): (9, 14),
    (460, 180): (9, 18),
    (110, 270): (13, 4),
    (210, 270): (13, 8),
    (310, 270): (13, 12),
    (410, 270): (13, 16),
    (510, 270): (13, 20),
    (160, 360): (17, 6),
    (260, 360): (17, 10),
    (360, 360): (17, 14),
    (460, 360): (17, 18),
    (210, 470): (21, 8),
    (310, 470): (21, 12),
    (410, 470): (21, 16),
}


# fmt: off
all_settlement_spots = [(2, 8), (2, 12), (2, 16), (4, 6), (4, 10), (4, 14), (4, 18), (6, 6), (6, 10), (6, 14), (6, 18), 
                        (8, 4), (8, 8), (8, 12), (8, 16), (8, 20), (10, 4), (10, 8), (10, 12), (10, 16), (10, 20), 
                        (12, 2), (12, 6), (12, 10), (12, 14), (12, 18), (12, 22), (14, 2), (14, 6), (14, 10), (14, 14), (14, 18), (14, 22), 
                        (16, 4), (16, 8), (16, 12), (16, 16), (16, 20), (18, 4), (18, 8), (18, 12), (18, 16), (18, 20), 
                        (20, 6), (20, 10), (20, 14), (20, 18), (22, 6), (22, 10), (22, 14), (22, 18), (24, 8), (24, 12), (24, 16)]
all_road_spots = [(3, 7), (3, 9), (3, 11), (3, 13), (3, 15), (3, 17), (5, 6), (5, 10), (5, 14), (5, 18), 
                  (7, 5), (7, 7), (7, 9), (7, 11), (7, 13), (7, 15), (7, 17), (7, 19), 
                  (9, 4), (9, 8), (9, 12), (9, 16), (9, 20), 
                  (11, 3), (11, 5), (11, 7), (11, 9), (11, 11), (11, 13), (11, 15), (11, 17), (11, 19), (11, 21), (13, 2), (13, 6), (13, 10), (13, 14), (13, 18), (13, 22), (15, 3), (15, 5), (15, 7), (15, 9), (15, 11), (15, 13), (15, 15), (15, 17), (15, 19), (15, 21), (17, 4), (17, 8), (17, 12), (17, 16), (17, 20), (19, 5), (19, 7), (19, 9), (19, 11), (19, 13), (19, 15), (19, 17), (19, 19), (21, 6), (21, 10), (21, 14), (21, 18), (23, 7), (23, 9), (23, 11), (23, 13), (23, 15), (23, 17)]
# fmt: on



class ColonistIOAutomator():
    def __init__(self) -> None:

        self.settlement_pxl_mapping = {}
        self.road_pxl_mapping = {}

        self.pr = sync_playwright().start()
        self.page = self.create_page()
        self.canvas = self.create_1v1_game()
        # self.canvas_img = cv2.imdecode(
        #     np.frombuffer(self.canvas.screenshot(), np.uint8), cv2.IMREAD_GRAYSCALE
        # )
        # self.board_coords = self.get_board_coords_ocr()
        self.last_messages = self.page.query_selector_all(".message-post")
        self.users_settlements = {}
        self.wait_for_turn()

    def generate_board(self):
        return super().generate_board()

    def generate_settlement_pxl_mapping(self):
        for spt_y, spt_x in all_settlement_spots:
            nomarlised_y = (spt_y - 2) / 2
            nomarlised_x = (spt_x - 2) / 2
            y_pxl = (
                nomarlised_y // 2 * (y_2_spacing + y_1_spacing)
                + (0 if nomarlised_y % 2 == 0 else y_1_spacing)
                + starting_coords[0]
            )
            x_pxl = nomarlised_x * (x_spacing) + starting_coords[1]
            self.settlement_pxl_mapping[(spt_y, spt_x)] = (y_pxl, x_pxl)

    def generate_road_pxl_mapping(self):
        for spt_y, spt_x in all_road_spots:
            nomarlised_y = (spt_y - 3) / 2
            nomarlised_x = (spt_x - 3) / 2
            y_pxl = (
                (y_2_spacing / 2 + y_1_spacing / 2 if nomarlised_y % 2 == 1 else 0)
                + nomarlised_y // 2 * (y_2_spacing + y_1_spacing)
                + y_1_spacing / 2
                + starting_coords[0]
            )
            x_pxl = (nomarlised_x * (x_spacing) + x_spacing / 2) + starting_coords[1]
            self.road_pxl_mapping[(spt_y, spt_x)] = (y_pxl, x_pxl)

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

        self.set_users()

        self.page.click("text=Start Game")
        canvas = None
        while canvas is None:
            canvas = self.page.locator("canvas").nth(0)

        return canvas

    def set_users(self):
        users = self.page.locator("span.room_player_username").all()

        self.users = {
            users[i].inner_text().replace(" (You)", ""): i - 10
            for i in range(len(users))
            if users[i].inner_text() != "Player"
        }

    def click_buy(self, coords, selection_spacing=45):
        if coords in all_settlement_spots:
            mapping = self.settlement_pxl_mapping
        elif coords in all_road_spots:
            mapping = self.settlement_pxl_mapping

        self.canvas.click(
            position={
                "x": mapping[coords][1],
                "y": mapping[coords][0],
            }
        )
        self.canvas.click(
            position={
                "x": mapping[coords][1],
                "y": mapping[coords][0] - selection_spacing,
            }
        )
        # load graphics
        time.sleep(1)

    def end_turn(self):
        self.page.keyboard.press("Space")

    def wait_for_turn(self):
        while True:
            messages = self.page.locator(".message-post").all()
            last_msg_cnt = len(self.last_messages)
            new_msg_cnt = len(messages)
            if new_msg_cnt > last_msg_cnt:
                new_messages = messages[last_msg_cnt:new_msg_cnt]
                for message in new_messages:
                    self.parse_message(message)
            else:
                time.sleep(5)
            # if rolled(messages[-1]):

            self.last_messages = messages

    def parse_message(self, message: Locator):
        inner_html = message.inner_html()
        if "rolled" in message.inner_text():
            dice_rolled = self.query_rolled(message)
            print(dice_rolled)

        elif "placed a" in message.inner_text():
            message_alt_text = message.get_by_alt_text("settlement").is_visible()
            if message_alt_text:
                placed_settlement = self.query_settlement(message)
                print(placed_settlement)

    def query_rolled(self, message: Locator) -> list[int, int]:
        dices = ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "dice_6"]
        dice_rolled = []
        for dice in dices:
            if matchs := message.query_selector_all(f'img[alt="{dice}"]') is not None:
                for _ in matchs:
                    dice_rolled.append(dice)
        assert len(dice_rolled) == 2
        return dice_rolled

    def query_settlement(self, message: Locator) -> list[int, int]:
        for user in self.users:
            if user in message.inner_text():
                self.users_settlements[user] = self.users_settlements.get(user, 0) + 1

    def get_center(self, coords, h, w):
        return tuple(map(sum, zip(coords, (h / 2, w / 2))))

    def match_coords(
        self,
        coord,
        box_coords: dict[tuple[int, int] : tuple[int, int]],
        box_size: tuple[int, int],
    ):
        for x, y in box_coords.keys():
            if (
                x - box_size[0] < coord[0] < x + box_size[0]
                and y - box_size[1] < coord[1] < y + box_size[1]
            ):
                return box_coords[(x, y)]
        raise Exception("No match found")

    def get_board_coords_ocr(self) -> dict[int, list]:
        numbers_coords = {number: [] for number in numbers}
        for tile, tile_count in numbers.items():
            template = cv2.imread(
                f"image-matching/tile_{str(tile)}.png", cv2.IMREAD_GRAYSCALE
            )
            h, w = template.shape

            for count in range(tile_count):
                res = cv2.matchTemplate(self.canvas_img, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                numbers_coords[tile].append(
                    self.match_coords(
                        self.get_center(max_loc, h, w), box_coords, box_size
                    )
                )
        return numbers_coords

    def get_roads_ocr(self, num_roads: int) -> list:
        for road in range(num_roads):
            template = cv2.imread(
                f"image-matching/tile_{str(1)}.png", cv2.IMREAD_GRAYSCALE
            )
            h, w = template.shape

    def get_settlement_ocr(self, message: Locator) -> list[int, int]:
        """Get the coordinates of the current board"""
        pass

class PlayerAutomator(Player):

    
# find text on page
if __name__ == "__main__":
    colonistUI = ColonistIOAutomator()
