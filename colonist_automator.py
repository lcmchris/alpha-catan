import time
from playwright.sync_api import sync_playwright, Locator
import pathlib
import cv2 as cv2
import numpy as np
from catan_game import Catan



start_time = time.time()



class ColonistIOAutomator(Catan):# constants
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
    box_coords = {
        (210, 90): (5, 8),        (310, 90): (5, 12),        (410, 90): (5, 16),        (160, 180): (9, 6),
        (260, 180): (9, 10),        (360, 180): (9, 14),        (460, 180): (9, 18),        (110, 270): (13, 4),
        (210, 270): (13, 8),        (310, 270): (13, 12),        (410, 270): (13, 16),        (510, 270): (13, 20),
        (160, 360): (17, 6),        (260, 360): (17, 10),        (360, 360): (17, 14),        (460, 360): (17, 18),
        (210, 470): (21, 8),        (310, 470): (21, 12),        (410, 470): (21, 16),    }
    def __init__(self, model_path:str) -> None:
        super().__init__()
        self.model = self.load_model_pickle(model_path)


        self.settlement_pxl_mapping = {}
        self.road_pxl_mapping = {}

        self.pr = sync_playwright().start()
        self.page = self.create_page()
        self.canvas = self.create_1v1_game()
        # self.canvas_img = cv2.imdecode(
        #     np.frombuffer(self.canvas.screenshot(), np.uint8), cv2.IMREAD_GRAYSCALE
        # )
        self.board_coords = self.get_board_coords_ocr()
        self.last_messages = self.page.query_selector_all(".message-post")
        self.users_settlements = {}
        self.wait_for_turn()

    
    def load_model_pickle(self,rel_model_path:str):
        parent = pathlib.Path(__file__).resolve()
        model_path = parent.joinpath(rel_model_path)

        return pickle.loads(model_path)


    def pick_action(self):
        
        x = self.prepro()  # append board state
        _, _, _, _, _, a3 = self.policy_forward(x, self.action_space)


        best_action = np.argmax(a3)

        return self.action_idx_to_action_tuple(best_action)
    
    def perform_action(self):




    def generate_settlement_pxl_mapping(self):
        for spt_y, spt_x in self.all_settlement_spots:
            nomarlised_y = (spt_y - 2) / 2
            nomarlised_x = (spt_x - 2) / 2
            y_pxl = (
                nomarlised_y // 2 * (self.y_2_spacing + self.y_1_spacing)
                + (0 if nomarlised_y % 2 == 0 else self.y_1_spacing)
                + self.starting_coords[0]
            )
            x_pxl = nomarlised_x * (self.x_spacing) + self.starting_coords[1]
            self.settlement_pxl_mapping[(spt_y, spt_x)] = (y_pxl, x_pxl)

    def generate_road_pxl_mapping(self):
        for spt_y, spt_x in self.all_road_spots:
            nomarlised_y = (spt_y - 3) / 2
            nomarlised_x = (spt_x - 3) / 2
            y_pxl = (
                (self.y_2_spacing / 2 + self.y_1_spacing / 2 if nomarlised_y % 2 == 1 else 0)
                + nomarlised_y // 2 * (self.y_2_spacing + self.y_1_spacing)
                + self.y_1_spacing / 2
                + self.starting_coords[0]
            )
            x_pxl = (nomarlised_x * (self.x_spacing) + self.x_spacing / 2) + self.starting_coords[1]
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
        if coords in self.all_settlement_spots:
            mapping = self.settlement_pxl_mapping
        elif coords in self.all_road_spots:
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
    
    ):
        for x, y in self.box_coords.keys():
            if (
                x - self.box_size[0] < coord[0] < x + self.box_size[0]
                and y - self.box_size[1] < coord[1] < y + self.box_size[1]
            ):
                return self.box_coords[(x, y)]
        raise Exception("No match found")

    def get_board_coords_ocr(self):
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
        numbers_coords = {number: [] for number in numbers}

        for tile, tile_count in numbers.items():
            template = cv2.imread(
                f"image-matching/tile_{str(tile)}.png", cv2.IMREAD_GRAYSCALE
            )
            h, w = template.shape

            for count in range(tile_count):
                res = cv2.matchTemplate(self.canvas_img, template, self.match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                numbers_coords[self.match_coords(
                        self.get_center(max_loc, h, w), self.box_coords, self.box_size
                    )].append(
                    tile
                )

        tile_tyoe_coords = {number: [] for number in self.resource_cards}

        for tile, tile_count in numbers.items():
            template = cv2.imread(
                f"image-matching/type_{str(tile)}.png", cv2.IMREAD_GRAYSCALE
            )
            h, w = template.shape

            for count in range(tile_count):
                res = cv2.matchTemplate(self.canvas_img, template, self.match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                tile_tyoe_coords[self.match_coords(
                        self.get_center(max_loc, h, w), self.box_coords, self.box_size
                    )].append(tile
                    
                )

        
        return numbers_coords,tile_tyoe_coords
    
    def generate_board(self):
        arr = self.empty_board

        numbers_coords, tile_tyoe_coords = self.get_board_coords_ocr()



        for coord in self.center_coords:
            x,y = coord
            number = numbers_coords[coord]
            resource = tile_tyoe_coords[coord]
            arr[y, x] = 50  # Knight reference
            arr[y - 1, x] = resource
            arr[y + 1, x] = number

        return arr

    def generate_players(self):
        '''
        Always start with myself and rotate clockwise.
        '''
        return [Catan.Player, Catan.Player]
    



        

    def get_settlement_ocr(self, message: Locator) -> list[int, int]:
        """Get the coordinates of the current board"""
        pass

class PlayerAutomator(Player):

    
# find text on page
if __name__ == "__main__":
    colonistUI = ColonistIOAutomator()
