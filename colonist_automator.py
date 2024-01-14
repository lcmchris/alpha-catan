import time
from playwright.sync_api import sync_playwright, ElementHandle

import cv2 as cv2
import numpy as np

pr = sync_playwright().start()

browser = pr.chromium.launch(headless=False)
context = browser.new_context()
page = context.new_page()

page.goto("https://colonist.io/")
page.click("text=Agree")
page.click("text=Lobby")
page.click("text=Create Room")
page.click("text=Add bot")
page.locator("#botspeed_settings_right_arrow").click()
page.click("text=Start Game")


canvas = page.locator("canvas").nth(0)
time.sleep(3)


starting_coords = [80, 80]

x_spacing = 50
y_1_spacing = 30
y_2_spacing = 60
selection_spacing = 45

nodes_count = [3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3]

road = y_1_spacing / 2


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
settlement_pxl_mapping = {}
road_pxl_mapping = {}


for spt_y, spt_x in all_settlement_spots:
    nomarlised_y = (spt_y - 2) / 2
    nomarlised_x = (spt_x - 2) / 2
    y_pxl = (
        nomarlised_y // 2 * (y_2_spacing + y_1_spacing)
        + (0 if nomarlised_y % 2 == 0 else y_1_spacing)
        + starting_coords[0]
    )
    x_pxl = nomarlised_x * (x_spacing) + starting_coords[1]
    settlement_pxl_mapping[(spt_y, spt_x)] = (y_pxl, x_pxl)


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
    road_pxl_mapping[(spt_y, spt_x)] = (y_pxl, x_pxl)


def click_buy(coords, mapping, selection_spacing=45):
    canvas.click(
        position={
            "x": mapping[coords][1],
            "y": mapping[coords][0],
        }
    )
    canvas.click(
        position={
            "x": mapping[coords][1],
            "y": mapping[coords][0] - 45,
        }
    )
    # load graphics
    time.sleep(1)


click_buy(coords=(2, 8), mapping=settlement_pxl_mapping)
click_buy(coords=(3, 7), mapping=road_pxl_mapping)
click_buy(coords=(6, 6), mapping=settlement_pxl_mapping)
click_buy(coords=(5, 6), mapping=road_pxl_mapping)

time.sleep(3)
canvas.screenshot(path="data/pr_board.png")

# find text on page
orig_messages = page.query_selector_all(".message-post")


def query_rolled(message: ElementHandle) -> [int, int]:
    dices = ["dice_1", "dice_2", "dice_3", "dice_4", "dice_5", "dice_6"]

    dice_rolled = []
    for dice in dices:
        if message.query_selector(f'img[alt="{dice}"]'):
            dice_rolled.append(dice)
    assert len(dice_rolled) == 2
    return dice_rolled


def query_placed_a(message: ElementHandle) -> None:
    check_map_for_new_structure()
    return


def check_map_for_new_structure():
    pass


def rolled(message: ElementHandle):
    return "rolled" in message.inner_text()


def place_a(message: ElementHandle):
    return "placed a" in message.inner_text()


# dice_1, dice_2, dice_3, dice_4,...

while True:
    messages = page.query_selector_all(".message-post")

    while orig_messages == messages:
        time.sleep(5)
    if rolled(messages[-1]):
        dice_rolled = query_rolled(messages[-1])
    elif place_a(messages[-1]):
        query_placed_a()

    orig_messages = page.query_selector_all(".message-post")

# page.query_selector_all("#game-log-text")
# orig_log_text = cur_log_text

## End turn
end_turn_button = page.keyboard.press("Space")
