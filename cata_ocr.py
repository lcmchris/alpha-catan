import pyautogui
import time
from playwright.sync_api import sync_playwright
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import random

# pr = sync_playwright().start()

# browser = pr.chromium.launch(headless=False)
# context = browser.new_context()
# page = context.new_page()

# page.goto("https://colonist.io/")
# page.click("text=Agree")
# page.click("text=Lobby")
# page.click("text=Create Room")
# page.locator("#botspeed_settings_right_arrow").click()
# page.click("text=Start Game")


# canvas = page.locator("canvas").nth(0)
# canvas_img = canvas.screenshot()
# canvas_nparr = np.frombuffer(canvas_img, np.uint8)
# img = cv2.imdecode(canvas_nparr, cv2.IMREAD_GRAYSCALE)

img = cv2.imread("data/0_playwright_img_collection.png", cv2.IMREAD_GRAYSCALE)

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
resource_coords = {resource_card: [] for resource_card in resource_cards}
numbers_coords = {number: [] for number in numbers}


def get_center(coords, h, w):
    return tuple(map(sum, zip(coords, (h / 2, w / 2))))


def match_coords(
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


for tile, tile_count in numbers.items():
    template = cv2.imread(f"image-matching/tile_{str(tile)}.png", cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    for count in range(tile_count):
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        numbers_coords[tile].append(
            match_coords(get_center(max_loc, h, w), box_coords, box_size)
        )

        img[
            max_loc[1] - 1 : max_loc[1] + h + 1,
            max_loc[0] - 1 : max_loc[0] + w + 1,
        ] = 0

        # img = cv2.rectangle(
        #     img,
        #     (max_loc[0], max_loc[1]),
        #     (max_loc[0] + w + 1, max_loc[1] + h + 1),
        #     (0, 255, 0),
        # )


cv2.imshow("Match", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
poll watch roll

watch settlements

watch cities

"""
