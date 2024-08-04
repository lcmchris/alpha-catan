import cv2 as cv2
import numpy as np

img = cv2.imread("game_images/screenshot_0.png", cv2.IMREAD_COLOR)


# for i in range(3):
template = cv2.imread(f"image-matching/resource_card_grain.png", cv2.IMREAD_COLOR)
h, w = template.shape[:2]
while True:
    result = cv2.matchTemplate(image=img, templ=template, method=cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if max_val < 0.8:
    #     break

    img[
        max_loc[1] - 10 : max_loc[1] + h + 10,
        max_loc[0] - 10 : max_loc[0] + w + 10,
    ] = 0

    img = cv2.rectangle(
        img,
        (max_loc[0], max_loc[1]),
        (max_loc[0] + w + 1, max_loc[1] + h + 1),
        (0, 255, 0),
    )
    print(max_val)
    break

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
