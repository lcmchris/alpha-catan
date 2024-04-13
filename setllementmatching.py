import cv2 as cv2
import numpy as np

img = cv2.imread("data/colonist.io_.png", cv2.IMREAD_GRAYSCALE)


# for i in range(3):
template = cv2.imread(f"image-matching/road_red_5.png", cv2.IMREAD_GRAYSCALE)
h, w = template.shape[:2]
for i in range(1):
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if max_val > 0.95:
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

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
