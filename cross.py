import cv2
import numpy as np
from edge import generate_filter, apply_spatial_filter

iteration = 5

image = cv2.imread('lamp_01_noflash.jpg')
image_noflash = image.astype(np.float64) / 255.0
image22 = cv2.imread('lamp_00_flash.jpg')
image_flash = image22.astype(np.float64) / 255.0
image33 = cv2.imread('lamp_03_our_result.jpg')
image_their = image33.astype(np.float64) / 255.0

W = 10
lambda_val = 5
h_horizon1, h_vertical1 = generate_filter(image_flash, W)
image1_horizon, image1_vertical, image2, image5, image_out1 = apply_spatial_filter(image_noflash, h_horizon1, h_vertical1, lambda_val, W, iteration)

cv2.imwrite("cross_lamp.jpg", (image_out1 * 255).astype(np.uint8))


cv2.imshow("Filtered By Filter From Flashed Image", image_out1)
# cv2.imshow("Filtered By Filter From No-Flashed Image", image_out2)
cv2.imshow("Their Result", image_their)
cv2.waitKey(0)
cv2.destroyAllWindows()
