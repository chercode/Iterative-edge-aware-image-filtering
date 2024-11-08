import cv2
import numpy as np

def generate_filter(image, width):
    permeability_horizon = np.ones_like(image[:, :, 0])
    permeability_vertical = np.ones_like(image[:, :, 0])

    permeability_horizon = 1 - 3 * permeability_horizon / 20
    permeability_vertical = 1 - 3 * permeability_vertical / 20

    alpha_val = 2
    sigma_val = 0.5

    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if i != (image.shape[1] - 1):
                p = image[j, i]
                p_prime = image[j, i + 1]
                permeability_horizon[j, i] = 1.0 / (1 + np.linalg.norm((p - p_prime) / sigma_val, ord=2) ** alpha_val)

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if j != (image.shape[0] - 1):
                p = image[j, i]
                p_prime = image[j + 1, i]
                permeability_vertical[j, i] = 1.0 / (1 + np.linalg.norm((p - p_prime) / sigma_val, ord=2) ** alpha_val)

    pi_horizon = np.ones((image.shape[0], image.shape[1], 2 * width + 1))
    pi_vertical = np.ones((image.shape[0], image.shape[1], 2 * width + 1))

    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            pi_left = 1
            pi_right = 1
            for z in range(0, -width - 1, -1):
                if i + z > 0:
                    pi_left *= permeability_horizon[j, i + z]
                    pi_horizon[j, i, z + width] = pi_left * permeability_horizon[j, i]
            for z in range(0, width + 1):
                if i + z < image.shape[1]:
                    pi_right *= permeability_horizon[j, i + z]
                    pi_horizon[j, i, z + width] = pi_right * permeability_horizon[j, i]

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            pi_up = 1
            pi_down = 1
            for z in range(0, -width - 1, -1):
                if j + z > 0:
                    pi_up *= permeability_vertical[j + z, i]
                    pi_vertical[j, i, z + width] = pi_up * permeability_vertical[j, i]
            for z in range(0, width + 1):
                if j + z < image.shape[0]:
                    pi_down *= permeability_vertical[j + z, i]
                    pi_vertical[j, i, z + width] = pi_down * permeability_vertical[j, i]

    h_horizon = pi_horizon / np.sum(pi_horizon, axis=2)[:, :, np.newaxis]
    h_vertical = pi_vertical / np.sum(pi_vertical, axis=2)[:, :, np.newaxis]

    return h_horizon, h_vertical

def apply_spatial_filter(image, h_horizon, h_vertical, lambda_value, width, final_iteration):
    size = image.shape
    image_next = np.copy(image)
    image_pre = np.copy(image)

    for iteration in range(final_iteration):
        for j in range(size[0]):
            for i in range(size[1]):
                temp1 = 0
                temp2 = 0
                temp3 = 0
                for z in range(-width, width + 1):
                    if z < 0:
                        if i + z > 0:
                            temp1 += h_horizon[j, i, z + width] * image_pre[j, i + z, 0]
                            temp2 += h_horizon[j, i, z + width] * image_pre[j, i + z, 1]
                            temp3 += h_horizon[j, i, z + width] * image_pre[j, i + z, 2]
                    if z == 0:
                        temp1 += h_horizon[j, i, width] * image_pre[j, i, 0]
                        temp2 += h_horizon[j, i, width] * image_pre[j, i, 1]
                        temp3 += h_horizon[j, i, width] * image_pre[j, i, 2]
                    if z > 0:
                        if i + z < size[1]:
                            temp1 += h_horizon[j, i, z + width] * image_pre[j, i + z, 0]
                            temp2 += h_horizon[j, i, z + width] * image_pre[j, i + z, 1]
                            temp3 += h_horizon[j, i, z + width] * image_pre[j, i + z, 2]
                image_next[j, i, 0] = temp1 + lambda_value * h_horizon[j, i, width] * (image[j, i, 0] - image_pre[j, i, 0])
                image_next[j, i, 1] = temp2 + lambda_value * h_horizon[j, i, width] * (image[j, i, 1] - image_pre[j, i, 1])
                image_next[j, i, 2] = temp3 + lambda_value * h_horizon[j, i, width] * (image[j, i, 2] - image_pre[j, i, 2])

        image_pre = np.copy(image_next)

    image1_horizon = image_next if final_iteration == 1 else None

    for i in range(size[1]):
        for j in range(size[0]):
            temp1 = 0
            temp2 = 0
            temp3 = 0
            for z in range(-width, width + 1):
                if z < 0:
                    if j + z > 0:
                        temp1 += h_vertical[j, i, z + width] * image_pre[j + z, i, 0]
                        temp2 += h_vertical[j, i, z + width] * image_pre[j + z, i, 1]
                        temp3 += h_vertical[j, i, z + width] * image_pre[j + z, i, 2]
                if z == 0:
                    temp1 += h_vertical[j, i, width] * image_pre[j, i, 0]
                    temp2 += h_vertical[j, i, width] * image_pre[j, i, 1]
                    temp3 += h_vertical[j, i, width] * image_pre[j, i, 2]
                if z > 0:
                    if j + z < size[0]:
                        temp1 += h_vertical[j, i, z + width] * image_pre[j + z, i, 0]
                        temp2 += h_vertical[j, i, z + width] * image_pre[j + z, i, 1]
                        temp3 += h_vertical[j, i, z + width] * image_pre[j + z, i, 2]
            image_next[j, i, 0] = temp1 + lambda_value * h_vertical[j, i, width] * (image[j, i, 0] - image_pre[j, i, 0])
            image_next[j, i, 1] = temp2 + lambda_value * h_vertical[j, i, width] * (image[j, i, 1] - image_pre[j, i, 1])
            image_next[j, i, 2] = temp3 + lambda_value * h_vertical[j, i, width] * (image[j, i, 2] - image_pre[j, i, 2])

    image_pre = np.copy(image_next)

    image1_vertical = image_next if final_iteration == 1 else None
    image2 = image_next if final_iteration == 2 else None
    image5 = image_next if final_iteration == 5 else None

    image_out = image_next

    return image1_horizon, image1_vertical, image2, image5, image_out


iteration = 5

# Load input image
input_image = cv2.imread('data/al.jpg')
input_image_processed = input_image.astype(np.float64) / 255.0

# Filter width variations
width_values = [5, 10, 20]
lambda_values = [0.1, 1, 10]

for width in width_values:
    for lambda_val in lambda_values:
        h_horizon, h_vertical = generate_filter(input_image_processed, width)
        _, _, _, _, filtered_image = apply_spatial_filter(input_image_processed, h_horizon, h_vertical, lambda_val, width, iteration)
        cv2.imwrite(f"filtered_image2_w{width}_lambda{lambda_val}.jpg", (filtered_image * 255).astype(np.uint8))

# # Horizontal filtering results_edge
# h_horizon, _ = generate_filter(input_image_processed, 5)
# image1_horizon, _, _, _, _ = apply_spatial_filter(input_image_processed, h_horizon, None, 1, 5, 1)
# cv2.imwrite("1_horizontal_filtering_result2.jpg", (image1_horizon * 255).astype(np.uint8))
#
# # Vertical filtering results_edge after horizontal filtering
# _, h_vertical = generate_filter(image1_horizon, 5)
# _, image1_vertical, _, _, _ = apply_spatial_filter(image1_horizon, None, h_vertical, 1, 5, 1)
# cv2.imwrite("1_vertical_filtering_result2.jpg", (image1_vertical * 255).astype(np.uint8))
#
# # Results after second iteration
# _, _, _, _, image_after_second_iteration = apply_spatial_filter(image1_vertical, None, None, 1, 5, 2)
# cv2.imwrite("1_after_second_iteration2.jpg", (image_after_second_iteration * 255).astype(np.uint8))
#
#
