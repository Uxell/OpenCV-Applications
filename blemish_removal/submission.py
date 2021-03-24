import cv2 as cv2
import numpy as np

number_of_candidates = 15
radius = 15


# Returns patches of image from around our blemished area
def find_candidate_patches(center_patch_coords):
    x = center_patch_coords[0]
    y = center_patch_coords[1]

    # Finding specific number of evenly spaced points around our patch
    t = np.linspace(0, 2 * np.pi, number_of_candidates + 1)  # +1 because, the first and the last one will be the same
    candidate_x = (np.round(2 * radius * np.cos(t)) + x).astype(int)
    candidate_y = (np.round(2 * radius * np.sin(t)) + y).astype(int)
    candidate_centers = np.c_[candidate_x, candidate_y][:-1]  # We remove the last one

    # Remove the patches that dont fit into our image
    condition = np.array([0 + radius < center[0] < image.shape[1] - radius
                          and 0 + radius < center[1] < image.shape[0] - radius
                          for center in candidate_centers])
    candidate_centers_filtered = candidate_centers[condition]

    # These are candidate patches that we get using the center coordinates and radius,
    # which we use here as half of the square side length
    candidate_patches = [image[y - radius:y + radius, x - radius:x + radius] for [x, y] in candidate_centers_filtered]

    return candidate_patches


# Calculates single number gradient score for every patch
def calculate_candidate_gradient_measures(patches):
    # Calculate horizontal and vertical sobel gradients:
    sobel_x = [calculate_mean_gradient(patch, 1, 0) for patch in patches]
    sobel_y = [calculate_mean_gradient(patch, 0, 1) for patch in patches]

    total_gradients = np.add(sobel_x, sobel_y)

    return total_gradients


# Calculates a single number gradient score (vertical or horizontal) for a single image
def calculate_mean_gradient(image_patch, xorder, yorder):
    sobel = cv2.Sobel(image_patch, cv2.CV_64F, xorder, yorder, ksize=3)
    abs_sobel = np.abs(sobel)  # Because sobel results can be negative
    mean_gradient = np.mean(np.uint8(abs_sobel))
    return mean_gradient


# Clears blemish around place where point and click our cursor
def clear_blemish(action, x, y, flags, userdata):
    global image
    # Action to be taken when left mouse button is pressed
    # Find and replace the blemish
    if action == cv2.EVENT_LBUTTONDOWN:
        blemish_center = (x, y)

        # dont do blemish removal if the patch doesnt fit in the image
        patch_fits = 0 + radius < x < image.shape[1] - radius and 0 + radius < y < image.shape[0] - radius

        if not patch_fits:
            return

        candidate_patches = find_candidate_patches(blemish_center)

        candidate_gradients = calculate_candidate_gradient_measures(candidate_patches)

        minimum_gradient_index = np.argmin(candidate_gradients)
        minimum_gradient_patch = candidate_patches[int(minimum_gradient_index)]

        mask = np.full_like(minimum_gradient_patch, 255, dtype=minimum_gradient_patch.dtype)

        image = cv2.seamlessClone(minimum_gradient_patch, image, mask, blemish_center, cv2.NORMAL_CLONE)

        cv2.imshow("Window", image)


# Read our photo to correct
image = cv2.imread("blemish.png", 1)

# Make a dummy image, will be useful to reset the photo
dummy = image.copy()

# Create window and show image
cv2.namedWindow("Window")
cv2.imshow("Window", image)

# highgui function called when mouse events occur
cv2.setMouseCallback("Window", clear_blemish)

k = 0
# loop until escape character is pressed
while k != 27:
    k = cv2.waitKey(20) & 0xFF
    if k == 99:
        image = dummy.copy()
        cv2.imshow("Window", image)

cv2.destroyAllWindows()
