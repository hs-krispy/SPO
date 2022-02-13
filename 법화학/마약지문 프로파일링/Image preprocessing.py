import os
import numpy as np
from skimage.feature import hog
import cv2

dir = "C:/Users/user/Desktop/ions(32 ions)/internal scaling(306)"

data = []
hog_data = []

for file in os.listdir(dir):
    file_path = os.path.join(dir, file)
    img_array = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = img[20:-45, 30:-30]
    img = cv2.resize(img, (800, 400))
    # 3 channel(RGB image)
    data.append(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(otsu_threshold)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, otsu_threshold, 255)
    fd, hog_image = hog(canny, orientations=12, visualize=True, transform_sqrt=True, pixels_per_cell=(8, 8),
                        cells_per_block=(4, 4), multichannel=False)
    # cv2.imwrite(f"./hog/{file}", hog_image * 255)

    # Feature Descriptor
    hog_data.append(fd)

np.save("./Data/ions_internal_scaling_dataset_306(800x400).npy", data)
np.save("./Data/ions_internal_scaling_hog_dataset_306(800x400).npy", hog_data)
