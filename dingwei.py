import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_blue_pixel(b, g, r):
    return b >= r + g and 50 < b < 180

def locate_license_plate(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image at {image_path}")
    
    blue_pixel_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    rows, cols, _ = image.shape
    
    for row in range(rows):
        for col in range(cols):
            b, g, r = image[row, col]
            if is_blue_pixel(b, g, r):
                blue_pixel_mask[row, col] = 1
    
    # Count blue pixels in each row
    row_counts = np.sum(blue_pixel_mask, axis=1)
    # Find row indices with the most blue pixels
    k = np.argmax(row_counts)
    
    # Adjust region based on threshold
    T = 20
    while row_counts[k] > T and k > 0:
        k -= 1
    py1 = k
    k = np.argmax(row_counts)
    while row_counts[k] > T and k < rows - 1:
        k += 1
    py2 = k
    
    # Count blue pixels in each column within the detected rows
    column_counts = np.sum(blue_pixel_mask[py1:py2, :], axis=0)
    k1 = np.argmax(column_counts)
    
    # Adjust region based on threshold
    while column_counts[k1] > T and k1 > 0:
        k1 -= 1
    px1 = k1
    k1 = np.argmax(column_counts)
    while column_counts[k1] > T and k1 < cols - 1:
        k1 += 1
    px2 = k1
    
    # Extract the region of interest
    roi = image[py1:py2, px1:px2]
    
    # Display the segmented license plate
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title('Segmented License Plate')
    plt.show()

# Usage example:
if __name__ == '__main__':
    image_path = 'image.png'
    locate_license_plate(image_path)
