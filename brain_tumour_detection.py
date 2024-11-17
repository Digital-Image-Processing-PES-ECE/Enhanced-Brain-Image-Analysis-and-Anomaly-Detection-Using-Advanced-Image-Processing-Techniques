import cv2
import numpy as np
from matplotlib import pyplot as plt

# Loading image of the brain
file_path = 'brain.png'
brain_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

#Enhancing contrast using CLAHE (adaptive histogram equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(brain_image)

# Highlighting edges 
laplacian_edges = cv2.Laplacian(contrast, cv2.CV_64F)
normalized_edges = cv2.normalize(laplacian_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#creating a gradient-based mask 
sobel_x = cv2.Sobel(contrast, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(contrast, cv2.CV_64F, 0, 1, ksize=3)
intensity = cv2.magnitude(sobel_x, sobel_y)
gradient = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Creating a binary mask to isolate high-gradient areas
_, anomaly = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)

# Applying sharpening 
filter = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
sharpened_image = cv2.filter2D(contrast, -1, filter)
highlighted_areas = cv2.bitwise_and(sharpened_image, sharpened_image, mask=anomaly)
result = cv2.addWeighted(contrast, 0.7, highlighted_areas, 0.3, 0)

color_output = cv2.applyColorMap(result, cv2.COLORMAP_JET)

# Segmenting high-intensity regions 
_, intensity_mask = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
closed_mask = cv2.morphologyEx(intensity_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

#finding counters
contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_contours = cv2.cvtColor(brain_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_contours, contours, -1, (0, 255, 0), 2)

# Analyzing the segmented regions
region_details = []
for contour in contours:
    region_area = cv2.contourArea(contour)
    region_mask = np.zeros_like(brain_image)
    cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
    average_intensity = cv2.mean(brain_image, mask=region_mask)[0]
    region_details.append({"Area": region_area, "Mean Intensity": average_intensity})

# Analysis
print("Detected Region Statistics:")
for i, details in enumerate(region_details):
    print(f"Region {i + 1}: Area = {details['Area']:.2f}, Mean Intensity = {details['Mean Intensity']:.2f}")


#Visualize the processing outputs
plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(brain_image, cmap='gray')

plt.subplot(2, 4, 2)
plt.title("Enhanced Contrast")
plt.imshow(contrast, cmap='gray')

plt.subplot(2, 4, 3)
plt.title("Gradient Map")
plt.imshow(gradient, cmap='gray')

plt.subplot(2, 4, 4)
plt.title("Anomaly Mask")
plt.imshow(anomaly, cmap='gray')

plt.subplot(2, 4, 5)
plt.title("Selective Sharpening")
plt.imshow(result, cmap='gray')

plt.subplot(2, 4, 6)
plt.title("Segmented Areas")
plt.imshow(closed_mask, cmap='gray')

plt.subplot(2, 4, 7)
plt.title("Contours Overlay")
plt.imshow(cv2.cvtColor(output_contours, cv2.COLOR_BGR2RGB))

plt.subplot(2, 4, 8)
plt.title("Colorized Output")
plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()