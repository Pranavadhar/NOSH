import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = '2_pencils.jpg'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
image_with_lines = image.copy()
line_coordinates = []

# Draw lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_coordinates.append((x1, y1, x2, y2))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

def line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
min_length = 300  
filtered_lines = [line for line in line_coordinates if line_length(*line) > min_length]
filtered_lines = sorted(filtered_lines, key=lambda line: line_length(*line), reverse=True)[:2]

# Draw the filtered lines on the image
image_filtered_lines = image.copy()
for line in filtered_lines:
    x1, y1, x2, y2 = line
    cv2.line(image_filtered_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Calculate the lengths of the two main pencils
lengths = [line_length(*line) for line in filtered_lines]
lengths_mm = [length * 0.1 for length in lengths]  # Convert from pixels to mm

# Calculate the angle between the two pencils
def calculate_angle(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = np.arctan2(y2 - y1, x2 - x1)
    angle2 = np.arctan2(y4 - y3, x4 - x3)
    angle = np.abs(np.degrees(angle1 - angle2))
    if angle > 180:
        angle = 360 - angle
    return angle
angle = calculate_angle(filtered_lines[0], filtered_lines[1])

# Calculate the intersection point of the two lines (if they intersect)
def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Line 1 
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    # Line 2 
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # Lines are parallel
        return None
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return int(x), int(y)

intersection = find_intersection(filtered_lines[0], filtered_lines[1])

# Draw the intersection point on the image
image_intersection = image_filtered_lines.copy()
if intersection is not None:
    cv2.circle(image_intersection, intersection, 10, (0, 0, 255), -1)

# Display the results
plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_filtered_lines, cv2.COLOR_BGR2RGB))
plt.title('Filtered Lines (Pencils)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_intersection, cv2.COLOR_BGR2RGB))
plt.title('Intersection Highlighted')
plt.axis('off')

plt.show()

# Print the results
print(f"Length of Pencil A: {lengths_mm[0]:.2f} mm")
print(f"Length of Pencil B: {lengths_mm[1]:.2f} mm")
print(f"Angle between the pencils: {angle:.2f} degrees")
