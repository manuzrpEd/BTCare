"""
Se pide:

Un código ejecutable en Python que tenga como resultado la identificación de la forma geométrica y el color de la misma en un formato que especifique el usuario final tras una pregunta en el terminal.

Para ello, el código aceptará una imagen de nombre “image.jpg” que se encontrará en la misma carpeta. Esta será una figura geométrica plana, sin bordes de otro color, de hasta 5 lados y círculos.

Las opciones en las que se pueda recibir la información del color (hsv, hex o RGB), será preguntada por terminal al usuario como primer paso en la ejecución del código.\
Adjuntar código e imagen con la que se han hecho las comprobaciones.
"""

# libraries
import cv2
import os
#
import numpy as np
#

# parameters
color_representations = [
    "RGB", 
    "HSV", 
    "HEX"
    ]
vertices_shapes_map = {
    3: "Triangle",
    4: "Quadrilateral",
    5: "Pentagon",
    10: "Circle"
}
scaling_factor_DP = 0.01
gaussian_kernel = (5, 5) # In the context of image processing, a kernel is a small matrix or a convolution matrix that is used for various operations like blurring, sharpening, edge detection, etc.
sigmaX = 0
lower_threshold_edge = 10
upper_threshold_edge = 30

def detect_geometric_shape(image_path, color_representation):
    # Read the image in a NumPy array where each element of the array corresponds to the intensity of a pixel in the image.
    image = cv2.imread(image_path)

    # For simplicity, efficiency and reducing dimensionality, convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection, The size of the Gaussian kernel
    blurred_image = cv2.GaussianBlur(gray_image, gaussian_kernel, sigmaX)

    # Use Canny edge detection to find edges in the image. The thresholds are gradient magnitudes used to determine which edges to consider
    edges = cv2.Canny(blurred_image, lower_threshold_edge, upper_threshold_edge)

    # Retrieve contours (only the extreme outer contours), approximating the contour points with the `CHAIN_APPROX_SIMPLE` method
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    # Approximate the polygon using the Douglas-Peucker algorithm, this reduces the number of points.
    epsilon = scaling_factor_DP * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Get the number of vertices in the shape
    num_vertices = len(approx)

    # Determine the type of shape based on the number of vertices
    shape_type = ""
    if num_vertices in [3, 4, 5]:
        shape_type = vertices_shapes_map[num_vertices]
    elif num_vertices >= 10:
        # If the shape has a large number of vertices, consider it a circle
        shape_type = vertices_shapes_map[10]
    else:
        shape_type = None

    # Calculate the average color within the contour
    mask = np.zeros_like(image)
    # get a white region that corresponds to the area inside the contour and black elsewhere.
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, mask) # keep only the pixels from the original image where the corresponding pixel is non-zero (white)
    average_color = cv2.mean(masked_image)[:3]  # Mean color in BGR format inside the contour in the masked image

    # Convert BGR to RGB, HEX, HSV
    # RGB intensity coordinates, from 255 (max intensity) to 0 (min intensity)
    if color_representation == "RGB": 
        average_color = tuple(reversed(average_color))
    # hexadecimal color, using a hexadecimal (base-16) notation, where 00 is the minimum intensity (no color contribution), and FF is the maximum intensity (full color contribution).
    elif color_representation == "HEX": 
        average_color = "#{:02X}{:02X}{:02X}".format(*map(int, average_color))
    # Hue: 0 and 360 both represent red, 120 represents green, and 240 represents blue.
    # Saturation: 0 corresponds to a shade of gray, while a saturation value of 1 represents the most vivid version of the color.
    # Value: 0 corresponds to black, and 1 corresponds to the full brightness of the color.
    elif color_representation == "HSV":
        average_color = tuple(cv2.cvtColor(np.uint8([[average_color]]), cv2.COLOR_BGR2HSV)[0][0])

    return shape_type, average_color

if __name__ == '__main__':
    image_path = 'image.jpg'
    assert os.path.exists(image_path), f"The file '{image_path}' was not found. Ensure to place an 'image.jpg' file in this directory and re-run the executable."
    print(f"Reading {image_path}")

    while True:
        color_representation = input(f"Enter color representation ({color_representations[0]}, {color_representations[1]}, {color_representations[2]}): ").upper()
        if color_representation in color_representations:
            print(f"Selected color representation: {color_representation}")
            break
        else:
            print(f"Invalid input. Please enter {color_representations[0]}, {color_representations[1]}, or {color_representations[2]}.")

    shape_type, color = detect_geometric_shape(image_path, color_representation)
    # Print the detected shape and color information
    print(f"Detected shape: {shape_type}")
    print(f"Average Color ({color_representation}): {color}")