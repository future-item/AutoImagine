import sys
import open3d as o3d
import numpy as np
import os
import argparse
import cv2 
import ast
import base64

from openai import OpenAI

MODEL_NAME = "openai/gpt-4o"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('OPENROUTER_API_KEY')
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_list_from_last_line(text):
    lines = text.strip().split('\n')
    
    for line in reversed(lines):
        start_index = line.find('[')  
        end_index = line.rfind(']') 
        if start_index != -1 and end_index != -1 and end_index-start_index > 2:
            list_content = line[start_index:end_index+1]  
            return list_content
    
    print("STOP_ITERATION")
    raise


def overlay_images(background, foreground, position):
    h, w = foreground.shape[:2]
    alpha_foreground = foreground[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_foreground

    for c in range(0, 3):
        background[position[1]:position[1]+h, position[0]:position[0]+w, c] = (
            alpha_foreground * foreground[:, :, c] +
            alpha_background * background[position[1]:position[1]+h, position[0]:position[0]+w, c]
        )


def stack_images(image_paths, positions, stackpieces=True):
    background = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    
    if background.shape[2] == 3:
        background = np.dstack([background, np.ones((background.shape[0], background.shape[1]), dtype=np.uint8) * 255])

    height, width = background.shape[:2]

    edge = min(height, width) // 2

    new_height = height + 2*edge
    new_width = width + 2*edge + edge
    result = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    result[edge:edge+height, edge:edge+width] = background.copy() 
    result[:,:,3] = 255

    background = result.copy()

    if not stackpieces:
        return background.copy()

    for i in range(1, len(image_paths)):
        foreground = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)

        if foreground.shape[2] == 3:
            foreground = np.dstack([foreground, np.ones((foreground.shape[0], foreground.shape[1]), dtype=np.uint8) * 255])

        overlay_images(background, foreground, positions[i])

    return background.copy()


def expand_image(image, padding: int = 4) -> np.ndarray:
    height, width, channels = image.shape
    expanded_image = np.zeros((height + 2 * padding, width + 2 * padding, channels), dtype=np.uint8)
    expanded_image[padding:padding + height, padding:padding + width] = image
    return expanded_image


def add_outline_to_image(image, outline_color: tuple = (0, 0, 255), thickness: int = 4) -> np.ndarray:

    image = expand_image(image, padding=thickness)

    alpha_channel = image[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary_mask, 100, 200)
    output_image = image.copy()
    output_image[edges > 0] = [0, 0, 0, 0]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (thickness, thickness))
    thick_edges = cv2.dilate(edges, kernel)

    output_image[thick_edges > 0] = list(outline_color) + [255]  
    return output_image


def check_in_img(img, x, y):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return 0
    return 1

max_retries = 6
