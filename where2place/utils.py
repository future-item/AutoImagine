import sys
import numpy as np
import os
import cv2 
import base64
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_list_from_last_line(text):
    lines = text.strip().split('\n') 
    
    for i in reversed(range(len(lines))): 
        line = lines[i]
        
        start_index = line.find('[') 
        end_index = line.rfind(']') 
        
        if start_index != -1 and end_index != -1 and end_index - start_index > 1:
            list_content = line[start_index:end_index+1] 
            return list_content
        
        boxed_pattern = r'\\boxed\{\\text\{([^}]+)\}\}'
        match = re.search(boxed_pattern, line)
        if match:
            content = match.group(1) 
            return f"['{content}']" 
    print("STOP_ITERATION")
    raise

color_list = [
    ("black", (0, 0, 0)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Gray", (128, 128, 128)),
    ("Cyan", (255, 255, 0)),
    ("Orange", (0, 165, 255)),
    ("Purple", (128, 0, 128)),
    ("Pink", (203, 192, 255)),
    ("Brown", (42, 42, 165)),
]

def find_rectangle_boundaries(img, obj_id=1):

    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_color = np.array(color_list[obj_id][1])
    upper_color = np.array(color_list[obj_id][1])
    mask = cv2.inRange(img, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    x += 1
    y += 1
    w -= 3
    h -= 3
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2) 

    top = y
    bottom = y + h
    left = x
    right = x + w
    
    return top, bottom, left, right


max_retries = 30

_orig_system = os.system

def _logged_system(cmd: str):
    print(f"[ {os.path.basename(sys.argv[0])} ] -> {cmd}")
    return _orig_system(cmd)

os.system = _logged_system