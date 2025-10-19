import sys
import numpy as np
import os
import cv2 
import base64
import re
from typing import List, Tuple

from openai import OpenAI

MODEL_NAME = "openai/gpt-4o"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY')
)

def get_logprob_from_regex(token_logprob_pairs: List[Tuple[str, float]], regexes: List[str]) -> List[Tuple[str, float]]:

    for regex in regexes:
        tokens = []
        logprobs = []
        token_positions = []  # [(start, end), ...]
        current_pos = 0
        
        for token, logprob in token_logprob_pairs:
            tokens.append(token)
            logprobs.append(logprob)
            start = current_pos
            end = current_pos + len(token)
            token_positions.append((start, end))
            current_pos = end
        
        full_sentence = "".join(tokens)
        
        match = re.search(regex, full_sentence)
        
        results = []
        
        if match:
            groups = match.groups()
            
            for i, group in enumerate(groups):
                group_start = match.start(i + 1)  
                group_end = match.end(i + 1)
            
                left, right = 0, len(token_positions) - 1
                token_start_idx = len(token_positions) - 1
                
                while left <= right:
                    mid = (left + right) // 2
                    if token_positions[mid][0] <= group_start < token_positions[mid][1]:
                        token_start_idx = mid
                        break
                    elif token_positions[mid][1] <= group_start:
                        left = mid + 1
                    else:
                        right = mid - 1
                        token_start_idx = mid
                
                left, right = 0, len(token_positions) - 1
                token_end_idx = 0
                
                while left <= right:
                    mid = (left + right) // 2
                    if token_positions[mid][0] < group_end <= token_positions[mid][1]:
                        token_end_idx = mid
                        break
                    elif token_positions[mid][1] < group_end:
                        left = mid + 1
                        token_end_idx = mid
                    else:
                        right = mid - 1
                
                while token_end_idx < len(token_positions) - 1 and token_positions[token_end_idx][1] < group_end:
                    token_end_idx += 1
                
                matched_tokens = tokens[token_start_idx:token_end_idx + 1]
                matched_logprobs = logprobs[token_start_idx:token_end_idx + 1]
                
                matched_text = "".join(matched_tokens)
                logprob_sum = sum(matched_logprobs)
                
                results.append((matched_text, logprob_sum))
            return results

    raise ValueError(f"No match found for any regex in regexes: {regexes}")

def get_a_percentile(token_logprob_pairs: List[Tuple[str, float]], percentile: float) -> Tuple[str, float]:
    logprobs = [item[1] for item in token_logprob_pairs]
    quantile_value = np.quantile(logprobs, percentile)
    closest_idx = min(range(len(logprobs)), key=lambda i: abs(logprobs[i] - quantile_value))
    token, logprob = token_logprob_pairs[closest_idx]
    return token, np.exp(logprob)

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


max_retries = 5

_orig_system = os.system

def _logged_system(cmd: str):
    print(f"[ {os.path.basename(sys.argv[0])} ] -> {cmd}")
    return _orig_system(cmd)

os.system = _logged_system