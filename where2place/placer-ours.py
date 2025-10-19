import os
import ast
import json
import cv2
import fcntl
import logging
from datetime import datetime
import argparse
import numpy as np

from utils import encode_image, extract_list_from_last_line
from openai import OpenAI

# ===== 1. Argument Parsing =====
parser = argparse.ArgumentParser(description="Determine where to place an object in a 2D image using iterative movement and checking.")
parser.add_argument('--question_id', type=int, required=True, help="The question_id in point_questions.jsonl to process.")
parser.add_argument('--data_jsonl', type=str, default='dataset/point_questions.jsonl', help="Path to point_questions.jsonl file.")
parser.add_argument('--image_dir', type=str, default='dataset/images', help="Directory where raw images are stored.")
parser.add_argument('--output_dir', type=str, default='output', help="Directory to save outputs.")
parser.add_argument('--use_session_id', action='store_true', help="Use session_id in output directory and file naming.")
parser.add_argument('--session_id', type=str, default=None, help="Session ID for grouping outputs. If not provided, will be auto-generated.")
parser.add_argument('--use_model', type=str, required=True)
args = parser.parse_args()

# ===== 2. Load Question Entry =====
question_entry = None
with open(args.data_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('question_id') == args.question_id:
            question_entry = entry
            break

if question_entry is None:
    raise ValueError(f"question_id {args.question_id} not found in {args.data_jsonl}")

image_path = os.path.join(args.image_dir, question_entry['image'])
question_text = question_entry['text']

if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# ===== 3. Setup Logging and Directories =====
image_id = os.path.splitext(os.path.basename(image_path))[0]

# Generate or use provided session_id
if args.session_id:
    session_id = args.session_id
else:
    session_id = f"{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create output directories
if args.use_session_id:
    output_image_dir = os.path.join(args.output_dir, f"images-ours-{session_id}", str(args.question_id))
    output_log_dir = os.path.join(args.output_dir, f"logs-ours-{session_id}", str(args.question_id))
else:
    output_image_dir = os.path.join(args.output_dir, "images-ours", str(args.question_id))
    output_log_dir = os.path.join(args.output_dir, "logs-ours", str(args.question_id))

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_log_dir, exist_ok=True)

# Define default final image path for saving and later reference
final_image_path = os.path.join(output_image_dir, "final_location.png")

# Setup logging
log_file = os.path.join(output_log_dir, f"{session_id}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename=log_file, mode='w'),
        logging.StreamHandler(),
    ]
)

logging.info(f"Starting session: {session_id}")
logging.info(f"Arguments: {args}")

# ===== 4. Load Prompts =====
prompt_move_path = 'prompt/prompt-move'
with open(prompt_move_path, 'r', encoding='utf-8') as file:
    prompt_move = file.read()

prompt_check_path = 'prompt/prompt-check'
with open(prompt_check_path, 'r', encoding='utf-8') as file:
    prompt_check = file.read()

# ===== 5. Choose LLM Client =====
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('OPENROUTER_API_KEY', '')
)

local_client = OpenAI(
    base_url="http://localhost:8010/v1",
    api_key=os.environ.get('LOCAL_API_KEY', '')
)

if args.use_model.endswith(':local'):
    client = local_client
    model_name = args.use_model[:-len(':local')]
else:
    client = openrouter_client
    model_name = args.use_model

# ===== 6. Helper Functions =====
def check_location(image_base64: str, max_retries: int, prompt_check: str, editing_instruction_obj: str, client: OpenAI, model_name: str) -> int:
    """
    Asks the LLM if the current location (marked by a red dot) is suitable.
    Returns: 1 if suitable ('yes'), 0 otherwise ('no').
    """
    for attempt in range(max_retries):
        try:
            logging.info("Checking location suitability...")
            
            response = client.chat.completions.create(
                temperature=1.0,
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_check},
                        {"type": "text", "text": editing_instruction_obj},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ]
                }]
            )

            content = response.choices[0].message.content

            if content is None:
                raise ValueError("API response content is None.")

            logging.info(f"LLM Check Response:\n{content}")

            last_line = extract_list_from_last_line(content)
            check_result = ast.literal_eval(last_line)

            if check_result[0] == 'yes':
                logging.info("Location is suitable.")
                return 1
            elif check_result[0] == 'no':
                logging.info("Location is not suitable.")
                return 0
            else:
                raise ValueError(f"Unexpected check result: {check_result[0]}")

        except Exception as e:
            logging.error(f"An error occurred during checking: {e}")
            if attempt < max_retries - 1:
                logging.warning(f"Retrying check... (Attempt {attempt + 2}/{max_retries})")
            else:
                logging.error("Max retries reached for checking. Failing.")
                raise
    return 0

def get_move_direction(x_pixel: int, y_pixel: int, width: int, height: int, iter_step: int, 
                      max_steps: int, min_scale: float, max_scale: float, 
                      original_img: np.ndarray, max_retries: int, 
                      prompt_move: str, editing_instruction_obj: str, 
                      client: OpenAI, model_name: str, output_image_dir: str) -> str:
    """
    Get move direction from LLM using the current position with action labels.
    Returns: one of 'a', 'b', 'c', 'd'
    """
    img_for_move = original_img.copy()
    cv2.circle(img_for_move, (x_pixel, y_pixel), 6, (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max((max_steps - iter_step) / max_steps * max_scale, min_scale)

    moving_dict = {'a': (0, 1), 'b': (0, -1), 'c': (-1, 0), 'd': (1, 0)}

    for dirstr, (dx, dy) in moving_dict.items():
        text_pos_x = x_pixel + int(dx * scale * width)
        text_pos_y = y_pixel - int(dy * scale * height)

        (text_width, text_height), _ = cv2.getTextSize(dirstr, font, 1, 2)

        # Draw white background for text
        rect_start = (text_pos_x - text_width // 2, text_pos_y - text_height // 2)
        rect_end = (text_pos_x + text_width // 2, text_pos_y + text_height // 2)
        cv2.rectangle(img_for_move, rect_start, rect_end, (255, 255, 255), -1)

        # Draw text
        text_draw_pos = (text_pos_x - text_width // 2 + 1, text_pos_y + text_height // 2 - 1)
        cv2.putText(img_for_move, dirstr, text_draw_pos, font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    move_image_path = os.path.join(output_image_dir, f"{iter_step}-move.png")
    cv2.imwrite(move_image_path, img_for_move)
    encoded_image = encode_image(move_image_path)

    # Get Move Direction from LLM
    logging.info("Requesting next move from LLM...")
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                temperature=1.0,
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_move},
                        {"type": "text", "text": editing_instruction_obj},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ]
                }]
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API response content is None.")

            logging.info(f"LLM Move Response:\n{content}")

            last_line = extract_list_from_last_line(content)
            move_result = ast.literal_eval(last_line)
            
            direction = move_result[0]
            if direction in ['a', 'b', 'c', 'd']:
                logging.info(f"Selected move direction: {direction}")
                return direction
            else:
                raise ValueError(f"Unexpected move direction: {direction}")

        except Exception as e:
            logging.error(f"An error occurred while getting move direction: {e}")
            if attempt < max_retries - 1:
                logging.warning(f"Retrying move... (Attempt {attempt + 2}/{max_retries})")
            else:
                logging.error("Max retries reached for moving. Failing.")
                raise
    
    # Fallback - should not reach here due to exception handling
    return 'a'

def log_result(question_id: int, x_pixel: int, y_pixel: int, output_dir: str, session_id: str = None, use_session_id: bool = False):
    """Write placement result to a log file."""
    try:
        if use_session_id and session_id:
            result_log_path = os.path.join(output_dir, f"result-ours-{session_id}.log")
        else:
            result_log_path = os.path.join(output_dir, "result-ours.log")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.makedirs(os.path.dirname(result_log_path), exist_ok=True)

        with open(result_log_path, 'a', encoding='utf-8') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(f"ID: {question_id:03d}, Time: {timestamp}, Coords: ({x_pixel}, {y_pixel})\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        logging.info(f"Result logged to {result_log_path}: ID={question_id}, Coords=({x_pixel}, {y_pixel})")
    except Exception as e:
        logging.error(f"Failed to log result: {e}")

# ===== 7. Main Processing =====
max_retries = 3
max_steps = 20  # 基于MCTS中的iters = 20
min_scale = 0.04
max_scale = 0.2

# Output current directory for debugging
logging.info(f"Current working directory: {os.getcwd()}")


try:
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    height, width, _ = original_img.shape
    x_pixel, y_pixel = width // 2, height // 2  # 从图像中心开始

    logging.info(f"Starting from center position: ({x_pixel}, {y_pixel})")

    # Iterative movement and checking process
    for step in range(max_steps):
        logging.info(f"=== Step {step + 1}/{max_steps} ===")
        
        # Create image with current position
        img_with_dot = original_img.copy()
        cv2.circle(img_with_dot, (x_pixel, y_pixel), 6, (0, 0, 255), 2)  # Red dot
        
        check_dot_path = os.path.join(output_image_dir, f"{step}-checkdot.png")
        cv2.imwrite(check_dot_path, img_with_dot)
        
        encoded_check_image = encode_image(check_dot_path)
        
        # Check location suitability twice (as mentioned in requirements)
        check1 = check_location(encoded_check_image, max_retries, prompt_check, question_text, client, model_name)
        check2 = check_location(encoded_check_image, max_retries, prompt_check, question_text, client, model_name)
        
        logging.info(f"Check results: {check1}, {check2}")
        
        if check1 == 1 and check2 == 1:
            logging.info("Both checks passed! Found suitable location.")
            break
        
        # Get next move direction
        direction = get_move_direction(x_pixel, y_pixel, width, height, step, max_steps,
                                     min_scale, max_scale, original_img, max_retries,
                                     prompt_move, question_text, client, model_name, output_image_dir)
        
        # Move according to direction with scale decay
        moving_dict = {'a': (0, 1), 'b': (0, -1), 'c': (-1, 0), 'd': (1, 0)}
        dx, dy = moving_dict[direction]
        
        scale = max((max_steps - step) / max_steps * max_scale, min_scale)
        new_x = x_pixel + int(width * scale * dx)
        new_y = y_pixel - int(height * scale * dy)
        
        # Clamp coordinates to be within image bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        logging.info(f"Moving from ({x_pixel}, {y_pixel}) to ({new_x}, {new_y}) with direction '{direction}' and scale {scale:.3f}")
        
        x_pixel, y_pixel = new_x, new_y
        
        # If we've reached the maximum steps, break
        if step == max_steps - 1:
            logging.warning("Reached maximum steps without finding suitable location.")
    
    # Draw and save the final location
    final_img_with_dot = original_img.copy()
    cv2.circle(final_img_with_dot, (x_pixel, y_pixel), 6, (0, 0, 255), 2)  # red dot
    cv2.imwrite(final_image_path, final_img_with_dot)
    logging.info(f"Final image saved to {final_image_path}")

    # Log result
    log_result(args.question_id, x_pixel, y_pixel, args.output_dir, session_id, args.use_session_id)

except Exception as e:
    logging.critical(f"A critical error occurred: {e}", exc_info=True)
    # Even on failure, write to log with (-1,-1)
    log_result(args.question_id, -1, -1, args.output_dir, session_id, args.use_session_id)
    raise
