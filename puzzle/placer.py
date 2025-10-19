import cv2
import numpy as np
from utils import *
import random

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='puzzle')
parser.add_argument('--id', type=str)
parser.add_argument('--model', type=str)

args = parser.parse_args()

# path
task_id = args.id
tracking_dir = f'tracking/placer_{task_id}'
os.makedirs(tracking_dir, exist_ok=True)
os.system(f'rm {tracking_dir}/*')

prompt_placer_path = 'data/prompt-placer'  
with open(prompt_placer_path, 'r', encoding='utf-8') as file:  
    prompt_placer = file.read() 

placer_image_path = f'data/{task_id}-placer.png'
background_path = f'data/{task_id}_background.png'
selected_piece_path = f'data/{task_id}_selected_piece.png'

background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)

img_height, img_width = background.shape[0], background.shape[1]
edge = min(img_height, img_width) // 2

selected_piece = cv2.imread(selected_piece_path, cv2.IMREAD_UNCHANGED)
selected_piece = add_outline_to_image(selected_piece)
height, width = selected_piece.shape[:2]

x_pixel = edge + img_width // 2 - width // 2
y_pixel = edge + img_height // 2 - height // 2 


image_paths = [ background_path ]
positions = [ (0, 0) ]

iters = 20
max_scale = 0.10
min_scale = 0.03
search_finished = False

for iter in range(iters):

    print(x_pixel, y_pixel)

    img = stack_images(image_paths, positions, stackpieces=False)
                                                            
    overlay_images(img, selected_piece, (x_pixel, y_pixel))
    
    img = np.array(img[:,:,:3], dtype=np.uint8)


    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max((iters - iter) / iters * max_scale, random.uniform(0.67, 1.33) * min_scale)

    moving_dict = {
        'a': [0,1],
        'b': [0,-1],
        'c': [-1,0],
        'd': [1,0],
    }

    for dirstr, dir in moving_dict.items():
        (text_width, text_height), baseline = cv2.getTextSize(dirstr[0], font, 1, 2)
        text_width += 2 
        text_height += 2
        text_pos = [x_pixel + dir[0]*max(width // 2 + text_width // 2, int(scale*img_width)) + width // 2, \
                    y_pixel - dir[1]*max(height // 2 + text_height // 2, int(scale*img_height)) + height // 2]
        cv2.rectangle(img, (text_pos[0] - text_width//2, text_pos[1] - text_height//2), (text_pos[0] + text_width//2, text_pos[1] + text_height//2), (255,255,255), -1)
        cv2.putText(img, dirstr[0], (text_pos[0] - text_width//2 + 1, text_pos[1] + text_height//2 - 1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(placer_image_path, img)
    os.system(f"cp {placer_image_path} {tracking_dir}/{iter}.png")
    rendered_image = encode_image(placer_image_path)


    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(temperature=1.0,
                model=MODEL_NAME,
                messages=[ {
                    "role": "user",
                    "content": [ 
                        {   
                            "type": "text", 
                            "text": prompt_placer
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{rendered_image}" },
                        }
                    ]
                } ] 
            )   
    

            print(response.choices[0].message.content)
            print("\n---------------------------------------------------------------------------------------------------------------\n")

            last_line = extract_list_from_last_line(response.choices[0].message.content)
            result_searcher = ast.literal_eval(last_line)

            moving_dir = moving_dict[result_searcher[0]]
            x_pixel += int(img_width * scale * moving_dir[0])
            y_pixel -= int(img_height * scale * moving_dir[1])

            with open(f"{tracking_dir}/output.txt", "a") as file:
                file.write(f"{x_pixel} {y_pixel}\n")
                file.write(response.choices[0].message.content)
                file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))
            
            break
        
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1} times...")
            else:
                print("Error: Max retries reached.")
                raise 


print(f'Final Result: {x_pixel} {y_pixel}\n')

with open(f"{tracking_dir}/output.txt", "a") as file:
    file.write(f'\n\n{x_pixel} {y_pixel}\n')

