import os
import argparse
import cv2 
import ast
from pathlib import Path

from utils import *

PARENT_FOLDER = Path(__file__).parent.parent

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dining-table-5')
parser.add_argument('--iteration', type=str, default='90000')
parser.add_argument('--object', type=str, default='cup')
parser.add_argument("--coord", type=float, nargs='+')

args = parser.parse_args()

# path
dataset_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.dataset}"
ply_path = os.path.join(dataset_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')
image_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders/00000.png')

os.makedirs('tracking/searcher-locating', exist_ok=True)
os.system('rm tracking/searcher-locating/*')

# prepare for searching
prompt_searcher_locating_path = 'data/prompt-searcher-locating'  
with open(prompt_searcher_locating_path, 'r', encoding='utf-8') as file:  
    prompt_searcher_locating = file.read() 
prompt_searcher_checking_dot_path = 'data/prompt-searcher-checking-dot'  
with open(prompt_searcher_checking_dot_path, 'r', encoding='utf-8') as file:  
    prompt_searcher_checking_dot = file.read() 
prompt_searcher_checking_mask_path = 'data/prompt-searcher-checking-mask'  
with open(prompt_searcher_checking_mask_path, 'r', encoding='utf-8') as file:  
    prompt_searcher_checking_mask = file.read() 

checking_dot_image_path = 'data/00000-searcher-checking-dot.png'
checking_mask_image_path = 'data/00000-searcher-checking-mask.png'
rendered_image_path = 'data/00000-searcher-locating.png'

his = []
his.append([args.coord[0], args.coord[1], args.coord[2], args.coord[3], 1])


iters = 25
max_scale = 0.1
min_scale = 0.02
max_retry_num = 2
answer_coord = None

# checking
def check(rendered_image, prompt):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(temperature=0.0,
                model=MODEL_NAME,
                messages=[ {
                    "role": "user",
                    "content": [ 
                        {   
                            "type": "text", 
                            "text": prompt 
                        },
                        {   
                            "type": "text", 
                            "text": f"Object Description: {args.object}" 
                        },
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/jpeg;base64,{rendered_image}" },
                        }
                    ]
                } ] 
            )

            print(response.choices[0].message.content)
            print("\n--------------------------------------------------checking-----------------------------------------------------\n")
            with open("tracking/searcher-locating/output.txt", "a") as file:
                file.write(response.choices[0].message.content)
                file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))

            last_line = extract_list_from_last_line(response.choices[0].message.content)
            check_result = ast.literal_eval(last_line)
            
            if check_result[0] == 'no':
                return 0
            elif check_result[0] == 'yes':
                return 1
            else:
                raise

        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1} times...")
            else:
                print("Error: Max retries reached.")
                raise 

# locating here

for retry_num in range(max_retry_num):

    his_cp = his.copy()

    for iter in range(iters):

        x_min, x_max, y_min, y_max, flag = his[-1]

        os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
        print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
        os.chdir(f"{PARENT_FOLDER}/placing")

        img = cv2.imread(image_path)
        checkmask_img = img.copy()  

        height, width, _ = img.shape
        center = [height // 2, width // 2]
        cv2.circle(img, (center[1], center[0]), 6, (0, 0, 255), 2)

        cv2.imwrite(checking_dot_image_path, img)
        os.system(f"cp {checking_dot_image_path} tracking/searcher-locating/locating-{iter}-checkdot.png")
        checked_image = encode_image(checking_dot_image_path)

        check_result = check(checked_image, prompt_searcher_checking_dot)

        if check_result:

            # check mask
            os.chdir(f"{PARENT_FOLDER}/segment-anything-2/notebooks/")
            os.makedirs(f'videos/{args.dataset}', exist_ok=True)
            os.system(f'rm videos/{args.dataset}/* outputs/{args.dataset}/* renders/{args.dataset}/*')
            os.system(f'cp {image_path} videos/{args.dataset}/')

            os.system(f'python get_mask.py --dataset {args.dataset}')

            checkmask = cv2.imread(f'outputs/{args.dataset}/00000.jpg', cv2.IMREAD_GRAYSCALE)
            _, binarymask = cv2.threshold(checkmask, 127, 255, cv2.THRESH_BINARY)
            mask_3channel = cv2.merge([binarymask, binarymask, binarymask])
            os.chdir(f"{PARENT_FOLDER}/placing/")

            checkmask_img = cv2.bitwise_and(checkmask_img, mask_3channel)

            cv2.imwrite(checking_mask_image_path, checkmask_img)
            os.system(f"cp {checking_mask_image_path} tracking/searcher-locating/locating-{iter}-checkmask.png")
            checked_image = encode_image(checking_mask_image_path)

            check_result += check(checked_image, prompt_searcher_checking_mask) and check(checked_image, prompt_searcher_checking_mask)

        if check_result > 1:
            print(f'{check_result}')
            with open("tracking/searcher-locating/output.txt", "a") as file:
                print(f'\n\n{check_result}', file=file)
                file.write(f'{x_min} {x_max} {y_min} {y_max}\n')
            break
        elif check_result == 1:
            answer_coord = [x_min, x_max, y_min, y_max]
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max((iters - iter) / iters * max_scale, min_scale)

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
            text_pos = [center[1] + dir[0]*int(scale*img.shape[1]), \
                        center[0] - dir[1]*int(scale*img.shape[0])]
            cv2.rectangle(img, (text_pos[0] - text_width//2, text_pos[1] - text_height//2), (text_pos[0] + text_width//2, text_pos[1] + text_height//2), (255,255,255), -1)
            cv2.putText(img, dirstr[0], (text_pos[0] - text_width//2 + 1, text_pos[1] + text_height//2 - 1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
        cv2.imwrite(rendered_image_path, img)
        os.system(f"cp {rendered_image_path} tracking/searcher-locating/{retry_num}-{iter}.png")
        rendered_image = encode_image(rendered_image_path)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(temperature=0.0,
                    model=MODEL_NAME,
                    messages=[ {
                        "role": "user",
                        "content": [ 
                            {   
                                "type": "text", 
                                "text": prompt_searcher_locating 
                            },
                            {   
                                "type": "text", 
                                "text": f"Object to be located: {args.object}." 
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
                with open("tracking/searcher-locating/output.txt", "a") as file:
                    file.write(f"{his[-1]}\n")
                    file.write(response.choices[0].message.content)
                    file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))

                last_line = extract_list_from_last_line(response.choices[0].message.content)
                locate_result = ast.literal_eval(last_line)


                moving_dir = moving_dict[locate_result[0]]
                x_size = (x_max - x_min) 
                y_size = (y_max - y_min) 

                next_x_min = x_min + x_size * scale * moving_dir[0]
                next_x_max = x_max + x_size * scale * moving_dir[0]
                next_y_min = y_min + y_size * scale * moving_dir[1]
                next_y_max = y_max + y_size * scale * moving_dir[1]

                zooming_factor = 0.02
                next_x_min += x_size * zooming_factor
                next_x_max -= x_size * zooming_factor
                next_y_min += y_size * zooming_factor
                next_y_max -= y_size * zooming_factor

                his.append([next_x_min, next_x_max, next_y_min, next_y_max, 2])

                break

            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1} times...")
                else:
                    print("Error: Max retries reached.")
                    raise 


    if check_result == 2:
        break
    
    his = his_cp.copy()


print(f'{2 if check_result == 2 else 1 if answer_coord is not None else 0}')

print(f'Final Result: {his[-1]}\n')
with open("tracking/searcher-locating/output.txt", "a") as file:
    file.write(f'\n\n{his[-1][0]} {his[-1][1]} {his[-1][2]} {his[-1][3]}\n')
    
