import numpy as np
import argparse
import cv2 
import ast
from pathlib import Path

from utils import *

PARENT_FOLDER = Path(__file__).parent.parent

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='living-and-kitchen')
parser.add_argument('--iteration', type=str, default='90000')
parser.add_argument('--object', type=str, default='cup')
parser.add_argument('--select_obj_id', type=str, default='2')
parser.add_argument("--coord", type=float, nargs='+')
parser.add_argument('--supp_object', type=str, default='table')

args = parser.parse_args()

# path
dataset_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.dataset}"
ply_path = os.path.join(dataset_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')
checkimage_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders/00000.png')
image_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders_with_highlights/00000.png')

os.makedirs('tracking/placer-locating', exist_ok=True)
os.system('rm tracking/placer-locating/*')

prompt_placer_locating_path = 'data/prompt-placer-locating'  
with open(prompt_placer_locating_path, 'r', encoding='utf-8') as file:  
    prompt_placer_locating = file.read() 

editing_instruction_obj_path = 'data/editing-instruction-obj'  
with open(editing_instruction_obj_path, 'r', encoding='utf-8') as file:  
    editing_instruction_obj = file.read() 

prompt_placer_suppobj_locating_path = 'data/prompt-placer-suppobj-locating'  
with open(prompt_placer_suppobj_locating_path, 'r', encoding='utf-8') as file:  
    prompt_placer_suppobj_locating = file.read() 
prompt_placer_suppobj_checking_dot_path = 'data/prompt-placer-suppobj-checking-dot'  
with open(prompt_placer_suppobj_checking_dot_path, 'r', encoding='utf-8') as file:  
    prompt_placer_suppobj_checking_dot = file.read() 
prompt_placer_suppobj_checking_mask_path = 'data/prompt-placer-suppobj-checking-mask'  
with open(prompt_placer_suppobj_checking_mask_path, 'r', encoding='utf-8') as file:  
    prompt_placer_suppobj_checking_mask = file.read() 

checking_dot_image_path = 'data/00000-placer-suppobj-checking-dot.png'
checking_mask_image_path = 'data/00000-placer-suppobj-checking-mask.png'
rendered_suppobj_image_path = 'data/00000-placer-suppobj-locating.png'
rendered_image_path = 'data/00000-placer-locating.png'

his = []
x_min, x_max, y_min, y_max = args.coord[0], args.coord[1], args.coord[2], args.coord[3]

x_fac = (x_max-x_min) / max((x_max-x_min), (y_max-y_min))
y_fac = (y_max-y_min) / max((x_max-x_min), (y_max-y_min))

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
                            "text": f"Support object: {args.supp_object}." 
                        },
                        {   
                            "type": "text", 
                            "text": editing_instruction_obj
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
            with open("tracking/placer-locating/suppobj-output.txt", "a") as file:
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

max_retry_num = 3
iters = 20
max_scale = 0.2
min_scale = 0.04
# '''
for retry_num in range(max_retry_num):

    x_pixel = None
    y_pixel = None

    os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
    print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_obj {args.select_obj_id} --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
    os.chdir(f"{PARENT_FOLDER}/placing")

    obj_img = cv2.imread(checkimage_path, cv2.IMREAD_GRAYSCALE)
    mask = np.ones(obj_img.shape, dtype=np.uint8) * 255
    mask[obj_img != 0] = 255
    mask[obj_img == 0] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)

    os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
    print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_obj -{args.select_obj_id} --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
    os.chdir(f"{PARENT_FOLDER}/placing")

    inpainted_img = cv2.imread(checkimage_path)
    inpainted_img = cv2.inpaint(inpainted_img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    for iter in range(iters):

        img = inpainted_img.copy()
        checkmask_img = img.copy()  

        height, width, _ = img.shape
        if x_pixel == None:
            x_pixel = width // 2
            y_pixel = height // 2

        cv2.circle(img, (x_pixel, y_pixel), 6, (0, 0, 255), 2)

        cv2.imwrite(checking_dot_image_path, img)
        os.system(f"cp {checking_dot_image_path} tracking/placer-locating/locating-{iter}-checkdot.png")
        checked_image = encode_image(checking_dot_image_path)

        check_result = check(checked_image, prompt_placer_suppobj_checking_dot)

        if check_result:

            os.chdir(f"{PARENT_FOLDER}/segment-anything-2/notebooks/")
            os.makedirs(f'videos/{args.dataset}', exist_ok=True)
            os.system(f'rm videos/{args.dataset}/* outputs/{args.dataset}/* renders/{args.dataset}/*')
            cv2.imwrite(f'videos/{args.dataset}/00000.png', inpainted_img)

            os.system(f'python get_mask.py --dataset {args.dataset} --coord {x_pixel} {y_pixel}')

            checkmask = cv2.imread(f'outputs/{args.dataset}/00000.jpg', cv2.IMREAD_GRAYSCALE)
            _, binarymask = cv2.threshold(checkmask, 127, 255, cv2.THRESH_BINARY)
            mask_3channel = cv2.merge([binarymask, binarymask, binarymask])
            os.chdir(f"{PARENT_FOLDER}/placing/")

            checkmask_img = cv2.bitwise_and(checkmask_img, mask_3channel)

            cv2.imwrite(checking_mask_image_path, checkmask_img)
            os.system(f"cp {checking_mask_image_path} tracking/placer-locating/locating-{iter}-checkmask.png")
            checked_image = encode_image(checking_mask_image_path)

            check_result += check(checked_image, prompt_placer_suppobj_checking_mask) and check(checked_image, prompt_placer_suppobj_checking_mask)

        if check_result > 1:
            print(f'{check_result}')
            with open("tracking/placer-locating/suppobj-output.txt", "a") as file:
                print(f'\n\n{check_result}', file=file)
                file.write(f"{x_pixel} {y_pixel}\n")
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
            text_pos = [x_pixel + dir[0]*int(scale*img.shape[1]*x_fac), \
                        y_pixel - dir[1]*int(scale*img.shape[0]*y_fac)]
            cv2.rectangle(img, (text_pos[0] - text_width//2, text_pos[1] - text_height//2), (text_pos[0] + text_width//2, text_pos[1] + text_height//2), (255,255,255), -1)
            cv2.putText(img, dirstr[0], (text_pos[0] - text_width//2 + 1, text_pos[1] + text_height//2 - 1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
        cv2.imwrite(rendered_suppobj_image_path, img)
        os.system(f"cp {rendered_suppobj_image_path} tracking/placer-locating/suppobj-{retry_num}-{iter}.png")
        rendered_image = encode_image(rendered_suppobj_image_path)


        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(temperature=0.0,
                    model=MODEL_NAME,
                    messages=[ {
                        "role": "user",
                        "content": [ 
                            {   
                                "type": "text", 
                                "text": prompt_placer_suppobj_locating 
                            },
                            {   
                                "type": "text", 
                                "text": f"Support object: {args.supp_object}." 
                            },
                                                {   
                                "type": "text", 
                                "text": editing_instruction_obj
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
                with open("tracking/placer-locating/suppobj-output.txt", "a") as file:
                    file.write(f"{x_pixel} {y_pixel}\n")
                    file.write(response.choices[0].message.content)
                    file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))

                # only tracking the last obj
                last_line = extract_list_from_last_line(response.choices[0].message.content)
                locate_result = ast.literal_eval(last_line)

                moving_dir = moving_dict[locate_result[0]]

                x_pixel += int(width * scale * moving_dir[0] * x_fac)
                y_pixel -= int(height * scale * moving_dir[1] * y_fac)

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
    

if check_result != 2:
    print("locating support object failed.")
    exit(0)
# '''

### locating here ###

iters = 20
max_scale = 0.1
min_scale = 0.02

suppobj_mask_ori = cv2.imread(checking_mask_image_path, cv2.IMREAD_GRAYSCALE)
suppobj_mask = np.where(suppobj_mask_ori > 0, 1, 0)

def check_suppmask(dst_y_pixel, dst_x_pixel, h, w, scale=0.6):
    if dst_y_pixel - (h * scale) // 2 < 0 or dst_x_pixel - (w * scale) // 2 < 0 or dst_y_pixel + (h * scale) // 2 >= suppobj_mask.shape[0] or dst_x_pixel + (w * scale) // 2 >= suppobj_mask.shape[1]:
        return 0
    region = suppobj_mask[int(dst_y_pixel - (h * scale) // 2) : int(dst_y_pixel + (h * scale) // 2 + 1), int(dst_x_pixel - (w * scale) // 2) : int(dst_x_pixel + (w * scale) // 2 + 1)]
    num_ones = np.sum(region) 
    total_elements = region.size 
    ratio = num_ones / total_elements 

    return ratio > 0.8

for iter in range(iters):

    print(x_pixel, y_pixel)
    
    dst_x = x_min + (x_pixel - (width-width*x_fac)//2)   / (width*x_fac)  * (x_max-x_min)
    dst_y = y_max - (y_pixel - (height-height*y_fac)//2) / (height*y_fac) * (y_max-y_min)

    dst_z = 99

    os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
    print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation translate --select_obj_id {args.select_obj_id} --dst_center {dst_x} {dst_y} {dst_z} --render_all --render_coord {x_min} {x_max} {y_min} {y_max} --render_highlights {args.select_obj_id}")
    os.chdir(f"{PARENT_FOLDER}/placing")

    obj_bbox_img = cv2.imread(image_path)
    img = obj_bbox_img.copy()
    black_pixels = np.all(obj_bbox_img == [0, 0, 0], axis=-1)
    img[black_pixels] = inpainted_img[black_pixels]

    img = cv2.imread(image_path)
    u,d,l,r = find_rectangle_boundaries(img, int(args.select_obj_id))

    region = mask[u-2:d+3, l-2:r+3]
    if np.all(region == 0):
        img = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max((iters - iter) / iters * max_scale, min_scale)

    moving_dict = {
        'a': [0,1],
        'b': [0,-1],
        'c': [-1,0],
        'd': [1,0],
    }

    anydir = False

    for dirstr, dir in moving_dict.items():
        (text_width, text_height), baseline = cv2.getTextSize(dirstr[0], font, 1, 2)
        text_width += 2
        text_height += 2
        text_pos = [l + dir[0]*max((r-l) // 2 + text_width // 2 + 1,  int(scale*img.shape[1]*x_fac)) + (r-l) // 2, \
                    u - dir[1]*max((d-u) // 2 + text_height // 2 + 1, int(scale*img.shape[0]*y_fac)) + (d-u) // 2]
        moving_pos = [l + dir[0]*int(scale*img.shape[1]*x_fac) + (r-l) // 2, \
                      u - dir[1]*int(scale*img.shape[0]*y_fac) + (d-u) // 2]
        if check_suppmask(moving_pos[1], moving_pos[0], d-u, r-l, iter/iters):
            cv2.rectangle(img, (text_pos[0] - text_width//2, text_pos[1] - text_height//2), (text_pos[0] + text_width//2, text_pos[1] + text_height//2), (255,255,255), -1)
            cv2.putText(img, dirstr[0], (text_pos[0] - text_width//2 + 1, text_pos[1] + text_height//2 - 1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            anydir = True
    
    if not anydir:
        continue

    cv2.imwrite(rendered_image_path, img)
    os.system(f"cp {rendered_image_path} tracking/placer-locating/{iter}.png")
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
                            "text": prompt_placer_locating 
                        },
                        {   
                            "type": "text", 
                            "text": editing_instruction_obj
                        },
                        {   
                            "type": "text", 
                            "text": f"Object to be placed: {args.object}." 
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
            with open("tracking/placer-locating/output.txt", "a") as file:
                file.write(f"{dst_x} {dst_y} {x_pixel} {y_pixel}\n")
                file.write(response.choices[0].message.content)
                file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))


            last_line = extract_list_from_last_line(response.choices[0].message.content)
            locate_result = ast.literal_eval(last_line)

            moving_dir = moving_dict[locate_result[0]]
            moving_pos = [l + moving_dir[0]*int(scale*img.shape[1]*x_fac) + (r-l) // 2, \
                          u - moving_dir[1]*int(scale*img.shape[0]*y_fac) + (d-u) // 2]
            if check_suppmask(moving_pos[1], moving_pos[0], d-u, r-l, iter/iters):
                x_pixel += moving_dir[0]*int(scale*img.shape[1]*x_fac)
                y_pixel -= moving_dir[1]*int(scale*img.shape[0]*y_fac)
            
            break
        
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1} times...")
            else:
                print("Error: Max retries reached.")
                raise 

print(f'Final Result: {dst_x} {dst_y} {x_pixel} {y_pixel}\n')

with open("tracking/placer-locating/output.txt", "a") as file:
    file.write(f'\n\n{dst_x} {dst_y} {x_pixel} {y_pixel}\n')