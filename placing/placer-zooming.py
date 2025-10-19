import open3d as o3d
import numpy as np
import os
import argparse
import cv2 
import ast

from utils import *

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dining-table-5')
parser.add_argument('--iteration', type=str, default='90000')
parser.add_argument('--object', type=str, default='cup')
parser.add_argument('--select_obj_id', type=int, default=2)

args = parser.parse_args()

# path 
dataset_path = f'../gaussian-grouping/data/{args.dataset}'
ply_path = os.path.join(dataset_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')
image_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders/00000.png')

os.makedirs('tracking/placer-zooming', exist_ok=True)
os.system('rm tracking/placer-zooming/*')

point_cloud = o3d.io.read_point_cloud(ply_path)
points = np.asarray(point_cloud.points)

x_min = np.percentile(points[:,0], 1, axis=0)
x_max = np.percentile(points[:,0], 99, axis=0)
y_min = np.percentile(points[:,1], 1, axis=0)
y_max = np.percentile(points[:,1], 99, axis=0)
z_min = np.percentile(points[:,2], 1, axis=0)
z_max = np.percentile(points[:,2], 99, axis=0)

print(x_min, x_max, y_min, y_max, z_min, z_max)

with open("tracking/placer-zooming/output.txt", "a") as file:
    file.write(f'{x_min} {x_max} {y_min} {y_max}\n')
    file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))


# prepare for placing

prompt_placer_zooming_path = 'data/prompt-placer-zooming'  
with open(prompt_placer_zooming_path, 'r', encoding='utf-8') as file:  
    prompt_placer_zooming = file.read() 

editing_instruction_path = 'data/editing-instruction'  
with open(editing_instruction_path, 'r', encoding='utf-8') as file:  
    editing_instruction = file.read() 

rendered_image_path = 'data/00000-placer-zooming.png'


his = []
his.append([x_min, x_max, y_min, y_max, [], 0])

max_zooming_num = 1 if args.iteration == '15000' else 4

for num in range(max_zooming_num):

    x_min, x_max, y_min, y_max, las_result, flag = his[-1]
    
    os.chdir("../gaussian-grouping")
    print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_obj {args.select_obj_id} --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
    os.chdir("../placing")

    obj_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = np.ones(obj_img.shape, dtype=np.uint8) * 255
    mask[obj_img != 0] = 255
    mask[obj_img == 0] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)

    os.chdir("../gaussian-grouping")
    print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_obj -{args.select_obj_id} --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
    os.chdir("../placing")

    img = cv2.imread(image_path)
    img = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    expected_blocknum = 9

    height, width, _ = img.shape

    expected_len = np.sqrt(height*width / expected_blocknum)

    blocknum = [round(height / expected_len), round(width / expected_len)]

    print(blocknum)

    for i in range(1, blocknum[0]):
        y = i * (height // blocknum[0])
        cv2.line(img, (0, y), (width, y), (0, 0, 255), 2) 

    for i in range(1, blocknum[1]):
        x = i * (width // blocknum[1])
        cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(blocknum[0]):
        for j in range(blocknum[1]):
            cell_number = i * blocknum[1] + j + 1
            text = str(cell_number)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.6, 2)
            x = (j + 1) * (width // blocknum[1]) - text_width - 10 
            y = (i + 1) * (height // blocknum[0]) - 10
            cv2.putText(img, text, (x, y), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(rendered_image_path, img)
    os.system(f"cp {rendered_image_path} tracking/placer-zooming/zooming-{num}.png")
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
                            "text": prompt_placer_zooming 
                        },
                        {   
                            "type": "text", 
                            "text": editing_instruction 
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
            with open("tracking/placer-zooming/output.txt", "a") as file:
                for i in range(len(his)):
                    file.write(f"{his[i]}\n")
                file.write(f"{blocknum}\n\n")
                file.write(response.choices[0].message.content)
                file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))


            last_line = extract_list_from_last_line(response.choices[0].message.content)
            zoom_result = ast.literal_eval(last_line)

            block_id = int(zoom_result[0])

            x_size = (x_max - x_min) / blocknum[1]
            y_size = (y_max - y_min) / blocknum[0]

            next_x_min = x_min + x_size * ((block_id-1) % blocknum[1])
            next_x_max = next_x_min + x_size
            next_y_max = y_max - y_size * ((block_id-1) // blocknum[1])
            next_y_min = next_y_max - y_size
            
            next_x_min -= x_size / 2
            next_x_max += x_size / 2
            next_y_min -= y_size / 2
            next_y_max += y_size / 2

            if args.iteration == '15000':
                next_x_min = x_min
                next_x_max = x_max
                next_y_min = y_min
                next_y_max = y_max

            his.append([next_x_min, next_x_max, next_y_min, next_y_max, zoom_result, 0])

            break

        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1} times...")
            else:
                print("Error: Max retries reached.")
                raise 


zoom_result = his[-1][4]
with open("tracking/placer-zooming/output.txt", "a") as file:
    file.write(f'{zoom_result[1].replace(" ", "-")}\n')
    file.write(f'{his[-1][0]} {his[-1][1]} {his[-1][2]} {his[-1][3]}\n')

