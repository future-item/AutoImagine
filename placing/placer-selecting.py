import open3d as o3d
import numpy as np
import os
import argparse
import cv2 
from pathlib import Path

from utils import *

PARENT_FOLDER = Path(__file__).parent.parent

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dining-table')
parser.add_argument('--iteration', type=str, default='90000')
parser.add_argument('--object', type=str, default='cup')

args = parser.parse_args()


# path
dataset_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.dataset}"
image_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders_with_highlights/00000.png')

os.makedirs('tracking/placer-selecting', exist_ok=True)
os.system('rm tracking/placer-selecting/*')

# prepare for placing

prompt_placer_selecting_path = 'data/prompt-placer-selecting'  
with open(prompt_placer_selecting_path, 'r', encoding='utf-8') as file:  
    prompt_placer_selecting = file.read() 

editing_instruction_path = 'data/editing-instruction'  
with open(editing_instruction_path, 'r', encoding='utf-8') as file:  
    editing_instruction = file.read() 

rendered_image_path = 'data/00000-placer-selecting.png'

ply_path = os.path.join(dataset_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')

# get init coord from ply
point_cloud = o3d.io.read_point_cloud(ply_path)
points = np.asarray(point_cloud.points)

x_min = np.percentile(points[:,0], 1, axis=0)
x_max = np.percentile(points[:,0], 99, axis=0)
y_min = np.percentile(points[:,1], 1, axis=0)
y_max = np.percentile(points[:,1], 99, axis=0)
z_min = np.percentile(points[:,2], 1, axis=0)
z_max = np.percentile(points[:,2], 99, axis=0)


os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_all --render_coord {x_min} {x_max} {y_min} {y_max} --render_highlights -1")
os.chdir(f"{PARENT_FOLDER}/placing")

img = cv2.imread(image_path)

cv2.imwrite(rendered_image_path, img)

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
                        "text": prompt_placer_selecting
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
        with open("tracking/placer-selecting/output.txt", "a") as file:
            file.write(response.choices[0].message.content)
            file.write(("\n\n---------------------------------------------------------------------------------------------------------------\n\n"))

        os.system(f"cp {rendered_image_path} tracking/placer-selecting/0.png")

        import ast
        last_line = extract_list_from_last_line(response.choices[0].message.content)
        check_result = ast.literal_eval(last_line)

        break
    
    except Exception as e:
        print(f"Error: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying {attempt + 1} times...")
        else:
            print("Error: Max retries reached.")
            raise 

with open("tracking/placer-selecting/output.txt", "a") as file:
    file.write(f'\n\n{int(check_result[0])}\n')