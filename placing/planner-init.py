import open3d as o3d
import numpy as np
import os
import argparse
import cv2 
from pathlib import Path
import ast

from utils import *

PARENT_FOLDER = Path(__file__).parent.parent

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='dining-table-5')
parser.add_argument('--iteration', type=str, default='90000')

args = parser.parse_args()

# path 
dataset_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.dataset}"
ply_path = os.path.join(dataset_path, f'point_cloud/iteration_{args.iteration}/point_cloud.ply')
image_path = os.path.join(dataset_path, f'train/ours/iteration_{args.iteration}/renders/00000.png')

# get init coord from ply
point_cloud = o3d.io.read_point_cloud(ply_path)
points = np.asarray(point_cloud.points)

x_min = np.percentile(points[:,0], 1, axis=0)
x_max = np.percentile(points[:,0], 99, axis=0)
y_min = np.percentile(points[:,1], 1, axis=0)
y_max = np.percentile(points[:,1], 99, axis=0)
z_min = np.percentile(points[:,2], 1, axis=0)
z_max = np.percentile(points[:,2], 99, axis=0)

print(x_min, x_max, y_min, y_max, z_min, z_max)

# prepare for searching

prompt_planner_init_path = 'data/prompt-planner-init'  
with open(prompt_planner_init_path, 'r', encoding='utf-8') as file:  
    prompt_planner_init = file.read() 
editing_instruction_path = 'data/editing-instruction'  
with open(editing_instruction_path, 'r', encoding='utf-8') as file:  
    editing_instruction = file.read() 
rendered_image_global_path = 'data/00000-global.png'


os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
print("-"*10, os.path.basename(__file__), "-"*10); os.system(f"python edit_object_removal.py -m output/{args.dataset} --iteration {args.iteration} --operation skip --render_all --render_coord {x_min} {x_max} {y_min} {y_max}")
os.chdir(f"{PARENT_FOLDER}/placing")

img = cv2.imread(image_path)
cv2.imwrite(rendered_image_global_path, img)

rendered_image_global = encode_image(rendered_image_global_path)

for attempt in range(max_retries):
    try:
        response = client.chat.completions.create(temperature=0.0,
            model=MODEL_NAME,
            messages=[ {
                "role": "user",
                "content": [ 
                    {   
                        "type": "text", 
                        "text": prompt_planner_init 
                    },
                    {   
                        "type": "text", 
                        "text": editing_instruction 
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/jpeg;base64,{rendered_image_global}" },
                    }
                ]
            } ] 
        )

        last_line = extract_list_from_last_line(response.choices[0].message.content)
        result = ast.literal_eval(last_line)

        break
    
    except Exception as e:
        print(f"Error: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying {attempt + 1} times...")
        else:
            print("Error: Max retries reached.")
            raise 

print(response.choices[0].message.content)
with open("tracking/output-planning-init.txt", "w") as file:
    file.write(response.choices[0].message.content)
    file.write('\n')

os.system('cp data/00000-global.png tracking/global-init.png')