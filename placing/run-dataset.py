import json
import numpy as np
import os
import argparse
import cv2 
import ast
from pathlib import Path

from utils import extract_list_from_last_line

PARENT_FOLDER = Path(__file__).parent.parent


# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='dining-table-5')
parser.add_argument('--baseline', type=str, default='autoimagine')
parser.add_argument('--scale', type=int, default=0)
parser.add_argument('--specify_id', type=int, nargs='*', default=None)

args = parser.parse_args()

args.iteration = 15000 if args.scale == 0 else 90000
output_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.task}"
image_path = f"{output_path}/train/autoimagine/iteration_{args.iteration}/renders/00000.png"
mask_path = "tracking/log-mask"

os.makedirs('tracking/log-mask', exist_ok=True)
os.system('rm tracking/log-mask/*')

os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
os.system(f'cp {output_path}/point_cloud/iteration_{args.iteration}/point_cloud-cp.ply {output_path}/point_cloud/iteration_{args.iteration}/point_cloud.ply')
os.chdir(f"{PARENT_FOLDER}/placing")

json_path = f"{PARENT_FOLDER}/gaussian-grouping/data/{args.task}/correct.json"
with open(json_path, 'r') as file:
    data_list = json.load(file)

def read_last_floats(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        floats = list(map(float, last_line.split()))
    return floats

def read_2nd_last_ints(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        last_line = lines[-2]
        ints = list(map(int, last_line.split()))
    return ints


def calculate_iou(rect_A, rect_B):
    x1_A, y1_A, x2_A, y2_A = rect_A
    x1_B, y1_B, x2_B, y2_B = rect_B

    x1_int = max(x1_A, x1_B)
    y1_int = max(y1_A, y1_B)
    x2_int = min(x2_A, x2_B)
    y2_int = min(y2_A, y2_B)

    inter_width = max(0, x2_int - x1_int)
    inter_height = max(0, y2_int - y1_int)
    inter_area = inter_width * inter_height

    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)

    union_area = area_A + area_B - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# 0: correct; 1: selecting failed; 2: placing failed
for id in args.specify_id if args.specify_id else range(len(data_list)):
    data = data_list[id]
    prompt = data['prompt']
    with open('data/editing-instruction', 'w') as file:
        file.write(f'Editing Instruction: \n{prompt}\n')
    with open('data/editing-instruction-obj', 'w') as file:
        file.write(f'Editing Instruction: \n{prompt}\n')
    os.system(f'cp {output_path}/point_cloud/iteration_{args.iteration}/point_cloud-cp.ply {output_path}/point_cloud/iteration_{args.iteration}/point_cloud.ply')

    ### 3d reconstruction & init planning ###

    os.system(f'python planner-init.py --dataset {args.task} --iteration {args.iteration}')

    os.makedirs(f'img/{args.task}/{id}', exist_ok=True)
    os.system(f'cp tracking/global-init.png img/{args.task}/{id}/')
    os.system(f'cp tracking/output-planning-init.txt img/{args.task}/{id}/')
    os.system(f'cp data/editing-instruction img/{args.task}/{id}/')

    with open("tracking/output-planning-init.txt", "r") as file:
        content = file.read()
        last_line = extract_list_from_last_line(content)
        assert last_line is not None, "last_line is None"
        moving_obj_list = ast.literal_eval(last_line)

    moving_object_name = moving_obj_list[0][0]
    moving_object_name = moving_object_name.replace(' ', '-')

    # selecting

    try:
        os.system(f'bash find-to-seg-single.sh {args.task} {moving_object_name} {args.scale}')
        locating_file="tracking/searcher-locating/output.txt"

        tmp = cv2.imread(image_path)
        x_pixel_obj = tmp.shape[1] // 2
        y_pixel_obj = tmp.shape[0] // 2
        obj_coord=read_last_floats(locating_file)[-4:]

        os.system(f'cp tracking/searcher-locating img/{args.task}/{id}/ -r')
        
    except Exception as e:
        print(f"Error in locating phase: {e}")
        print(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: -1")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: -1    Error: {str(e)}\n")
        continue

    if not (isinstance(obj_coord, (list, tuple)) and len(obj_coord) == 4 and all(isinstance(x, float) for x in obj_coord)):
        print(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: -1")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: -1\n")
        continue

    gt_obj_bbox_3d = data['obj_bbox']
    os.chdir(f"{PARENT_FOLDER}/gaussian-grouping")
    os.system(f"python edit_object_removal.py -m output/{args.task} --iteration {args.iteration} --render_all --render_coord {obj_coord[0]} {obj_coord[1]} {obj_coord[2]} {obj_coord[3]} \
                --get_bbox_2d {gt_obj_bbox_3d['x_min']} {gt_obj_bbox_3d['x_max']} {gt_obj_bbox_3d['y_min']} {gt_obj_bbox_3d['y_max']} {gt_obj_bbox_3d['z_min']} {gt_obj_bbox_3d['z_max']}")
    os.chdir(f"{PARENT_FOLDER}/placing")

    os.system(f"cp {output_path}/bbox_2d.json data/")
    with open('data/bbox_2d.json', 'r') as file:
        gt_obj_bbox_2d = json.load(file)
        gt_obj_bbox = [gt_obj_bbox_2d['x_min'], gt_obj_bbox_2d['y_min'], gt_obj_bbox_2d['x_max'], gt_obj_bbox_2d['y_max']]
    
    os.chdir(f"{PARENT_FOLDER}/segment-anything-2/notebooks/")
    os.makedirs(f'videos/{args.task}', exist_ok=True)
    os.system(f'rm videos/{args.task}/* outputs/{args.task}/* renders/{args.task}/*')
    os.system(f'cp {image_path} videos/{args.task}/00000.png')

    os.system(f'python get_mask.py --dataset {args.task} --coord {x_pixel_obj} {y_pixel_obj}')

    obj_mask = cv2.imread(f'outputs/{args.task}/00000.jpg', cv2.IMREAD_GRAYSCALE)

    os.chdir(f"{PARENT_FOLDER}/placing/")

    white_pixels = np.argwhere(obj_mask == 255)
    y_min_white_pixels, x_min_white_pixels = white_pixels.min(axis=0)
    y_max_white_pixels, x_max_white_pixels = white_pixels.max(axis=0)
    pred_obj_bbox = [x_min_white_pixels, y_min_white_pixels, x_max_white_pixels, y_max_white_pixels]

    iou2d = calculate_iou(pred_obj_bbox, gt_obj_bbox)

    os.system(f"cp data/bbox_2d.json {mask_path}/{args.task}.json")  
    cv2.imwrite(f"{mask_path}/{args.task}.png", obj_mask)            
    os.system(f'cp tracking/log-mask img/{args.task}/{id}/ -r')

    print('located obj iou2d: ', iou2d) 
    if iou2d < 0.25:
        print(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: {iou2d:.2f}")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 1    IOU2d: {iou2d:.2f}\n")
        continue

    # placing

    try:
        os.system(f'bash placer.sh {args.task} {moving_object_name} {args.scale}')

        # record
        os.system(f'cp tracking/placer-zooming img/{args.task}/{id}/ -r')
        os.system(f'cp tracking/placer-locating img/{args.task}/{id}/ -r')

        dst_coord = read_last_floats(f"{output_path}/coord")
    
    except Exception as e:
        print(f"Error in placing phase: {e}")
        print(f"Dataset: {args.task}    ID: {id}    Result: 2    IOU2d: {iou2d:.2f}")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 2    IOU2d: {iou2d:.2f}    Error: {str(e)}\n")
        continue

    if  dst_coord[0] > data["dst_bbox"]["x_min"] and dst_coord[0] < data["dst_bbox"]["x_max"] and \
        dst_coord[1] > data["dst_bbox"]["y_min"] and dst_coord[1] < data["dst_bbox"]["y_max"] and \
        dst_coord[2] > data["dst_bbox"]["z_min"] and dst_coord[2] < data["dst_bbox"]["z_max"]:
        print(f"Dataset: {args.task}    ID: {id}    Result: 0    IOU2d: {iou2d:.2f}    DST_coord: {dst_coord[0]:.2f} {dst_coord[1]:.2f} {dst_coord[2]:.2f}")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 0    IOU2d: {iou2d:.2f}    DST_coord: {dst_coord[0]:.2f} {dst_coord[1]:.2f} {dst_coord[2]:.2f}\n")
    else:
        print(f"Dataset: {args.task}    ID: {id}    Result: 2    IOU2d: {iou2d:.2f}    DST_coord: {dst_coord[0]:.2f} {dst_coord[1]:.2f} {dst_coord[2]:.2f}")
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(f"Dataset: {args.task}    ID: {id}    Result: 2    IOU2d: {iou2d:.2f}    DST_coord: {dst_coord[0]:.2f} {dst_coord[1]:.2f} {dst_coord[2]:.2f}\n")
    
    