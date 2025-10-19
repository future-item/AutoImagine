import cv2
import numpy as np
from utils import *
import json
import os
import argparse
import sys
import logging
import subprocess
import concurrent.futures
from datetime import datetime
import ast

# args ...
parser = argparse.ArgumentParser()

parser.add_argument('--baseline', type=str, default='ours')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--max_num', type=int, default=20)
parser.add_argument('--model', type=str, required=True, help='model')
parser.add_argument('--tasks', type=int, nargs='+', required=True, help='List of task IDs to run')
args = parser.parse_args()

def run_command(command, **kwargs):

    kwargs.setdefault('shell', True)
    kwargs.setdefault('check', False) 
    if sys.platform != "win32":
        kwargs.setdefault('close_fds', True)
        
    return subprocess.run(command, **kwargs)

def setup_task_logger(task_id, baseline):
    logger_name = f'task_{task_id}'
    logger = logging.getLogger(logger_name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    
    log_dir = f'logs/{baseline}'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f'{log_dir}/task_{task_id}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(f'%(asctime)s - TASK_{task_id} - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

tot_puzzle_number = 11
max_num = args.max_num
threshold = 100


def run_task(puzzle_number):
    task_id = puzzle_number
    
    logger = setup_task_logger(task_id, args.baseline)
    logger.info(f"Starting task {task_id} for baseline {args.baseline}")

    pieces_path = f'data/pieces_{task_id}'
    init_img_path = f'data/{task_id}_init.png'
    background_path = f'data/{task_id}_background.png'
    selected_piece_path = f'data/{task_id}_selected_piece.png'
    positions_path = f'data/{task_id}_positions.txt'
    
    placer_tracking_dirs = {
        'ours': f'tracking/placer_{task_id}',
    }
    
    try:
        run_command(f'rm -rf {pieces_path}')
        run_command(f'cp -r dataset/{task_id:05} {pieces_path}')

        with open(f'{pieces_path}/correct.json', 'r') as file:
            data = json.load(file)

        bg = cv2.imread(f'{pieces_path}/0.png', cv2.IMREAD_UNCHANGED)

        image_paths = [os.path.join(pieces_path, f'{i}.png') for i in range(len(data) + 1)]
        edge = min(bg.shape[0], bg.shape[1]) // 2
        x_pos = bg.shape[1] + 2 * edge
        pieces = [ 0 ]
        y_height = []
        max_width = 0
        for i in range(1, len(data) + 1):
            tmp = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
            pieces.append(tmp)
            y_height.append(tmp.shape[0])
            max_width = max(max_width, tmp.shape[1])

        if sum(y_height) > bg.shape[0] + 2*edge:
            logger.warning(f'y_shape exceeded! {sum(y_height)} > {bg.shape[0] + 2*edge}')
            y_dis = (bg.shape[0] + 2*edge - sum(y_height) // 2) // (len(data) // 2 + 1)
            positions = [ (0, 0) ] + [(x_pos + edge - max_width - 20,   y_dis * (i+1) + sum(y_height[:i])) for i in range(len(data) // 2)]
            positions = positions  + [(x_pos + edge - 2*max_width - 40, y_dis * (i+1) + sum(y_height[:i])) for i in range(len(data) // 2)]
        else:
            y_dis = (bg.shape[0] + 2*edge - sum(y_height)) // (len(data) + 1)
            positions = [ (0, 0) ] + [(x_pos, y_dis * (i+1) + sum(y_height[:i])) for i in range(len(data))]

        init_img = stack_images(image_paths, positions)[:,:,:3]
        init_img = np.array(init_img, dtype=np.uint8)
        cv2.imwrite(init_img_path, init_img)
    
        img_dir = f"img/{task_id}"
        os.makedirs(img_dir, exist_ok=True)
        run_command(f'cp {init_img_path} {img_dir}/')

        score = 0
        done = [0 for i in range(len(data)+1)]
        his = []
        tested_points = 0 

        for num in range(max_num):
            
            logger.info(f"Attempt {num}:")

            selected_piece_id = np.random.randint(1, len(data) + 1)
            while done[selected_piece_id]:
                selected_piece_id = np.random.randint(1, len(data) + 1)
            
            logger.info(f"Attempt {num}: selected piece {selected_piece_id}")

            run_command(f'cp {pieces_path}/{selected_piece_id}.png {selected_piece_path}')
            cv2.imwrite(background_path, bg)

            with open(positions_path, 'w') as file:
                file.write(f'{positions[selected_piece_id][0]} {positions[selected_piece_id][1]}\n')

            placer_cmd = ''
            placer_output_file = ''

            placer_cmd = f'python placer.py --id {task_id} --model {args.model}'
            placer_output_file = f"{placer_tracking_dirs['ours']}/output.txt"
            
            if placer_cmd:
                run_command(placer_cmd)
                with open(placer_output_file, "r") as file:
                    last_line = file.readlines()[-1]
                    coords_to_try = []

                    place_result = list(map(int, last_line.split()))
                    coords_to_try = [tuple(place_result)]
            else:
                logger.error(f"Unknown baseline {args.baseline}")
                continue
            success = False
            for px, py in coords_to_try:

                x_t = px-edge+(pieces[selected_piece_id].shape[1]-1)//2
                y_t = py-edge+(pieces[selected_piece_id].shape[0]-1)//2

                if abs(x_t-data[selected_piece_id-1]['gt_x'])+abs(y_t-data[selected_piece_id-1]['gt_y']) < threshold:
                    success = True
                    place_x_pixel, place_y_pixel = x_t, y_t  
                    break

            if success:
                positions[selected_piece_id] = (data[selected_piece_id-1]['gt_x'] - (pieces[selected_piece_id].shape[1]-1) // 2 + edge, \
                                                data[selected_piece_id-1]['gt_y'] - (pieces[selected_piece_id].shape[0]-1) // 2 + edge )
                init_img = stack_images(image_paths, positions)[:,:,:3]
                init_img = np.array(init_img, dtype=np.uint8)
                cv2.imwrite(init_img_path, init_img)

                overlay_images(bg, pieces[selected_piece_id], (data[selected_piece_id-1]['gt_x'] - (pieces[selected_piece_id].shape[1]-1) // 2, \
                                                               data[selected_piece_id-1]['gt_y'] - (pieces[selected_piece_id].shape[0]-1) // 2 ))

                done[selected_piece_id] = 1
                score += 1
                his.append(0)

                score_img_dir = f"img/{task_id}/score-{score}"
                os.makedirs(score_img_dir, exist_ok=True)
                run_command(f'cp -r {placer_tracking_dirs["ours"]} {score_img_dir}/')
                run_command(f'cp {init_img_path} {score_img_dir}/score-{score:04d}.png')

                if score == len(data):
                    break
            else:
                his.append(2)

            tested_points += len(coords_to_try)

        final_message = f'ID: {task_id}    Score: {score}    History: {his}    Done: {done}    Attempt: {num+1 if score == len(data) else max_num}'
        logger.info(final_message)
        with open(f'tracking/log-dataset-{args.baseline}', 'a') as file:
            file.write(final_message + '\n')
        
        return {
            'id': task_id,
            'score': score,
            'history': his,
            'attempts': num+1 if score == len(data) else max_num,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        return {'id': task_id, 'status': 'failed', 'error': str(e)}
    finally:
        img_tracking_dir = f"img/{task_id}"
        os.makedirs(img_tracking_dir, exist_ok=True)
        run_command(f'cp -r tracking/* {img_tracking_dir}/')
        
        run_command(f'rm -rf {pieces_path} {init_img_path} {background_path} {selected_piece_path} {positions_path}')
        for d in placer_tracking_dirs.values():
            run_command(f'rm -rf {d}')
        
        logger_name = f'task_{task_id}'
        task_logger = logging.getLogger(logger_name)
        for handler in task_logger.handlers[:]:
            handler.close()
            task_logger.removeHandler(handler)

if __name__ == "__main__":
    main_log_filename = f'logs/main-run-puzzle-{args.baseline}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    os.makedirs(os.path.dirname(main_log_filename), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - MAIN - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Starting puzzle processing with baseline: {args.baseline}, workers: {args.workers}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run_task, args.tasks))

    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    result_filename = f'{results_dir}/results-puzzle-{args.baseline}-max_num:{args.max_num}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=4)

    main_logger.info(f"Results saved to {result_filename}")
    main_logger.info("All tasks completed.")
