#!/usr/bin/env python3

import os
import re
import cv2
import argparse
import logging
from datetime import datetime
from pathlib import Path

def check_coordinate_in_mask(x_pixel: int, y_pixel: int, question_id: int, 
                           mask_dir: str, image_dir: str = None) -> tuple[int, str]:
    try:
        mask_path = os.path.join(mask_dir, f"{question_id:02d}.jpg")
        
        if not os.path.exists(mask_path):
            return 1, f"Mask file not found: {mask_path}"
        
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return 1, f"Failed to read mask file: {mask_path}"
        
        h, w = mask_img.shape
        if not (0 <= x_pixel < w and 0 <= y_pixel < h):
            return 1, f"Coordinates ({x_pixel}, {y_pixel}) out of bounds for mask size {w}x{h}"
        
        pixel_value = mask_img[y_pixel, x_pixel]
        
        if pixel_value > 128: 
            return 2, f"SUCCESS: pixel_value={pixel_value}"
        else:
            return 0, f"FAILED: pixel_value={pixel_value}"
            
    except Exception as e:
        return 1, f"Exception: {str(e)}"

def parse_result_log(log_path: str) -> list[dict]:
    results = []
    
    patterns = [
        re.compile(r"ID:\s*(\d+),.*?Coords:\s*\((\d+),\s*(\d+)\)"),
        re.compile(r"ID:\s*(\d+),.*?Coords:\s*\((\d+),\s*(\d+)\),.*?Result:\s*(\d+)"),
    ]
    
    if not os.path.exists(log_path):
        logging.error(f"Log file not found: {log_path}")
        return results
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            matched = False
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    groups = match.groups()
                    question_id = int(groups[0])
                    x_pixel = int(groups[1])
                    y_pixel = int(groups[2])
                    
                    if x_pixel == -1 and y_pixel == -1:
                        logging.warning(f"Skipping ID {question_id} (error coordinates: -1, -1)")
                        matched = True
                        break
                    
                    results.append({
                        'question_id': question_id,
                        'x_pixel': x_pixel,
                        'y_pixel': y_pixel,
                        'line_num': line_num,
                        'original_line': line
                    })
                    matched = True
                    break
            
            if not matched:
                logging.debug(f"Line {line_num} could not be parsed: {line}")
    
    logging.info(f"Parsed {len(results)} valid entries from {log_path}")
    return results

def write_checked_results(results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Mask Check Results\n")
        f.write("# Format: ID: xxx, Time: xxx, Coords: (x, y), Result: [0=FAILED, 1=ERROR, 2=SUCCESS], Message: xxx\n\n")
        
        for result in results:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"ID: {result['question_id']:03d}, "
                   f"Time: {timestamp}, "
                   f"Coords: ({result['x_pixel']}, {result['y_pixel']}), "
                   f"Result: {result['check_result']}, "
                   f"Message: {result['check_message']}\n")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-log', type=str)
    parser.add_argument('--output-log', type=str)
    
    parser.add_argument('--mask-dir', type=str, default='dataset/masks')
    parser.add_argument('--image-dir', type=str, default='dataset/images')
    
    parser.add_argument('--batch-check', action='store_true')
    parser.add_argument('--output-dir', type=str, default='output')
    
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not os.path.exists(args.mask_dir):
        logging.error(f"Mask directory not found: {args.mask_dir}")
        return 1
    
    if args.batch_check:
        result_files = list(Path(args.output_dir).glob("result-*.log"))
        if not result_files:
            logging.warning(f"No result-*.log files found in {args.output_dir}")
            return 1
        
        logging.info(f"Found {len(result_files)} result log files for batch checking")
        
        for log_file in result_files:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing: {log_file}")
            
            output_file = log_file.parent / f"{log_file.stem}-checked.log"
            
            process_single_file(str(log_file), str(output_file), args.mask_dir)
        
        return 0
    
    if not args.input_log:
        logging.error("--input-log or --batch-check")
        parser.print_help()
        return 1
    
    if args.output_log:
        output_log = args.output_log
    else:
        input_path = Path(args.input_log)
        output_log = input_path.parent / f"{input_path.stem}-checked.log"
    
    return process_single_file(args.input_log, output_log, args.mask_dir)

def process_single_file(input_log: str, output_log: str, mask_dir: str) -> int:
    logging.info(f"Input:  {input_log}")
    logging.info(f"Output: {output_log}")
    
    results = parse_result_log(input_log)
    if not results:
        logging.error("No valid results found in log file")
        return 1
    
    success_count = 0
    error_count = 0
    fail_count = 0
    
    for result in results:
        question_id = result['question_id']
        x_pixel = result['x_pixel']
        y_pixel = result['y_pixel']
        
        check_result, check_message = check_coordinate_in_mask(
            x_pixel, y_pixel, question_id, mask_dir
        )
        
        result['check_result'] = check_result
        result['check_message'] = check_message
        
        if check_result == 2:
            success_count += 1
            logging.info(f"ID {question_id:03d}: ✓ {check_message}")
        elif check_result == 0:
            fail_count += 1
            logging.info(f"ID {question_id:03d}: ✗ {check_message}")
        else:
            error_count += 1
            logging.warning(f"ID {question_id:03d}: ⚠ {check_message}")
    
    write_checked_results(results, output_log)
    
    total = len(results)
    logging.info(f"\n{'='*60}")
    logging.info(f"检查完成! 结果已保存到: {output_log}")
    logging.info(f"总计: {total} 个样本")
    logging.info(f"成功: {success_count} ({success_count/total*100:.1f}%)")
    logging.info(f"失败: {fail_count} ({fail_count/total*100:.1f}%)")
    logging.info(f"错误: {error_count} ({error_count/total*100:.1f}%)")
    
    return 0

if __name__ == "__main__":
    exit(main())
