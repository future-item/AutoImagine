import json
from pathlib import Path
import argparse

def group_stats(items, id_start, id_end):
    subset = [item for item in items if id_start <= item["id"] <= id_end]
    total_score = sum(item.get("score", 0) for item in subset)

    avg_including_failed = (
        total_score / len(subset) if subset else 0
    )
    success_items = [item for item in subset if item.get("status") != "failed"]
    avg_excluding_failed = (
        total_score / len(success_items) if success_items else 0
    )
    return total_score, avg_including_failed, avg_excluding_failed

def evaluate_file(file_path):
    """Evaluate a single JSON file and print results."""
    print(f"{'=' * 80}")
    print(f"File: {file_path}")
    print(f"{'=' * 80}")
    
    try:
        data = json.loads(Path(file_path).read_text())
        for group, (start, end) in {"0-10": (0, 10), "11-21": (11, 21)}.items():
            total, avg_all, avg_pass = group_stats(data, start, end)
            print(f"ID {start}–{end}")
            print(f"{total}")
            if start == 0:
                print(f"  w/ failed avg: {avg_all:.2f} / 4 = {avg_all/4:.2f} ({avg_all/4*100:.1f}%)")
                print(f"  w/o failed avg: {avg_pass:.2f} / 4 = {avg_pass/4:.2f} ({avg_pass/4*100:.1f}%)\n")
            else:
                print(f"  w/ failed avg: {avg_all:.2f} / 6 = {avg_all/6:.2f} ({avg_all/6*100:.1f}%)")
                print(f"  w/o failed avg: {avg_pass:.2f} / 6 = {avg_pass/6:.2f} ({avg_pass/6*100:.1f}%)\n")
    except Exception as e:
        print(f"Error processing {file_path}: {e}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='*', default=None, 
                        help='Path(s) to results JSON file(s). If not specified, evaluates all JSON files in results/ directory.')
    args = parser.parse_args()
    
    # Determine which files to process
    if args.files is None:
        # Default: all JSON files in results/ directory
        results_dir = Path('results')
        if results_dir.exists() and results_dir.is_dir():
            json_files = sorted(results_dir.glob('*.json'))
            if not json_files:
                print("No JSON files found in results/ directory.")
                return
        else:
            print("Results directory not found.")
            return
    else:
        # Use specified files
        json_files = [Path(f) for f in args.files]
    
    # Evaluate each file
    for json_file in json_files:
        evaluate_file(json_file)
    
    print(f"{'=' * 80}")
    print(f"Total files evaluated: {len(json_files)}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
