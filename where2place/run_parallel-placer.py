import os
import argparse
import subprocess
import json
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import functools


def run_placer(question_id: int,
               output_dir: str,
               data_jsonl: str,
               image_dir: str,
               use_model: str,
               baseline: str = "coord",
               session_id: Optional[str] = None,
               use_session_id: bool = False):
    """Call placer-coord.py or placer-ours.py for a specific question_id."""
    print(f"Processing question_id={question_id} with baseline={baseline} ...")

    # Choose the script based on baseline
    script_name = "placer-coord.py" if baseline == "coord" else "placer-ours.py"
    
    command = [
        "python3",
        script_name,
        "--question_id", str(question_id),
        "--data_jsonl", data_jsonl,
        "--image_dir", image_dir,
        "--output_dir", output_dir,
        "--use_model", use_model,
    ]

    # Add session_id related arguments if provided
    if use_session_id:
        command.append("--use_session_id")
    if session_id:
        command.extend(["--session_id", session_id])

    try:
        subprocess.run(command, check=True)
        print(f"Successfully processed question_id={question_id}.")
        return question_id, True, ""
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Failed to process question_id={question_id}.\n"
            f"Return Code: {e.returncode}\n"
            f"Stdout: {e.stdout}\n"
            f"Stderr: {e.stderr}\n"
        )
        print(error_message)
        return question_id, False, error_message


def main():
    parser = argparse.ArgumentParser(description="Run placer baseline methods in parallel for all questions in point_questions.jsonl.")
    parser.add_argument('--data_jsonl', type=str, default='dataset/point_questions.jsonl', help="Path to point_questions.jsonl file.")
    parser.add_argument('--image_dir', type=str, default='dataset/images', help="Directory containing all images.")
    parser.add_argument('--output_dir', type=str, default='output', help="Root directory for saving outputs.")
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1, help="Number of parallel processes to use.")
    parser.add_argument('--session_id', type=str, default=None, help="Session ID for grouping outputs. If not provided, each process will generate its own.")
    parser.add_argument('--use_session_id', action='store_true', help="Use session_id in output directory and file naming.")
    parser.add_argument('--use_model', type=str, required=True)
    parser.add_argument('--baseline', type=str, choices=['coord', 'ours'], default='coord', help='Baseline method to use: coord (direct coordinates) or ours (iterative movement)')
    parser.add_argument('--specify-id', type=int, nargs='*', default=None, help='Specify question IDs to process (e.g., --specify-id 1 2 3). If not provided, process all questions.')
    args = parser.parse_args()

    # Load all question_ids from jsonl
    all_question_ids = []
    with open(args.data_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            all_question_ids.append(entry['question_id'])

    if not all_question_ids:
        print(f"No questions found in '{args.data_jsonl}'")
        return

    # Filter question_ids based on --specify-id argument
    if args.specify_id is not None:
        # Use specified question IDs
        question_ids = args.specify_id
        # Validate that all specified IDs exist in the jsonl file
        invalid_ids = [qid for qid in question_ids if qid not in all_question_ids]
        if invalid_ids:
            print(f"Warning: The following question IDs were not found in '{args.data_jsonl}': {invalid_ids}")
            question_ids = [qid for qid in question_ids if qid in all_question_ids]
    else:
        # Use all question IDs from the file
        question_ids = all_question_ids

    if not question_ids:
        print("No valid question IDs to process.")
        return

    if args.specify_id is not None:
        print(f"Processing specified question IDs: {sorted(question_ids)}")
    else:
        print("Processing all questions from file")
    print(f"Total: {len(question_ids)} questions to process with {args.workers} workers.")
    
    if args.use_session_id:
        if args.session_id:
            print(f"Using session_id: {args.session_id}")
        else:
            print("Using session_id: auto-generated per process")

    success_count = 0
    failure_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        partial_func = functools.partial(
            run_placer,
            output_dir=args.output_dir,
            data_jsonl=args.data_jsonl,
            image_dir=args.image_dir,
            use_model=args.use_model,
            baseline=args.baseline,
            session_id=args.session_id,
            use_session_id=args.use_session_id,
        )

        for qid, success, _ in executor.map(partial_func, question_ids):
            if success:
                success_count += 1
            else:
                failure_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total questions: {len(question_ids)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {failure_count}")
    print("--------------------------")


if __name__ == "__main__":
    main()
