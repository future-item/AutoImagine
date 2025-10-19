set -e

eval "$(conda shell.bash hook)"
conda activate sam2

tasks=(
    "classroom-2"
    "classroom-3"
    "classroom-4"
    "classroom-yq-1"
    "classroom-zjg-1"
    "dining-table-1"
    "dining-table-2"
    "dining-table-3"
    "dining-table-4"
    "dining-table-5"
    "dining-table-6"
    "dining-table-7"
    "gym-1"
    "gym-2"
    "meetingroom-1"
    "meetingroom-2"
    "meetingroom-3"
)
for task in "${tasks[@]}"; do
    echo "$dataset"
    
    python -u run-dataset.py --task "$task" > "log/run-dataset-${task}.log" 2>&1
done
