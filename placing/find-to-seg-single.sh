dataset=$1
object=$2
scale=$3

# Get the parent directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_FOLDER="$(dirname "$SCRIPT_DIR")"

output_path="${PARENT_FOLDER}/gaussian-grouping/data/${dataset}"

### find ###

cd "${PARENT_FOLDER}/placing"

if [ "$scale" -eq 0 ]; then
    iteration=15000
    python searcher-zooming.py --dataset ${dataset} --object ${object} --iteration ${iteration}
    zooming_file="tracking/searcher-zooming/output.txt"
    init_coord=$(head -n 1 "$zooming_file")
    loc_coord=$(head -n 1 "$zooming_file")
else
    iteration=90000
    python searcher-zooming.py --dataset ${dataset} --object ${object} --iteration ${iteration}
    zooming_file="tracking/searcher-zooming/output.txt"
    init_coord=$(head -n 1 "$zooming_file")
    loc_coord=$(tail -n 1 "$zooming_file")
fi

python searcher-locating.py --dataset ${dataset} --object ${object} --coord ${loc_coord} --iteration ${iteration}
locating_file="tracking/searcher-locating/output.txt"
obj_coord=$(tail -n 1 "$locating_file")

### seg ###

cd "${PARENT_FOLDER}/gaussian-grouping"

log_and_run() {
  echo "[ $(basename "$0") ] -> $*"
  "$@"
}

log_and_run python edit_object_removal.py -m output/${dataset} --iteration ${iteration} --operation skip --render_video --render_coord ${obj_coord}

cd "${PARENT_FOLDER}/segment-anything-2/notebooks/"
mkdir videos/${dataset}
rm videos/${dataset}/* outputs/${dataset}/* renders/${dataset}/*
cp ${output_path}/train/ours/iteration_${iteration}/renders/* videos/${dataset}/

python get_mask.py --dataset ${dataset}

mkdir ${output_path}/object_mask/
cp outputs/${dataset}/* ${output_path}/object_mask/
cp ${output_path}/train/ours/iteration_${iteration}/cam_infos.pkl ${output_path}/object_mask/

cd "${PARENT_FOLDER}/gaussian-grouping"
python train.py -s data/dining-table -r 1 -m output/${dataset} --config_file config/gaussian_dataset/train.json --train_split --train_labels --iteration ${iteration}

cd "${PARENT_FOLDER}/placing"
