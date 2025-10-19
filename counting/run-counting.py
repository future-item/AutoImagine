from omegaconf import OmegaConf
import sys, os, shutil
import json
from utils import load_method
from concurrent.futures import ThreadPoolExecutor

BASE_CONFIG_PATH = os.path.join("configs", "base.yaml")
DATASET_DIR = "dataset"
SAVE_DIR = "results"

def process_single_ann(cfg, ann, result_file):
    counting_method = load_method(cfg)
    # try:
    answer, steps = counting_method(ann)
    # except:
    #     answer, steps = -1, -1
    result = (answer == ann["gt"])
    ann["answer"] = answer
    ann["result"] = result
    ann["steps"] = steps
    with open(result_file, "a") as f:
        f.write(json.dumps(ann) + "\n")

def main(cfg):
    dataset_root = os.path.join(DATASET_DIR, cfg.dataset)
    save_folder = os.path.join(SAVE_DIR, f"{cfg.dataset}_{cfg.method}_{cfg.model.replace('/', '-').replace(' ', '-')}")
    
    cfg["save"]=save_folder
    if os.path.isdir(cfg.save):         
        shutil.rmtree(cfg.save)
    os.makedirs(cfg.save,exist_ok=True)
    
    result_file = os.path.join(cfg.save, "results.json")
    
    with open(os.path.join(dataset_root, "correct.json"), "r") as f:
        annotations = json.load(f)
    
    if cfg.single_case:
        if len(annotations) > 0:
            process_single_ann(cfg, annotations[0], result_file)
        return
    
    if cfg.num_workers <= 1: # for debug    
        for ann in annotations:
            process_single_ann(cfg, ann, result_file)
    else:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            for ann in annotations:
                ex.submit(process_single_ann, cfg, ann, result_file)
    
    
    
if __name__ == "__main__":
    base_cfg = OmegaConf.load(BASE_CONFIG_PATH)
    if len(sys.argv) > 1:
        name = sys.argv[1]
        OVERRIDE_CONFIG_PATH = os.path.join("configs", f"{name}.yaml")
        if not os.path.exists(OVERRIDE_CONFIG_PATH):
            raise SystemExit(f"No config file: {OVERRIDE_CONFIG_PATH}")
        override_cfg = OmegaConf.load(OVERRIDE_CONFIG_PATH)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg
    main(cfg)