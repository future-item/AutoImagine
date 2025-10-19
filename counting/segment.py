import os
import shutil
import time
import numpy as np
import torch
from PIL import Image

def _pick_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def _to_numpy_rgb(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return np.array(pil_image)

def _logits_to_mask_pil(mask_logits, size_hw):
    if isinstance(mask_logits, np.ndarray):
        arr = mask_logits
    else:
        arr = mask_logits.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    mask = (arr > 0).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

class Sam:
    def __init__(self, checkpoint_path: str | None = None, model_cfg: str = "sam2_hiera_l.yaml", device: str | torch.device | None = None):
        self.device = _pick_device() if device is None else device
        # segment-anything-2/checkpoints/sam2_hiera_large.pt
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                "../","segment-anything-2", "checkpoints", "sam2_hiera_large.pt"
            )
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.model_cfg = model_cfg

        self._predictor = None
        self._mode = None  # "image" or "video"
        self._build_predictor()

    def _build_predictor(self):
        # try:
        #     from sam2.build_sam import build_sam2
        #     from sam2.sam2_image_predictor import SAM2ImagePredictor
        #     sam_model = build_sam2(self.model_cfg, self.checkpoint_path, device=str(self.device))
        #     self._predictor = SAM2ImagePredictor(sam_model)
        #     self._mode = "image"
        #     return
        # except ImportError as e:
        #     print("Build SAM2 image predictor failed, try video predictor:", e)
        # except Exception as e:
        #     print("Build SAM2 image predictor failed, try video predictor:", e)
        from sam2.build_sam import build_sam2_video_predictor
        self._predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=str(self.device))
        self._mode = "video"

    def _predict_with_image_predictor(self, img_np: np.ndarray, points=None, labels=None, box=None) -> Image.Image:
        pred = self._predictor
        pred.set_image(img_np)
        out = pred.predict(points=points, labels=labels, box=box, multimask_output=False)
        if isinstance(out, dict):
            logits = out.get("logits", out.get("masks"))
        else:
            if len(out) >= 3:
                logits = out[2]
            else:
                logits = out[0].astype(np.float32) * 2 - 1
        return _logits_to_mask_pil(logits, size_hw=(img_np.shape[0], img_np.shape[1]))

    def _predict_with_video_predictor(self, img_np: np.ndarray, points=None, labels=None, box=None) -> Image.Image:
        pred = self._predictor
        state = None
        tmp_dir = None
        try:
            try:
                state = pred.init_state(frames=[img_np])
            except TypeError:
                base_tmp = os.path.join(os.path.dirname(__file__), "tmp")
                ts = int(time.time() * 1000)
                tmp_dir = os.path.join(base_tmp, f"sam2_{ts}_{os.getpid()}")
                os.makedirs(tmp_dir, exist_ok=True)

                img_path = os.path.join(tmp_dir, "00000.jpg")
                Image.fromarray(img_np).save(img_path, "JPEG", quality=95)

                state = pred.init_state(video_path=tmp_dir)

            pred.reset_state(state)
            kwargs = dict(inference_state=state, frame_idx=0, obj_id=1)
            if box is not None:
                kwargs["box"] = np.array(box, dtype=np.float32)
            if points is not None and labels is not None:
                kwargs["points"] = np.array(points, dtype=np.float32)
                kwargs["labels"] = np.array(labels, dtype=np.int32)

            _, out_obj_ids, out_mask_logits = pred.add_new_points_or_box(**kwargs)
            logits = out_mask_logits[0]
            return _logits_to_mask_pil(logits, size_hw=(img_np.shape[0], img_np.shape[1]))
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                
    def get_mask_from_point(self, pil_image: Image.Image, x: int, y: int) -> Image.Image:
        img_np = _to_numpy_rgb(pil_image)
        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        if self._mode == "image":
            return self._predict_with_image_predictor(img_np, points=points, labels=labels, box=None)
        else:
            return self._predict_with_video_predictor(img_np, points=points, labels=labels, box=None)

    def get_mask_from_bbox(self, pil_image: Image.Image, bbox: list[int] | tuple[int, int, int, int]) -> Image.Image:
        img_np = _to_numpy_rgb(pil_image)
        box = np.array(bbox, dtype=np.float32)
        if self._mode == "image":
            return self._predict_with_image_predictor(img_np, points=None, labels=None, box=box)
        else:
            return self._predict_with_video_predictor(img_np, points=None, labels=None, box=box)