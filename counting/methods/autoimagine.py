
from openai import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
from utils import pil_to_imageurl, load_prompt, parse_last_list, client
import numpy as np
import random
from segment import Sam
import cv2

class CountingMethod:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_folder = os.path.join("dataset", cfg.dataset)
        self.client = client
        
        # init
        self.steps = 0
        self.cur_save_dir = None
        # prompt
        if self.cfg.few_shot:
            self.prompt_searcher_checking = load_prompt("searcher-checking", "fewshot")
            self.prompt_searcher_locating = load_prompt("searcher-locating", "fewshot")
            self.prompt_searcher_checking_dot = load_prompt("searcher-checking-dot","fewshot")
            self.prompt_searcher_checking_mask = load_prompt("searcher-checking-mask","fewshot")
        else:
            self.prompt_searcher_checking = load_prompt("searcher-checking", "normal")
            self.prompt_searcher_locating = load_prompt("searcher-locating", "normal")
            self.prompt_searcher_checking_dot = load_prompt("searcher-checking-dot","normal")
            self.prompt_searcher_checking_mask = load_prompt("searcher-checking-mask", "normal") 
        
        self.sam = Sam()
        
    def _check_exist(self, image_url, obj):
        self.steps += 1
        for attempt in range(self.cfg.max_retries):
            try:
                if self.cfg.few_shot:
                    prompt_text = self.prompt_searcher_checking.replace("{object_name}", str(obj))
                    parts = prompt_text.split("<image>")
                    content = []
                    image_dir = os.path.join("prompts", "fewshot", "image")
                    num_ph = len(parts) - 1
                    for i, seg in enumerate(parts):
                        seg = seg.strip()
                        if seg:
                            content.append({"type": "text", "text": seg})
                        if i < num_ph:
                            if i == num_ph - 1:
                                url = image_url
                                content.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                ex_path = os.path.join(image_dir, f"checking{i+1}.png")
                                if os.path.isfile(ex_path):
                                    im = Image.open(ex_path).convert("RGB")
                                    ex_url = pil_to_imageurl(im)
                                    content.append({"type": "image_url", "image_url": {"url": ex_url}})
                    response = self.client.chat.completions.create(
                        model=self.cfg.model,
                        temperature=0.0,
                        messages=[{"role": "user", "content": content}],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.cfg.model,
                        temperature=0.0,
                        messages=[ {
                            "role": "user",
                            "content": [ 
                                {   
                                    "type": "text", 
                                    "text": self.prompt_searcher_checking
                                },
                                {   
                                    "type": "text", 
                                    "text": f"Object to be found: {obj}." 
                                },
                                {
                                    "type": "image_url",
                                    "image_url": { "url": image_url },
                                }
                            ]
                        } ],
                    )
                response_text = response.choices[0].message.content
                print(("\n\n[CHECK_EXIST]\n"))
                print(response_text)
                

                with open(f"{self.cur_save_dir}/output.txt", "a") as file:
                    file.write(("\n\n[CHECK_EXIST]\n"))
                    file.write(response_text)
                    

                check_result = parse_last_list(response_text)

                with open(f"{self.cur_save_dir}/output.txt", "a") as file:
                    file.write(f'\n\n{check_result[0]}\n')

                if check_result[0] == 'found':
                    return 1
                elif check_result[0] == 'missing':
                    return 0
                else:
                    raise 
            
            except Exception as e:
                print(f"Error: {e}")
                if attempt < self.cfg.max_retries - 1:
                    print(f"Retrying {attempt + 1} times...")
                else:
                    print("Error: Max retries reached.")
                    raise 
        
    def _check(self, image_url, obj, prompt, check_type=""):
        self.steps += 1
        for attempt in range(self.cfg.max_retries):
            try:
                if self.cfg.few_shot and check_type == "MASK":
                    prompt_text = prompt.replace("{object_name}", str(obj))
                    parts = prompt_text.split("<image>")
                    content = []
                    image_dir = os.path.join("prompts", "fewshot", "image")
                    num_ph = len(parts) - 1
                    for i, seg in enumerate(parts):
                        seg = seg.strip()
                        if seg:
                            content.append({"type": "text", "text": seg})
                        if i < num_ph:
                            if i == num_ph - 1:
                                url = image_url
                                content.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                ex_path = os.path.join(image_dir, f"mask{i+1}.png")
                                if os.path.isfile(ex_path):
                                    im = Image.open(ex_path).convert("RGB")
                                    ex_url = pil_to_imageurl(im)
                                    content.append({"type": "image_url", "image_url": {"url": ex_url}})
                    response = self.client.chat.completions.create(
                        model=self.cfg.model,
                        temperature=0.0,
                        messages=[{"role": "user", "content": content}],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.cfg.model,
                        temperature=0.0,
                        messages=[ {
                            "role": "user",
                            "content": [ 
                                {   
                                    "type": "text", 
                                    "text": prompt
                                },
                                {   
                                    "type": "text", 
                                    "text": f"Object Description: {obj}" 
                                },
                                {
                                    "type": "image_url",
                                    "image_url": { "url": image_url },
                                }
                            ]
                        } ] ,
                    )
                response_text = response.choices[0].message.content
                print(f"\n\n[CHECK {check_type}]\n")
                print(response_text)
                
                with open(f"{self.cur_save_dir}/output.txt", "a") as file:
                    file.write((f"\n\n[CHECK {check_type}]\n"))
                    file.write(response_text)
                    

                check_result = parse_last_list(response_text)
                
                if check_result[0] == 'no':
                    return 0
                elif check_result[0] == 'yes':
                    return 1
                else:
                    raise 
            
            except Exception as e:
                print(f"Error: {e}")
                if attempt < self.cfg.max_retries - 1:
                    print(f"Retrying {attempt + 1} times...")
                else:
                    print("Error: Max retries reached.")
                    raise 

    def _locate_direction(self, image_url, obj, x_pixel, y_pixel):
        self.steps += 1
        for attempt in range(self.cfg.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=0.0,
                    messages=[ {
                        "role": "user",
                        "content": [ 
                            {   
                                "type": "text", 
                                "text": self.prompt_searcher_locating 
                            },
                            {   
                                "type": "text", 
                                "text": f"Object to be located: {obj}." 
                            },
                            {
                                "type": "image_url",
                                "image_url": { "url": image_url },
                            }
                        ]
                    } ] 
                )
                response_text = response.choices[0].message.content
                print("\n\n[LOCATE]\n")
                print(response_text)
                
                with open(f"{self.cur_save_dir}/output.txt", "a") as file:
                    file.write(("\n\n[LOCATE]\n"))
                    file.write(f"{x_pixel} {y_pixel}\n")
                    file.write(response_text)
                    
                locate_result = parse_last_list(response_text)

                return str(locate_result[0]).strip().lower() if locate_result else None

            except Exception as e:
                print(f"Error: {e}")
                if attempt < self.cfg.max_retries - 1:
                    print(f"Retrying {attempt + 1} times...")
                else:
                    print("Error: Max retries reached.")
                    raise 
             
    def _get_mask_from_point(self, pil_image, x, y):
        mask_pil = self.sam.get_mask_from_point(pil_image, x, y)
        return np.array(mask_pil, dtype=np.uint8)
         
    def _get_mask_from_bbox(self, pil_image, bbox):
        mask_pil = self.sam.get_mask_from_bbox(pil_image, bbox)
        return np.array(mask_pil, dtype=np.uint8)
    
    def _bbox_from_mask(self, binary_mask: np.ndarray, low_p: float = 2.0, high_p: float = 98.0):
        ys, xs = np.where(binary_mask != 0)
        if ys.size == 0 or xs.size == 0:
            raise ValueError("empty mask cannot compute bbox")
        x1, x2 = np.percentile(xs, [low_p, high_p])
        y1, y2 = np.percentile(ys, [low_p, high_p])
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def _masked_image(self, image: Image.Image, binary_mask: np.ndarray):
        mask = Image.fromarray(binary_mask.astype(np.uint8)).convert("L")
        black = Image.new("RGB", image.size, (0, 0, 0))
        return Image.composite(image, black, mask)
    
    def _draw_dot(self, image: Image.Image, x: int, y: int, radius: int = 6) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 0), width=2)
        return img
    
    def _overlay_directions(self, image: Image.Image, x: int, y: int, scale_px: int):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        dirs = {'a': (0, 1), 'b': (0, -1), 'c': (-1, 0), 'd': (1, 0)}
        pad = 3
        for k, (dx, dy) in dirs.items():
            tx = x + dx * scale_px
            ty = y + dy * scale_px
            w, h = draw.textlength(k, font=font), font.size
            draw.rectangle([tx - w/2 - pad, ty - h/2 - pad, tx + w/2 + pad, ty + h/2 + pad], fill=(255,255,255))
            draw.text((tx - w/2, ty - h/2), k, fill=(255,0,0), font=font)
        return img
    
    def _crop_and_zoom(self, image: Image.Image, cx: int, cy: int, zooming_scale: float):
        W, H = image.size
        half_w = int(W * zooming_scale / 2)
        half_h = int(H * zooming_scale / 2)
        l = max(0, cx - half_w)
        r = min(W, cx + half_w + 1)
        u = max(0, cy - half_h)
        d = min(H, cy + half_h + 1)
        crop = image.crop((l, u, r, d))
        if zooming_scale <= 0:
            zooming_scale = 1e-6
        new_w = max(1, int(round(crop.size[0] / zooming_scale)))
        new_h = max(1, int(round(crop.size[1] / zooming_scale)))
        zoom = crop.resize((new_w, new_h), Image.BICUBIC)
        cx_zoom = int(round((cx - l) / zooming_scale))
        cy_zoom = int(round((cy - u) / zooming_scale))
        return zoom, (cx_zoom, cy_zoom)
    
    def _inpaint_and_mark(
        self,
        image: Image.Image,
        binary_mask: np.ndarray,
        dilate_iter: int = 2,
        inpaint_radius: int = 3,
        feather_px: int = 3,     
        add_grain: bool = True    
    ):
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[..., 0]
        W, H = image.size
        if binary_mask.shape != (H, W):
            binary_mask = np.array(Image.fromarray(binary_mask.astype(np.uint8)).resize((W, H), Image.NEAREST))
        mask255 = ((binary_mask > 0).astype(np.uint8)) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask_dil = cv2.dilate(mask255, kernel, iterations=int(dilate_iter))

        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask_dil, inpaintRadius=int(inpaint_radius), flags=cv2.INPAINT_TELEA)

        if feather_px and feather_px > 0:
            sigma = float(feather_px)
            soft = cv2.GaussianBlur(mask_dil, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32) / 255.0
            soft = soft * ((mask_dil > 0).astype(np.float32))
            soft = soft[..., None]  # (H,W,1)
            out_bgr = inpainted_bgr.astype(np.float32) * soft + bgr.astype(np.float32) * (1.0 - soft)
            out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)
        else:
            out_bgr = inpainted_bgr

        if add_grain:
            rng = np.random.default_rng()
            noise = rng.normal(0.0, 2.0, size=out_bgr.shape).astype(np.float32)  
            if feather_px and feather_px > 0:
                out_bgr = np.clip(out_bgr.astype(np.float32) + noise * soft, 0, 255).astype(np.uint8)
            else:
                m3 = (mask_dil > 0).astype(np.float32)[..., None]
                out_bgr = np.clip(out_bgr.astype(np.float32) + noise * m3, 0, 255).astype(np.uint8)

        out_img = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        if self.cur_save_dir and os.path.isdir(self.cur_save_dir):
            out_img.save(os.path.join(self.cur_save_dir, "inpainted.png"))
        return out_img, mask_dil
    
    def _blackout_and_mark(
        self,
        image: Image.Image,
        binary_mask: np.ndarray,
        dilate_iter: int = 2,
        feather_px: int = 0
    ):
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[..., 0]
        W, H = image.size
        if binary_mask.shape != (H, W):
            binary_mask = np.array(
                Image.fromarray(binary_mask.astype(np.uint8)).resize((W, H), Image.NEAREST)
            )
        mask255 = ((binary_mask > 0).astype(np.uint8)) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask_dil = cv2.dilate(mask255, kernel, iterations=int(dilate_iter))

        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if feather_px and feather_px > 0:
            soft = cv2.GaussianBlur(mask_dil, (0, 0), sigmaX=float(feather_px), sigmaY=float(feather_px)).astype(np.float32) / 255.0
            soft = soft * ((mask_dil > 0).astype(np.float32))
            soft = soft[..., None]  # (H,W,1)
            out_bgr = bgr.astype(np.float32) * (1.0 - soft) 
            out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)
        else:
            out_bgr = bgr.copy()
            m3 = (mask_dil > 0)
            out_bgr[m3] = 0  

        out_img = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        if self.cur_save_dir and os.path.isdir(self.cur_save_dir):
            out_img.save(os.path.join(self.cur_save_dir, "blackout.png"))
        return out_img, mask_dil
    
    def __call__(self, ann):
        image_path = os.path.join(self.data_folder, ann["image"])
        pil_image = Image.open(image_path).convert("RGB")
        save_dir = os.path.join(self.cfg.save, str(ann["id"]))
        os.makedirs(save_dir, exist_ok=False)
        
        obj = ann["object"]
        self.steps = 0
        cnt = 0
        
        W, H = pil_image.size
        inpainted_mask = np.zeros((H, W), dtype=np.uint8)
        moving_dict = {'a': (0, 1), 'b': (0, -1), 'c': (-1, 0), 'd': (1, 0)}
        max_shape = max(W, H)

        while True:
            self.cur_save_dir = os.path.join(save_dir, f'count-{cnt}')
            os.makedirs(self.cur_save_dir, exist_ok=False)
            pil_image.save(os.path.join(self.cur_save_dir, 'image.png'))
            Image.fromarray(inpainted_mask).save(os.path.join(self.cur_save_dir, 'inpainted_mask.png'))
            
            if not self._check_exist(pil_to_imageurl(pil_image), obj):
                break
            
            x_pixel = W // 2
            y_pixel = H // 2
            check_result = 0
            found_mask = None
            
            for it in range(self.cfg.iters):
                if inpainted_mask[y_pixel, x_pixel] == 0:
                    img_dot = self._draw_dot(pil_image, x_pixel, y_pixel, radius=6)
                    img_dot.save(os.path.join(self.cur_save_dir, f'dot_{it:02d}.png'))
                    check_result = self._check(pil_to_imageurl(img_dot), obj, self.prompt_searcher_checking_dot,check_type="DOT")
                    if check_result:
                        mask = self._get_mask_from_point(pil_image, x_pixel, y_pixel)
                        try:
                            bbox = self._bbox_from_mask(mask)
                            mask = self._get_mask_from_bbox(pil_image, bbox)
                        except Exception:
                            pass
                        img_masked = self._masked_image(pil_image, mask)
                        img_masked.save(os.path.join(self.cur_save_dir, f'masked_{it:02d}.png'))
                        
                        mask_result = True
                        for i in range(2):
                            mask_result = self._check(pil_to_imageurl(img_masked), obj, self.prompt_searcher_checking_mask, check_type="MASK")
                            if not mask_result:
                                break
                        
                        check_result += int(mask_result)                                   
                        if check_result > 1:
                            found_mask = mask
                            print(f'check result:{check_result}')
                            with open(f"{self.cur_save_dir}/output.txt", "a") as file:
                                print(f'\n\n{check_result}', file=file)
                                file.write(f'{x_pixel} {y_pixel}\n')
                            break
                
                if it < 5: # no zoom in early iter
                    moving_img = pil_image
                    zooming_scale = 1.0
                    moving_x, moving_y = x_pixel, y_pixel
                else:
                    zooming_scale = (1 - max(it - 4, 0) / self.cfg.iters * 0.75)
                    moving_img, (moving_x, moving_y) = self._crop_and_zoom(pil_image, x_pixel, y_pixel, zooming_scale)

                scale = max((self.cfg.iters - it) / self.cfg.iters * self.cfg.max_scale,
                            random.uniform(0.5, 1.5) * self.cfg.min_scale)

                dist_ratio = getattr(self.cfg, "dir_dist_ratio", 0.35)

                scale_px = int(max(1, max_shape * scale * dist_ratio))

                moving_image = self._overlay_directions(moving_img, moving_x, moving_y, scale_px)
                moving_image.save(os.path.join(self.cur_save_dir, f'moving_{it:02d}.png'))
                key = self._locate_direction(pil_to_imageurl(moving_image), obj, x_pixel, y_pixel)
                if key not in moving_dict:
                    continue
                dx, dy = moving_dict[key]

                dx, dy = moving_dict[key]
                sel_tx = int(round(moving_x + dx * scale_px))
                sel_ty = int(round(moving_y + dy * scale_px))

                l = int(round(x_pixel - moving_x * zooming_scale))
                u = int(round(y_pixel - moving_y * zooming_scale))
                x_pixel = int(round(l + sel_tx * zooming_scale))
                y_pixel = int(round(u + sel_ty * zooming_scale))

                x_pixel = max(0, min(W - 1, x_pixel))
                y_pixel = max(0, min(H - 1, y_pixel))
            
            if check_result > 1 and found_mask is not None:
                if self.cfg.few_shot:
                    pil_image, dilated_mask = self._blackout_and_mark(pil_image, found_mask)
                else:
                    pil_image, dilated_mask = self._inpaint_and_mark(pil_image, found_mask)
                inpainted_mask = np.bitwise_or(
                    inpainted_mask,
                    ((dilated_mask > 0).astype(np.uint8) * 255)
                )
                cnt += 1
            else:
                break
            
        return cnt, self.steps
            