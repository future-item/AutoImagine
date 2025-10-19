import importlib
import base64
from io import BytesIO
from PIL import Image
import io
from pathlib import Path
import ast
import re
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY')
)

def load_method(cfg):
    module_name = f"methods.{cfg.method}"
    class_name = "CountingMethod"

    module = importlib.import_module(module_name)
    method_class = getattr(module, class_name)
    method_instance = method_class(cfg)
    
    return method_instance


def pil_to_imageurl(pil_image: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    buffered.seek(0)  
    
    img_bytes = buffered.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    
    mime_type = f'image/{format.lower()}' if format.lower() != 'jpg' else 'image/jpeg'
    return f"data:{mime_type};base64,{encoded}"

def load_prompt(name: str, folder: str, root: str = "data") -> str:
    path = Path(root) / folder / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")
    
    
def parse_last_list0(text: str):
    if not text:
        raise ValueError("Empty response text")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith('[') and ln.endswith(']'):
            try:
                return ast.literal_eval(ln)
            except Exception as e:
                continue
    l = text.rfind('[')
    r = text.rfind(']')
    if l != -1 and r != -1 and r > l:
        return text[l:r+1]
        # return ast.literal_eval(text[l:r+1])
    raise ValueError("Failed to parse list from response")

def parse_last_list(text: str):
    if not text:
        raise ValueError("Empty response text")

    allowed = {"found", "missing", "yes", "no", "center", "a", "b", "c", "d"}
    tok_alt = r"(?:found|missing|yes|no|center|[abcd])"

    def norm(s): 
        return str(s).strip().strip('"\'' ).lower()

    search_blob = None
    try:
        v0 = parse_last_list0(text)
        if isinstance(v0, list) and v0:
            t = norm(v0[0])
            if t in allowed:
                return [t]
        elif isinstance(v0, str):
            search_blob = v0
    except Exception:
        pass

    if search_blob is None:
        search_blob = str(text)

    lines = [ln.strip() for ln in search_blob.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty response lines")

    # [token] / ['token'] / ["token"]
    pat_sq = re.compile(r"\[\s*(?P<q>['\"])?(?P<t>"+tok_alt+r")(?P=q)?\s*\]", flags=re.I)
    # {token} / {'token'} / {"token"}
    pat_br = re.compile(r"\{\s*(?P<q>['\"])?(?P<t>"+tok_alt+r")(?P=q)?\s*\}", flags=re.I)
    # 'token' / "token" / `token` / '''token''' / """token"""
    delim = r"(?:`{3}|'{3}|\"|'|`)"
    pat_qq = re.compile(
        r"(?<![\w])(?P<q>" + delim + r")\s*(?P<t>" + tok_alt + r")\s*(?P=q)(?![\w])",
        flags=re.I
    )
    for ln in reversed(lines):
        if ln.startswith('[') and ln.endswith(']'):
            try:
                val = ast.literal_eval(ln)
                if isinstance(val, list) and val:
                    t = norm(val[0])
                    if t in allowed:
                        return [t]
            except Exception:
                pass

        ms = list(pat_sq.finditer(ln))
        if ms:
            t = norm(ms[-1].group("t"))
            if t in allowed:
                return [t]

        ms = list(pat_br.finditer(ln))
        if ms:
            t = norm(ms[-1].group("t"))
            if t in allowed:
                return [t]

        ms = list(pat_qq.finditer(ln))
        if ms:
            t = norm(ms[-1].group("t"))
            if t in allowed:
                return [t]

    raise ValueError("Failed to parse list from response")