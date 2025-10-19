
from openai import OpenAI
import os
from PIL import Image
from utils import pil_to_imageurl, client

class CountingMethod:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_folder = os.path.join("dataset", cfg.dataset)
        self.client = client
        
    def process_response_text(self, response_text):    
        return int(response_text)
    
    def __call__(self, ann):
        image_path = os.path.join(self.data_folder, ann["image"])
        pil_image = Image.open(image_path).convert("RGB")
        image_url = pil_to_imageurl(pil_image)
        
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=0.0,
            messages=[ {
                "role": "user",
                "content": [ 
                    {   
                        "type": "text", 
                        "text": f'how many {ann["object"]} are there in the image? You should output a single number without any other characters.' 
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": image_url },
                    }
                ]
            } ] 
        )
        return self.process_response_text(response.choices[0].message.content), 1