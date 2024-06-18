import os
import json
import ast
import itertools
from collections import ChainMap
from execution import PromptExecutor
from . import register_node
import folder_paths
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import folder_paths
import latent_preview
import node_helpers
import numpy as np
import safetensors.torch
import torch


@register_node
class ImageInputNode:
    def __init__(self):
        self.images = []

    @classmethod
    def INPUT_TYPES(cls):
        json_folder = os.path.join(folder_paths.base_path,"image_input")
        json_files = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')]
        
        print(json_files)
        
        return {"required": { "json_file": (json_files, ),
                             }}

    RETURN_TYPES = ("SEQUENCE",)
    RETURN_NAMES = ("sequence", )
    FUNCTION = "get_next_image"
    CATEGORY = "test pipeline"

    def load_images_from_file(self, file_path):
        full_path = file_path
        with open(full_path, 'r') as file:
            data = json.load(file)
            self.images = [image['path'] for image in data.get('images', [])]

        if not self.images:
            raise ValueError(f"No prompts found in the JSON file: {full_path}")

    def get_next_image(self, json_file):
        if not self.images:
            self.load_images_from_file(json_file)
        
        images_as_string = str(self.images)
        
        return (ast.literal_eval(images_as_string), )

@register_node
class JSONPromptNode:
    
    def __init__(self):
        self.prompts = []

    @classmethod
    def INPUT_TYPES(cls):
        json_folder = os.path.join(folder_paths.base_path,"prompt")
        json_files = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')]
        
        print(json_files)
        
        return {"required": { "json_file": (json_files, ),
                             }}

    RETURN_TYPES = ("SEQUENCE",)
    RETURN_NAMES = ("sequence", )
    FUNCTION = "get_next_prompt"
    CATEGORY = "test pipeline"

    def load_prompts_from_file(self, file_path):
        full_path = file_path
        with open(full_path, 'r') as file:
            data = json.load(file)
            self.prompts = [prompt['text'] for prompt in data.get('prompts', [])]

        if not self.prompts:
            raise ValueError(f"No prompts found in the JSON file: {full_path}")

    def get_next_prompt(self, json_file):
        if not self.prompts:
            self.load_prompts_from_file(json_file)
        
        prompts_as_string = str(self.prompts)
        
        return (ast.literal_eval(prompts_as_string), )


@register_node
class MakeJob:
    """Turns a sequence into a job with one attribute."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sequence": ("SEQUENCE", ),
                "name": ("STRING", {"default": ''}),
            },
        }

    RETURN_TYPES = ("JOB", "INT")
    RETURN_NAMES = ("job", "count")
    FUNCTION = "go"
    CATEGORY = "test pipeline"

    def merge_dicts(self, *dicts):
        return dict(itertools.chain.from_iterable(d.items() for d in dicts))

    def go(self, sequence, name):
        result = [{name: value} for value in sequence]
        return (result, len(result))
    
@register_node
class ExtractImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "attributes": ("ATTRIBUTES",)
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "go"
    CATEGORY = "test pipeline"
    
    def go(self,attributes):
        image_path = next(iter(attributes.values()))
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    
@register_node
class FormatAttributes:
    """Applies attributes to a format string."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "attributes": ("ATTRIBUTES",),
                "format": ("STRING", {'default': '', 'multiline': True, "dynamicPrompts": False})
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    FUNCTION = "go"
    CATEGORY = "test pipeline"

    def go(self, attributes, format):
        return (format.format(**attributes), )

@register_node
class JobIterator:
    """Magic node that runs the workflow multiple times until all steps are done."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "job": ("JOB",),
                "start_step": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("ATTRIBUTES", "INT", "INT")
    RETURN_NAMES = ("attributes", "count", "step")
    FUNCTION = "go"
    CATEGORY = "test pipeline"

    def go(self, job, start_step):
        print(f'JobIterator: {start_step + 1} / {len(job)}')
        return (job[start_step], len(job), start_step)


orig_execute = PromptExecutor.execute


def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
    print("Prompt executor has been patched by Job Iterator!")
    orig_execute(self, prompt, prompt_id, extra_data, execute_outputs)

    steps = None
    job_iterator = None
    for k, v in prompt.items():
        try:
            if v['class_type'] == 'JobIterator':
                steps = self.outputs[k][1][0]
                job_iterator = k
                break
        except KeyError:
            continue
    else:
        return

    while prompt[job_iterator]['inputs']['start_step'] < (steps - 1):
        prompt[job_iterator]['inputs']['start_step'] += 1
        orig_execute(self, prompt, prompt_id, extra_data, execute_outputs)


PromptExecutor.execute = execute
