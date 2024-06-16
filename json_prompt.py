import os
import json
import ast
import itertools
from collections import ChainMap
from execution import PromptExecutor
from . import register_node

@register_node
class JSONPromptNode:
    JSON_DIRECTORY = "/data0/zbtrs/ComfyUI/prompt" 

    def __init__(self):
        self.prompts = []

    @classmethod
    def INPUT_TYPES(cls):
        json_files = [f for f in os.listdir(cls.JSON_DIRECTORY) if f.endswith('.json')]
        json_files_options = [(file, file) for file in json_files] 
        
        return {
            "required": {
                "json_file": ("STRING", {"options": json_files_options, "default": json_files[0]} if json_files else ("STRING", {"default": ""})),
            }
        }

    RETURN_TYPES = ("SEQUENCE",)
    RETURN_NAMES = ("sequence", )
    FUNCTION = "get_next_prompt"
    CATEGORY = "test pipeline"

    def load_prompts_from_file(self, file_path):
        full_path = os.path.join(self.JSON_DIRECTORY, file_path)
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
