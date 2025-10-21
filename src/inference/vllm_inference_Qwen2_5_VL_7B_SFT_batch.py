
import os
from typing import List, Union
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
# Ensure qwen_vl_utils.py is in your project or installed
from qwen_vl_utils import process_vision_info

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Singleton Pattern Implementation ---
# 1. Global variables to cache the initialized engine components.
#    They start as None.
_VLLM_ENGINE = None
_PROCESSOR = None
_SAMPLING_PARAMS = None
system_prompt = 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.\nDuring the reasoning process, identify and state the sub-goal of the current step by enclosing it within <sub-goal> </sub-goal> tags.\nYour final answer MUST BE enclosed within <answer> </answer> tags.\n'

def _initialize_engine(model_path: str = ''):
    """
    Internal function to initialize the vLLM engine, processor, and sampling params.
    This function is designed to run ONLY ONCE.
    """
    # Use the 'global' keyword to modify the global variables defined outside this function
    global _VLLM_ENGINE, _PROCESSOR, _SAMPLING_PARAMS

    print("="*30)
    print("First call detected. Initializing vLLM Engine...")
    print(f"Loading model from: {model_path}")
    if model_path is None:
    # model_path = '/dev/shm/Qwen2.5-VL-3B-twoScreenshot-Task_894_0717_epoch10'
        model_path = '/dev/shm/Qwen2.5-VL-3B-Instruct'
    # Initialize the vLLM engine. This is the slow part.
    _VLLM_ENGINE = LLM(model=model_path, gpu_memory_utilization=0.7, trust_remote_code=True)
    
    # Initialize the processor
    _PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Define and store sampling parameters
    _SAMPLING_PARAMS = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=512,
    )
    print("Engine initialization complete.")
    print("="*30)


def _run_inference(messages: List[dict]) -> str:
    """A shared, internal function to run the actual inference."""
    # Prepare vLLM input using the globally stored processor
    llm_inputs = []
    for msg in messages:
        text = _PROCESSOR.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        try:
            image_inputs, _, _ = process_vision_info(msg, return_video_kwargs=True)
        except:
            continue

        llm_input = {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        }
        llm_inputs.append(llm_input)
    # Perform inference with the globally stored engine and params
    outputs = _VLLM_ENGINE.generate(llm_inputs, _SAMPLING_PARAMS, use_tqdm=False)
    # generated_text = outputs[0].outputs[0].text
    # print(generated_text)
    generated_list = [output.outputs[0].text for output in outputs]
    # print('generated_list is ', generated_list)
    return generated_list

def ask_model(image_path_list: Union[str, Image.Image], query_list: str, model_path: str) -> str:
    """
    Performs inference for a single image.
    It ensures the vLLM engine is initialized before proceeding.
    """
    use_system = False
    # 2. Check if the engine needs to be initialized.
    if model_path is None or not os.path.exists(model_path):
        raise ValueError(f"Model path is invalid: {model_path}")
    if _VLLM_ENGINE is None:
        _initialize_engine(model_path)

    message_list = []
    # print('image_path list is ', image_path_list)
    for image_path, query in zip(image_path_list, query_list):
        try:
            # Prepare the image
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                image = image_path
            else:
                raise ValueError(f"Invalid image path or object. {image_path}. Exsist:  {os.path.exists(image_path)}")
        except Exception as e:
            continue
    
        # Prepare the message for the model
        if use_system:
            messages = [
                {"role": "system", "content": [{'type': 'text', 'text': system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ]}
            ]
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ]}
            ]
        message_list.append(messages)

    return _run_inference(message_list)
