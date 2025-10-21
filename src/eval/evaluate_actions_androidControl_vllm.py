import re
import os
import json
import time
import gc
import cv2
import torch
import pandas as pd
import sys
from inference.vllm_inference_Qwen2_5_VL_7B_SFT_batch import ask_model
from tqdm import tqdm
import random
from tqdm import tqdm
from utils import print_with_color, calculate_single_android, calculate_multi_android, correct_type
from prompt import template

benchmark_map_dict = {
"android_control_low_point": "benchmark_resource/android_control_low_point.json",
"android_control_high_point": "benchmark_resource/android_control_high_point.json",
"android_control_low_bbox": "benchmark_resource/android_control_low_bbox.json",
"android_control_high_bbox": "benchmark_resource/android_control_high_bbox.json",
"android_control_high_task-improved": "benchmark_resource/android_control_high_task-improved.json",
}

def get_all_response_android_batch(benchmark_json_path, result_path, model_path=None, random_number=None, batch_size=16, use_improve_task=False, exist_result_path=''):
    """
    获取所有的response
    :param benchmark_json_path: benchmark json文件路径
    :param result_path: 结果文件路径
    :param random_number: 随机选择的benchmark数量
    :param save_single: 是否每个sample保存一次
    """

    ## 判断是high还是low
    # 读取benchmark_par_dir下的所有文件夹
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmark_dict_list = json.load(f)
    result_dir = os.path.dirname(result_path)

    if random_number is not None:
        # 随机选择random_number个benchmark_dict
        print(f'random number is {random_number}')
        if random_number > len(benchmark_dict_list):
            print_with_color(f"random_number {random_number} is larger than benchmark_dict_list length {len(benchmark_dict_list)}, set to {len(benchmark_dict_list)}", 'yellow')
            random_number = len(benchmark_dict_list)
        random.seed(42)  # 设置随机种子以确保可重复性
        benchmark_dict_list = random.sample(benchmark_dict_list, random_number)

    existing_results_dict = {}
    if use_improve_task:
        if os.path.exists(exist_result_path):
            with open(exist_result_path, 'r') as f:
                existing_results_list = json.load(f)
            existing_results_dict = {item['image']: item for item in existing_results_list}

    # 如果结果文件存在, 则提醒
    cur_result_dict = {}
    if os.path.exists(result_path):
        inp = input(f'Result file "{result_path}" exists, Do you want to delete it? (y/n)')
        if inp == 'y':
            os.remove(result_path)
            print_with_color(f"Result file {result_path} deleted !", 'yellow')
        else:
            print_with_color(f"Result file {result_path} exists, and you don't want to delete it. Pls rename the {result_path} !", 'red')
            with open(result_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                cur_result_dict = {os.path.join(result_dir, item['image']): item for item in existing_data}
                if len(cur_result_dict) == len(benchmark_dict_list):
                    print_with_color(f"All images have been processed in {result_path}, exit now !", 'green')
                    return result_path
    else:
        os.makedirs(result_dir, exist_ok=True)

    res_all = []
    # 遍历dict
    all_image_paths = []
    all_prompts = []
    res_existing = []
    for benchmark_dict in tqdm(benchmark_dict_list):
        # 获取图片路径, 这里result_dir就是图片的根目录
        image_path = os.path.join(result_dir, benchmark_dict['image'])
        if image_path in cur_result_dict:
            response_org = cur_result_dict[image_path]['response']
            benchmark_dict['response'] = response_org
            res_existing.append(benchmark_dict)
            print_with_color(f"Image {image_path} already processed, skipping.", 'yellow')
            continue
        if 'gt_input_text' in benchmark_dict:
            if benchmark_dict['gt_input_text'] == "no input text":
                gt_action_type = benchmark_dict['gt_action'] 
            else:
                gt_action_type = benchmark_dict['gt_action'] + ":" + benchmark_dict['gt_input_text'].lower()
        else:
            gt_action_type = benchmark_dict['gt_action']
        if 'scroll' in gt_action_type:
            gt_action_type = correct_type(gt_action_type)
        if 'gt_bbox' not in benchmark_dict:
            gt_bbox = benchmark_dict['gt_max_bbox']
        else:
            gt_bbox = benchmark_dict['gt_bbox']
        gt_action = [gt_action_type, gt_bbox]
        if use_improve_task:
            # 多个候选动作
            gt_action = [gt_action]  

        # print_with_color(f"================== image_path: {image_path} | \n gt_action: {gt_action} ==================", "green")
        # 获取问题
        task = benchmark_dict['instruction']
        past_actions = benchmark_dict.get('history', '')
        benchmark_dict['action_bounds'] = gt_action

        # 获取候选动作
        if 'candidate_actions' in benchmark_dict and len(benchmark_dict['candidate_actions']) > 0:
            candidate_actions_list = benchmark_dict['candidate_actions']
            for candidate in candidate_actions_list:
                action_type = candidate.get('action_type', '')
                action_bounds = candidate.get('action_bounds', [])
                # 对齐Ground Truth的格式
                gt_action.append([action_type, action_bounds])

        if use_improve_task:
            if 'revised_task' in benchmark_dict or 'revised_memory' in benchmark_dict or image_path not in existing_results_dict:
                task = benchmark_dict.get('revised_task', benchmark_dict['instruction'])
                past_actions = benchmark_dict.get('revised_memory', benchmark_dict.get('history', '')) 
            else:
                # 取原始结果
                benchmark_res_existing = existing_results_dict[image_path]
                response_org = benchmark_res_existing['response']
                benchmark_dict['response'] = response_org
                res_existing.append(benchmark_dict)
                print_with_color(f"Image {image_path} not in improve_task, use original result.", 'yellow')
                continue
        
        prompt = template.format(
            Question=task,
            past_actions=past_actions
        )
        all_image_paths.append(image_path)
        all_prompts.append(prompt)
    
    # 分批处理
    for i in tqdm(range(0, len(all_image_paths), batch_size)):
        image_list = all_image_paths[i:i+batch_size]
        prompt_list = all_prompts[i:i+batch_size]
        benchmark_dict_tmp_list = benchmark_dict_list[i:i+batch_size]
        st = time.time()
        res = ask_model(image_list, prompt_list, model_path)
        # print_with_color(f"================== ask_model time: {time.time() - st} ==================", "green")

        for j, benchmark_dict in enumerate(benchmark_dict_tmp_list):
            benchmark_dict['response'] = res[j]
            res_all.append(benchmark_dict)
    
    # 保存结果
    print_with_color(f"Saving results to {result_path}, len exsiting {len(res_existing)} and len new {len(res_all)}", 'green')
    res_all = res_existing + res_all
    with open(result_path, 'w') as f:
        json.dump(res_all, f, indent=4)

    return result_path


def calculate_metrics(result_path, save_name='result.xlsx', random_number=None, use_distance=False, use_improve_task=False):
    """
    计算指标
    :param result_path: 结果文件路径, json格式
    :param save_name: 保存结果的excel文件名
    :param random_number: 随机数, 用于随机选择结果文件
    :param use_distance: 是否使用距离来计算box的准确率
    :param use_improve_task: 是否使用改进的任务描述, 即最后一个benchmark
    :return: box_acc, type_acc, type_bbox_acc
    """
    if not os.path.exists(result_path) and random_number is not None:
        result_path = result_path.replace(f'_{random_number}.json', '.json')
    # 读取结果
    with open(result_path, 'r') as f:
        res_all = json.load(f)

    result_dir = os.path.dirname(result_path)

    # 计算指标
    res_new = []   # 将flag保存到文件中
    box_flag_count = 0
    type_flag_count = 0
    type_bbox_flag_count = 0
    total_count = len(res_all)
    no_bbox_count = 0
    
    all_type_count_dict = {}  # 记录每个type的准确率
    if random_number is not None:
        # 随机选择random_number个benchmark_dict
        if random_number > len(res_all):
            print_with_color(f"random_number {random_number} is larger than res_all length {len(res_all)}, set to {len(res_all)}", 'yellow')
            random_number = len(res_all)
        random.seed(42)  # 设置随机种子以确保可重复性
        res_all = random.sample(res_all, random_number)
        if not os.path.exists(result_path):
            result_path = result_path.replace('.json', f'_{random_number}.json')
            # 保存结果
            with open(result_path, 'w') as f:
                json.dump(res_all, f, indent=4)
    
    # 遍历dict
    for benchmark_dict in tqdm(res_all):

        image_path = os.path.join(result_dir, benchmark_dict['image'])
        h, w ,_ = cv2.imread(image_path).shape
        gt_action = benchmark_dict['action_bounds']
        gt_type = gt_action[0][0] if use_improve_task else gt_action[0]
        pred = benchmark_dict['response']

        if use_improve_task:
             bbox_flag, type_flag, type_bbox_flag = calculate_multi_android(gt_action, pred, w, h, use_distance=False)
        else:
            bbox_flag, type_flag, type_bbox_flag = calculate_single_android(gt_action, pred, w, h, use_distance=use_distance)

        # 计算每个type的准确率
        gt_type = gt_type.split(":")[0]
        if gt_type not in all_type_count_dict:
            all_type_count_dict[gt_type] = [0, 0]
        else:
            all_type_count_dict[gt_type][0] += 1
            if type_flag:
                all_type_count_dict[gt_type][1] += 1

        if gt_type not in ['click', 'long_press']:
            no_bbox_count += 1
        benchmark_dict['bbox_flag'] = bbox_flag
        benchmark_dict['type_flag'] = type_flag
        benchmark_dict['type_bbox_flag'] = type_bbox_flag
        res_new.append(benchmark_dict)
        print_with_color(f"===== gt_action: {gt_action} | \n pred: {pred} =====| type_flag: {type_flag} | box_flag: {bbox_flag}", "cyan")

        # 计算指标
        if bbox_flag:
            box_flag_count += 1
        if type_flag:
            type_flag_count += 1
        if type_bbox_flag:
            type_bbox_flag_count += 1

    # 计算准确率, 且都取到小数点后2位
    # total_count = len(res_all)
    box_acc =  round(box_flag_count / (total_count - no_bbox_count) *100, 1)
    type_acc = round(type_flag_count / total_count *100, 1)
    type_bbox_acc = round(type_bbox_flag_count / total_count *100, 1)

    # 计算每个type的准确率
    # 计算所有类型的准确率
    all_type_count_dict = {k: (v[0], round(v[1]/v[0]*100, 1) if v[0]>0 else 0) for k, v in all_type_count_dict.items()}

    print_with_color(f">>>>>>>>>>>>>>>>> Total Count: {total_count}", 'yellow')
    print_with_color(f"Box Accuracy: {box_acc:.4f}", 'green')
    print_with_color(f"Type Accuracy: {type_acc:.4f}", 'green')
    print_with_color(f"Type and Box Accuracy: {type_bbox_acc:.4f}", 'green')
    print_with_color(f"all_type_count_dict: {all_type_count_dict}", 'green')

    # 保存结果
    save_json_path = result_path.replace('.json', '_flag.json')
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(res_new, f, indent=4, ensure_ascii=False)
    print_with_color(f"Results saved to {save_json_path}, length is {len(res_new)}", 'green')
    
    tmp_dict = {
        "result_path": result_path,
        'Grouding_Accuracy': box_acc,
        'Action_type_Accuracy': type_acc,
        'Step_Success_Rate': type_bbox_acc,
        "all_type_count_dict": all_type_count_dict,
        "template": template,
        'total_count': total_count,
    }
    # 将结果存入xlsx文件
    result_xlsx_path = os.path.join(os.path.dirname(result_path), save_name)
    if random_number is not None:
        result_xlsx_path = result_xlsx_path.replace('.xlsx', f'_{random_number}.xlsx')

    # save to xlsx_result folder
    result_xlsx_path = result_xlsx_path.replace('/result/', '/xlsx_result/')
    if os.path.exists(result_xlsx_path):
        df = pd.read_excel(result_xlsx_path)
        # 判断 result_path json路径 是否存在
        if result_path in df['result_path'].values:
            df = df[df['result_path'] != result_path]
            # 添加新的行
            df = pd.concat([df, pd.DataFrame([tmp_dict])], ignore_index=True)
        else:
            # 添加新的行
            df = pd.concat([df, pd.DataFrame([tmp_dict])], ignore_index=True)
    
        df.to_excel(result_xlsx_path, index=False)
    else:
        df = pd.DataFrame([tmp_dict])
        df.to_excel(result_xlsx_path, index=False)

    print_with_color(f"Results saved to {result_xlsx_path}", 'green')

    return box_acc, type_acc, type_bbox_acc


def evaluate_one_benchmark(model_path, only_metrics=False,  save_name='RL_benchmark_easy.xlsx', random_number=None, 
                           benchmark_name='', result_dir='results', batch_size=16, use_improve_task=False, exist_result_path=''):
    """
    评估单个 benchmark
    :param result_name: 结果文件名
    :param only_metrics: 是否只计算指标, 为True, 则不进行llm推理
    :param save_name: 保存结果的excel文件名
    :param random_number: 随机数, 用于随机选择结果文件
    :param benchmark_name: benchmark名称, 用于区分不同的benchmark, 从benchmark_map_dict中选择
    :param result_dir: 结果文件夹
    :param batch_size: 批处理大小"""
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(result_dir, f"{os.path.basename(model_path)}_{benchmark_name}_result.json")
    benchmark_json_path = benchmark_map_dict.get(benchmark_name, None)

    if random_number is not None:
        result_path = result_path.replace('.json', f'_{random_number}.json')
    if not only_metrics:
        result_path = get_all_response_android_batch(benchmark_json_path, result_path, model_path=model_path, random_number=random_number,
                                                     batch_size=batch_size, use_improve_task=use_improve_task, exist_result_path=exist_result_path)
    # 计算指标, result_path 存储的是response的结果 
    if 'point' in benchmark_name:
        use_distance = True
    else:
        use_distance = False

    print('------------ use improve task is ', use_improve_task)
    calculate_metrics(result_path, save_name=save_name, random_number=random_number, use_distance=use_distance, use_improve_task=use_improve_task)
    print_with_color(f"================== result_path: {result_path} ==================", 'green')


def evaluate_one(model_path=None, benchmark_name=None, only_metrics=False, save_name='RL_benchmark.xlsx', random_number=None,
                 batch_size=16, result_dir='results', use_improve_task=False, exist_result_path=''):
    """
    评估单个模型
    :param result_name: 结果文件名, 以json结尾
    :param only_metrics: 是否只计算指标, 为True, 则不进行llm推理
    :param save_name: 保存结果的excel文件名
    :param random_number: 随机数, 用于随机选择结果文件
    :param benchmark_name: benchmark名称, 用于区分不同的benchmark, 从benchmark_map_dict中选择
    :param result_dir: 结果文件夹
    :param batch_size: 批处理大小
    :param use_improve_task: 是否使用改进的任务描述, 即最后一个benchmark
    :param exist_result_path: 如果use_improve_task为True, 则需要提供已经存在的结果文件路径, 用于加载已经存在的结果
    """

    # 评估easy benchmark
    save_name = save_name.replace('.xlsx', f'_{benchmark_name}.xlsx')
    evaluate_one_benchmark(model_path, only_metrics=only_metrics, save_name=save_name, 
                           random_number=random_number, benchmark_name=benchmark_name, 
                           result_dir=result_dir, batch_size=batch_size, 
                           use_improve_task=use_improve_task, exist_result_path=exist_result_path)

def evaluate_all(model_path=None,  save_name='RL_benchmark.xlsx', only_metrics=False, random_number=None, 
                  batch_size=16, result_dir='results'):
    """
    评估所有benchmark
    :param model_path: 模型路径
    :param save_name: 保存结果的excel文件名
    :param only_metrics: 是否只计算指标, 为True, 则不进行推理
    :param random_number: 随机数, 用于随机选择结果文件,
    :param batch_size: 批处理大小
    :param result_dir: 结果文件夹
    :param use_improve_task: 是否使用改进的任务描述, 即最后一个benchmark
    :param exist_result_path: 如果use_improve_task为True, 则需要提供已经存在的结果文件路径, 用于加载已经存在的结果

    """
    
    for benchmark_name in list(benchmark_map_dict.keys())[4:]:
        if 'improved' in benchmark_name:
            use_improve_task = True
            exist_result_path = os.path.join(result_dir, f"{os.path.basename(model_path)}_android_control_high_bbox_result.json")
        else:
            use_improve_task = False
            exist_result_path = ''
        evaluate_one(model_path=model_path, benchmark_name=benchmark_name, only_metrics=only_metrics, 
                     save_name=save_name, random_number=random_number, batch_size=batch_size, result_dir=result_dir,
                     use_improve_task=use_improve_task, exist_result_path=exist_result_path)
        print_with_color(f"Evaluated {benchmark_name} successfully.", 'green')



            
if __name__ == "__main__":

    # store_true default is False
    # store_false default is True
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--save_name', type=str, default='Android_control.xlsx', help='Name of the Excel file to save results')
    parser.add_argument('--random_number', type=int, default=None, help='random number of total datasets to eval')
    parser.add_argument('--only_metrics', action='store_true', help='Whether to only calculate metrics without running inference')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--image_dir', type=str, default='results', help='Directory to Your images')


    args = parser.parse_args()
    # 调用evaluate_all函数
    evaluate_all(only_metrics=args.only_metrics, save_name=args.save_name, random_number=args.random_number, 
                 model_path=args.model_path, result_dir=args.image_dir, batch_size=args.batch_size)
