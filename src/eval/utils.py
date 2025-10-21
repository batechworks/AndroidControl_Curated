import re
import json
import ast
import os
from colorama import Fore, Style

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def print_with_color(text: str, color=""):
    if color == "red":
        print(Fore.RED + text)
    elif color == "green":
        print(Fore.GREEN + text)
    elif color == "yellow":
        print(Fore.YELLOW + text)
    elif color == "blue":
        print(Fore.BLUE + text)
    elif color == "magenta":
        print(Fore.MAGENTA + text)
    elif color == "cyan":
        print(Fore.CYAN + text)
    elif color == "white":
        print(Fore.WHITE + text)
    elif color == "black":
        print(Fore.BLACK + text)
    else:
        print(text)
    print(Style.RESET_ALL)



def inside_box(gt_box, pred_box):
    # print(f'----------- gt_box {gt_box} | pred_box {pred_box} -----------')
    if pred_box is None:
        return False
    # 判断pred_box的中心点是否在gt_box内
    # print(f'----------- gt_box {gt_box} | pred_box {pred_box} -----------')
    try:
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
        # 转换为int
        pred_x1, pred_y1, pred_x2, pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
    except Exception as e:
        print(f"Error in inside_box: gt_bbox: {gt_box}, pred_bbox: {pred_box}, error: {e}")
        return False
    
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2

    if gt_x1 <= pred_center_x <= gt_x2 and gt_y1 <= pred_center_y <= gt_y2:
        return True
    else:
        return False


def distance_similarity(gt_box, pred_box, w, h, pred=None):
    # 针对gt_bbox的len为2的情况
    if pred_box is None:
        return False
    try:
        if isinstance(gt_box, list) and len(gt_box) == 2:
            gt_x, gt_y = gt_box
        
        if len(pred_box) == 4:
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
            # 计算中心点
            pred_x = (pred_x1 + pred_x2) / 2
            pred_y = (pred_y1 + pred_y2) / 2
        elif len(pred_box) == 2:
            
            pred_x, pred_y = pred_box
            if pred_x <= 1 and pred_y <= 1:
                pred_x = pred_x * w
                pred_y = pred_y * h

        else:
            print(f"Error in distance_similarity: pred_box {pred_box} is not valid| orginal pred is {pred}.")
            return False
        
        # 计算欧氏距离
        distance = ((gt_x - pred_x)/w) ** 2 + ((gt_y - pred_y)/h) ** 2

        # print(f"Distance: {distance}, gt_box: {gt_box}, pred_box: {[pred_x, pred_y]}, flag is {distance < 0.14**2}")
        if distance < 0.14**2:
            # 如果距离小于0.14, 则认为是相似的
            return True
        else:
            return False
    except Exception as e:
        print(f"Can't calculate distance similarity , GT_box: {gt_box}, Pred_box: {pred_box}, error: {e} | original pred is {pred}.")
        return False



def calculate_f1_score(ground_truth_str, predicted_str):
    # calculate the text of prediction input
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    # 增加包含关系
    if ground_truth_str in predicted_str or predicted_str in ground_truth_str:
        return True
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score >= 0.5


def correct_type(type):
    if 'tap' in type:
        type = type.replace('tap', 'click')
    if 'scroll' in type:
        type = type.replace('scroll', 'swipe')
        if 'up' in type:
            type = type.replace('up', 'down')
        elif 'down' in type:
            type = type.replace('down', 'up')
        elif 'left' in type:
            type = type.replace('left', 'right')
        elif 'right' in type:
            type = type.replace('right', 'left')
    if 'press_back' in type:
        type = type.replace('press_back', 'navigate_back')
    if "type" in type:
        type = type.replace('type', 'input_text')

    return type


def type_acc(type_gt, type_pred):
    if type_pred is None:
        return False
    type_gt = correct_type(type_gt)
    type_pred = correct_type(type_pred)

    # 计算最终的reward
    flag = False
    if ':' in type_gt and ':' in type_pred:
        type1 = type_gt.split(":")[0]
        type2 = type_gt.split(":")[1]
        # if type_pred is not None and type1 in type_pred and type2 in type_pred:
        #     flag = True
        type_pred1 = type_pred.split(":")[0]
        type_pred2 = type_pred.split(":")[1]
        #if type1 == type_pred1 and type2.lower() in type_pred2.lower() or type_pred2.lower() in type2.lower():
        if type1 == type_pred1 and calculate_f1_score(type2, type_pred2):
            flag = True
    else:
        if type_pred is not None and type_gt.lower() == type_pred.lower():
            flag = True
    # print(f'----------- type_gt {type_gt} | type_pred {type_pred} ---- {flag} -------')
    return flag



def calculate_single_android(gt_action, pred, w, h, use_distance=False):
    """
    计算单个结果的指标"""
    print(f"================== gt_action: {gt_action} | \n pred: {pred} ==================")
    if len(gt_action) == 2:
        gt_type = gt_action[0]
        gt_bbox = gt_action[1]
    else:
        gt_type = gt_action
        gt_bbox = None
        # raise ValueError("Invalid gt_action format {}.".format(gt_action))

    # 解析返回值
    bbox_pred, type_pred = parse_response(pred)
    if gt_type not in ['click', 'long_press']:
        bbox_flag = None
        type_flag = type_acc(gt_type, type_pred)
    else:
        if use_distance:
            bbox_flag = distance_similarity(gt_bbox, bbox_pred, w, h, pred=pred)
        else:
            bbox_flag = inside_box(gt_bbox, bbox_pred)
        type_flag = type_acc(gt_type, type_pred)
    if bbox_flag is None:
        type_bbox_flag = type_flag
    else:
        type_bbox_flag = bbox_flag and type_flag

    return bbox_flag, type_flag, type_bbox_flag


def calculate_multi_android(gt_action_list, pred, w, h, use_distance=False):
    """
    计算多个候选项, 只要命中一个即可"""
    bbox_flag_list = []
    type_flag_list = []
    type_bbox_flag_list = []
    for gt_action in gt_action_list:
        bbox_flag, type_flag, type_bbox_flag = calculate_single_android(gt_action, pred, w, h , use_distance=use_distance)
        bbox_flag_list.append(bbox_flag)
        type_flag_list.append(type_flag)
        type_bbox_flag_list.append(type_bbox_flag)
    # 只要有一个为True, 则为True
    bbox_flag = any(bbox_flag_list)
    type_flag = any(type_flag_list)
    type_bbox_flag = any(type_bbox_flag_list)

    return bbox_flag, type_flag, type_bbox_flag


def parse_response(res):
    # <answer> </answer>
    content_matches = re.findall(r'<answer>(.*?)</answer>', res, re.DOTALL)

    content_matches2 = re.findall(r'<action>(.*?)</action>', res, re.DOTALL)
    student_answer = (content_matches+ content_matches2)[-1].strip(
    ) if content_matches else res.strip()

    try:
        pred_dict = eval(student_answer)
        if 'bbox_2d' in pred_dict:
            bbox_pred = pred_dict['bbox_2d']
        elif 'bbox' in pred_dict:
            bbox_pred = pred_dict['bbox']
        elif 'point' in pred_dict:
            bbox_pred = pred_dict['point']
        elif 'coordinate' in pred_dict:
            bbox_pred = pred_dict['coordinate']
        else:
            bbox_pred = None
        if isinstance(bbox_pred, str):
            bbox_pred = eval(bbox_pred)
        type_value = pred_dict.get('type') if pred_dict.get('type') else pred_dict.get('action_type')
    except Exception as e:
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        # bbox_pattern = r"'?bbox_2d'?:\s*(\[\d+,\s*\d+,\s*\d+,\s*\d+\])"
        bbox_match = re.search(bbox_pattern, student_answer)
        if bbox_match:
            bbox_pred = [
                int(bbox_match.group(1)),
                int(bbox_match.group(2)),
                int(bbox_match.group(3)),
                int(bbox_match.group(4))
            ]
        else:
            bbox_pred = None
        # 提取type的值
        # type_pattern = r"[\"']?type[\"']?\s*:\s*['\"]([^'\"]+)['\"]"
        type_pattern = r"[\"']?(?:action_)?type[\"']?\s*:\s*['\"]([^'\"]+)['\"]"
        type_match = re.search(type_pattern, res)
        if type_match:
            type_value = type_match.group(1)
        else:
            type_value = None

    return bbox_pred, type_value


