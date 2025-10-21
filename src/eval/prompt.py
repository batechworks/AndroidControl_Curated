template ="""
Instruction:  
Observe screenshot(s) carefully and propose the most possible element in screenshot1 with bbox_2d[x1, y1, x2, y2] and action_type that can make screenshot1 finish the following Task.

Task: {Question}. Past_Actions: {past_actions}.

Output_format:
 <answer>{{"bbox_2d": [x1, y1, x2, y2], "action_type": ACTION_TYPE}}</answer>

NOTES:
    - Output inside <answer> tag should always in JSON FORMAT
    - [x1, y1, x2, y2] should be the absolute coordinates of the screen
    - ACTION_TYPE includes: "click", "long_press", "swipe:up", "swipe:down", "swipe:left", "swipe:right", "input_text: some text", "wait", "navigate_back", "navigate_home", "open_app:app_name".
    - ONLY "click" and "long_press" need bbox_2d, others only need ACTION_TYPE.

Output_examples: 
    <answer>{{"bbox_2d": [x1, y1, x2, y2], "action_type": "click"}}</answer>
    <answer>{{"action_type": "input_text: tiger"}}</answer>   # only in keyboard page
"""