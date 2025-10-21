export CUDA_VISIBLE_DEVICES='1'
work_dir=/home/langchao/Project/dataset/benchmark/github
export PYTHONPATH="$work_dir:$PYTHONPATH"

cd $work_dir
python eval/evaluate_actions_androidControl_vllm.py --model_path /dev/shm/Qwen2.5-VL-3B-Instruct  --save_name AC_text.xlsx --image_dir /home/langchao/Project/dataset/GUI_benchmark --random_number 30