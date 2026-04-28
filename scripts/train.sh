source /home/xkr/.proxy_off

export CUDA_VISIBLE_DEVICES=4,5

python  scripts/train.py pi05_piper_lora  --fsdp-devices=1 --overwrite  
