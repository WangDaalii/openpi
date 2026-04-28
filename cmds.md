conda create -p /media/wenfei/165C02105C01EAF7/anaconda3/envs/openpi python=3.11

GIT_LFS_SKIP_SMUDGE=1 uv sync --cache-dir ~/workspace/openpi/uv_cache 
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --cache-dir ~/workspace/openpi/uv_cache


export OPENPI_DATA_HOME=/media/wenfei/165C02105C01EAF7/openpi/openpi_cache
mkdir -p "$OPENPI_DATA_HOME"
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"

cd /media/wenfei/165C02105C01EAF7/openpi
export HF_LEROBOT_HOME=/media/wenfei/165C02105C01EAF7/pipolicy_lerobot/lerobot_datasets
python  scripts/train.py pi05_biso101_threeview_lora  --overwrite 

\
--wandb-enabled false


export UV_VENV_DISABLE=1

mkdir -p /home/wenfei/.cache/huggingface/lerobot/datasets/lerobot_datasets
ln -s /media/wenfei/165C02105C01EAF7/pipolicy_lerobot/lerobot_datasets/lerobot_foldthetowel_run02_v21 \
      /home/wenfei/.cache/huggingface/lerobot/doukeyidoukeyi/lerobot_foldthetowel_run02_v21

修改：
/scripts/train.py
/src/openpi/training/config.py
~/anaconda3/envs/openpi/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py
/src/openpi/training/data_loader.py

rsync -avhPz /media/wenfei/165C02105C01EAF7/openpi/src  xkr@192.168.22.140:/home/xkr/workspace/openpi/src
rsync -avhPz /media/wenfei/165C02105C01EAF7/pipolicy_lerobot/lerobot_datasets/lerobot_foldthetowel_run02_v21  xkr@192.168.22.140:/home/xkr/.cache/huggingface/lerobot/doukeyidoukeyi
rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_biso101_threeview_vitfrozen_lora/pi05_ft_20260115_194121 /media/wenfei/165C02105C01EAF7/openpi/checkpoints/pi05_biso101_threeview_vitfrozen_lorarsync -avhPz /media/wenfei/165C02105C01EAF7/pipolicy_lerobot/lerobot_datasets/lerobot_foldthetowel_run02_v21  xkr@192.168.22.140:/home/xkr/.cache/huggingface/lerobot/doukeyidoukeyi
rsync -avhPz /media/wenfei/165C02105C01EAF7/openpi/lerobot_datasets/lerobot_foldthetowel_run02_v21  xkr@192.168.22.140:/home/xkr/.cache/huggingface/lerobot/doukeyidoukeyi
rsync -avhPz /media/wenfei/165C02105C01EAF7/pipolicy_lerobot/lerobot_datasets/lerobot_foldthetowel_run02_v21  xkr@192.168.22.140:/home/xkr/.cache/huggingface/lerobot/doukeyidoukeyi

rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_biso101_threeview_lora/pi05_lora_ft_20260116_183926/30000 /media/wenfei/165C02105C01EAF7/openpi/checkpoints/pi05_biso101_threeview_lora/pi05_lora_ft_20260116_183926
rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_biso101_threeview_lora/pi05_lora_ft_20260119_102814/18000 /media/wenfei/165C02105C01EAF7/openpi/checkpoints/pi05_biso101_threeview_lora/pi05_lora_ft_20260119_102814

rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_biso101_threeview_lora_OnDatasetV3/pi05_lora_ft_foldtowel3-2_LowBase_DeltaAction_resumeon3-1_run03/16000 /media/wenfei/165C02105C01EAF7/workspace/openpi/checkpoints/pi05_biso101_threeview_lora_OnDatasetV3/pi05_lora_ft_foldtowel3-2_LowBase_DeltaAction_resumeon3-1_run03
rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_biso101_threeview_lora_OnDatasetV3/pi05_lora_ft_foldtowel3-2_LowBase_DeltaAction_run03/39999 /media/wenfei/165C02105C01EAF7/workspace/openpi/checkpoints/pi05_biso101_threeview_lora_OnDatasetV3/pi05_lora_ft_foldtowel3-2_LowBase_DeltaAction_run03
rsync -avhPz xkr@192.168.22.140:/home/xkr/workspace/openpi/checkpoints/pi05_onearmso101_threeview_lora_OnDatasetV3/pi05_lora_ft_tnpbanana1_DeltaAction_run02/16000 /media/wenfei/165C02105C01EAF7/workspace/openpi/checkpoints/pi05_onearmso101_threeview_lora_OnDatasetV3/pi05_lora_ft_tnpbanana1_DeltaAction_run02


python  scripts/train.py pi05_onearmso101_threeview_lora_OnDatasetV3  --overwrite

python  scripts/train.py pi05_biso101_threeview_lora  --overwrite 
python  scripts/train.py pi05_biso101_threeview_lora  --resume  --exp-name=pi05_lora_ft_20260116_160824
python  scripts/train.py pi05_biso101_threeview_lora_OnDatasetV3 --overwrite
/home/xkr/workspace/openpi/src/openpi/models_pytorch/preprocessing_pytorch.py

python  scripts/train.py pi05_piper_lora  --overwrite 

\\



##
1 修改 ~/anaconda3/envs/openpi/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py
2 修改 ~/workspace/openpi/src/openpi/transforms.py