# scripts/verify_video_coverage.py
import numpy as np
import torch
import openpi.training.config as _config
from openpi.training.data_loader import create_torch_dataset

def find_episode_key(sample: dict):
    # 常见候选：你跑一下它会告诉你到底用哪个
    candidates = [
        "episode_index", "episode_id", "episode", "episode_name",
        "video_path", "video", "videos", "recording_id",
        "task_index",  # 备选（不一定等于视频）
    ]
    for k in candidates:
        if k in sample:
            return k
    return None

cfg = _config.get_config("pi05_biso101_threeview_vitfrozen_lora")  # 改成你的 config 名
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

# 注意：这里用“未做 openpi transforms 的原始 LeRobotDataset”，更容易拿到 episode 信息
ds = create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)

print("Dataset len:", len(ds))
s0 = ds[0]
print("Raw sample keys:", list(s0.keys()))

ep_key = find_episode_key(s0)
if ep_key is None:
    raise RuntimeError("找不到 episode/video 标识字段。请把 Raw sample keys 发我，我告诉你用哪个字段。")
print("Using episode key:", ep_key)

# 用和训练一致的随机机制
gen = torch.Generator().manual_seed(cfg.seed)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    generator=gen,
)

seen = set()
max_batches = 2000  # 想更稳就加大
for i, batch in enumerate(loader):
    v = batch[ep_key]
    # v 可能是 tensor / list / numpy
    if torch.is_tensor(v):
        v = v.cpu().numpy()
    v = np.asarray(v).reshape(-1)
    for x in v:
        # bytes/str/int 都处理一下
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8", errors="ignore")
        seen.add(str(x))
    if (i + 1) % 50 == 0:
        print(f"batches={i+1}, seen={len(seen)}")

    # 如果你知道总数就是 50，可以提前停
    if len(seen) >= 50:
        print("✅ reached 50 unique videos/episodes")
        break

print("Final seen count:", len(seen))
print("Seen examples:", list(seen)[:10])
