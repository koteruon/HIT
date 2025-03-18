import os
import random

import numpy as np
import torch

# def set_seed(seed, rank, world_size):
#     rng = random.Random(seed)
#     seed_per_rank = [rng.randint(0, 2**32-1) for _ in range(world_size)]
#     cur_seed = seed_per_rank[rank]
#     random.seed(cur_seed)
#     torch.manual_seed(cur_seed)
#     torch.cuda.manual_seed(cur_seed)
#     np.random.seed(cur_seed)


def set_seed(seed, rank, world_size):
    cur_seed = seed + rank  # 讓不同 rank 的 seed 可預測
    random.seed(cur_seed)
    np.random.seed(cur_seed)
    torch.manual_seed(cur_seed)
    torch.cuda.manual_seed(cur_seed)
    torch.cuda.manual_seed_all(cur_seed)  # 確保多 GPU 一致
    torch.backends.cudnn.deterministic = True  # 確保 CNN 計算可重現
    torch.backends.cudnn.benchmark = False  # 避免 cuDNN 自適應優化導致不同結果

    # 強制 PyTorch 只使用確定性算法
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 確保 CUBLAS 可重現
    torch.use_deterministic_algorithms(True)
