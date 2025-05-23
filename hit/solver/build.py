import torch
import torch.nn as nn

from hit.modeling.backbone.slowfast import SlowFast
from hit.modeling.roi_heads.action_head.hit_structure import HITStructure
from hit.modeling.roi_heads.action_head.SkateFormer_2D_for_hit import SkateFormer

from .lr_scheduler import HalfPeriodCosStepLR, WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    bn_param_set = set()
    backbone_param_set = set()
    transformer_param_set = set()
    skateformer_param_set = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_param_set.add(name + ".weight")
            bn_param_set.add(name + ".bias")
        elif isinstance(module, SlowFast):
            for param_name, _ in module.named_parameters(name):
                backbone_param_set.add(param_name)
        elif isinstance(module, HITStructure):
            for param_name, _ in module.named_parameters(name):
                transformer_param_set.add(param_name)
        elif isinstance(module, SkateFormer):
            for param_name, _ in module.named_parameters(name):
                skateformer_param_set.add(param_name)
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if key in backbone_param_set:
            continue
        if cfg.MODEL.SKATEFORMER_WEIGHT != "":
            if key in skateformer_param_set:
                continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key in bn_param_set:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BN
        elif "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key in transformer_param_set:
            lr = lr * cfg.SOLVER.IA_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    scheduler = cfg.SOLVER.SCHEDULER
    if scheduler not in ("half_period_cosine", "warmup_multi_step"):
        raise ValueError("Scheduler not available")
    if scheduler == "warmup_multi_step":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS if cfg.SOLVER.WARMUP_ON else 0,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif scheduler == "half_period_cosine":
        return HalfPeriodCosStepLR(
            optimizer,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS if cfg.SOLVER.WARMUP_ON else 0,
            max_iters=cfg.SOLVER.MAX_ITER,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
