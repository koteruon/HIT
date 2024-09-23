import json

import torch
import torch.nn as nn

from hit.modeling.roi_heads.action_head.hit_structure import HITStructure

from .lr_scheduler import HalfPeriodCosStepLR, WarmupMultiStepLR


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith("encoder.patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or var_name.startswith("encoder.blocks"):
        if var_name.startswith("encoder.blocks"):
            var_name = var_name[8:]
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, lr_scale=1.0
):
    parameter_group_names = {}
    parameter_group_vars = {}

    # 在这里修改区分scale，encoder一个学习率，其他人一个学习率
    # layer_decay: 需要加上get_num_layer和get_layer_scale
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (len(param.shape) == 1 or name.endswith(".bias") or name in skip_list) and name.startswith("encoder."):
            group_name = "no_decay_encoder"
            this_weight_decay = 0.0
            scale = 1.0
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay_others"
            this_weight_decay = 0.0
            scale = lr_scale
        elif name.startswith("encoder."):
            group_name = "decay_encoder"
            this_weight_decay = weight_decay
            scale = 1.0
        else:
            group_name = "decay_others"
            this_weight_decay = weight_decay
            scale = lr_scale

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id) * scale

            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def make_optimizer(cfg, model):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    if "vit" in cfg.MODEL.BACKBONE.CONV_BODY.lower():
        layer_decay = cfg.MODEL.BACKBONE.ViT.LAYER_DECAY < 1.0
        if layer_decay:
            assigner = LayerDecayValueAssigner(
                list(
                    cfg.MODEL.BACKBONE.ViT.LAYER_DECAY ** (cfg.MODEL.BACKBONE.ViT.DEPTH + 1 - i)
                    for i in range(cfg.MODEL.BACKBONE.ViT.DEPTH + 2)
                )
            )
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = set(cfg.MODEL.BACKBONE.ViT.NO_WEIGHT_DECAY)
        print("Skip weight decay list: ", skip_weight_decay_list)
        weight_decay = cfg.MODEL.BACKBONE.ViT.WEIGHT_DECAY
        backbone_parameters = get_parameter_groups(
            model.backbone,
            weight_decay,
            skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )

        non_bn_parameters = []
        hit_parameters = []
        for name, p in model.named_parameters():
            if "backbone" in name:
                continue
            elif "hit_structure" in name:
                hit_parameters.append(p)
            else:
                non_bn_parameters.append(p)

        optim_params = []
        optim_params = backbone_parameters.copy()
        optim_params.append(
            {
                "params": hit_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.IA_LR_FACTOR,
            }
        )
        optim_params.append(
            {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr": cfg.SOLVER.BASE_LR}
        )

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == sum(len(p["params"]) for p in backbone_parameters) + len(
            hit_parameters
        ) + len(non_bn_parameters), "parameter size does not match: {} + {} + {} != {}".format(
            sum(len(p["params"]) for p in backbone_parameters),
            len(hit_parameters),
            len(non_bn_parameters),
            len(list(model.parameters())),
        )
        print(
            "vit {}, hit {}, non bn {}".format(
                sum(len(p["params"]) for p in backbone_parameters),
                len(hit_parameters),
                len(non_bn_parameters),
            )
        )
    else:
        bn_parameters = []
        non_bn_parameters = []
        hit_parameters = []

        frozn_bn = cfg.MODEL.BACKBONE.FROZEN_BN
        # only running_mean and var frozen
        if frozn_bn:
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
        for name, param in model.named_parameters():
            if ("backbone" in name) and ("bn" in name):
                bn_parameters.append((name, param))
            elif "hit_structure" in name:
                hit_parameters.append((name, param))
            else:
                non_bn_parameters.append((name, param))

        optim_params = []
        for name, param in bn_parameters:
            if not param.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BN
            optim_params.append({"params": [param], "weight_decay": weight_decay, "lr": lr})
        for name, param in hit_parameters:
            if not param.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in name:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            lr = lr * cfg.SOLVER.IA_LR_FACTOR
            optim_params.append({"params": [param], "weight_decay": weight_decay, "lr": lr})
        for name, param in non_bn_parameters:
            if not param.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in name:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            optim_params.append({"params": [param], "weight_decay": weight_decay, "lr": lr})

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(bn_parameters) + len(hit_parameters) + len(
            non_bn_parameters
        ), "parameter size does not match: {} + {} + {} != {}".format(
            len(bn_parameters),
            len(hit_parameters),
            len(non_bn_parameters),
            len(list(model.parameters())),
        )
        print(
            "bn {}, hit {}, non bn {}".format(
                len(bn_parameters),
                len(hit_parameters),
                len(non_bn_parameters),
            )
        )

    optimizer = torch.optim.SGD(optim_params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_optimizer_1(cfg, model):
    params = []
    bn_param_set = set()
    transformer_param_set = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_param_set.add(name + ".weight")
            bn_param_set.add(name + ".bias")
        elif isinstance(module, HITStructure):
            for param_name, _ in module.named_parameters(name):
                transformer_param_set.add(param_name)
    for key, value in model.named_parameters():
        if not value.requires_grad:
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
