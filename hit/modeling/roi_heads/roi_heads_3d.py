import torch

from hit.modeling import registry

from .action_head.action_head import build_roi_action_head


@registry.COMBINED_3D_ROIHEADS.register("Combined3dROIHeads")
class Combined3dROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(Combined3dROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, slow_features, fast_features, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):
        result, loss_action, loss_weight, accuracy_action = self.action(
            slow_features, fast_features, boxes, objects, keypoints, extras, part_forward
        )

        return result, loss_action, loss_weight, accuracy_action

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + "." + key
                    weight_map[new_key] = val
        return weight_map


###AVA 特別制定模型 start
@registry.COMBINED_3D_ROIHEADS.register("Combined3dROIHeads_AVA")
class Combined3dROIHeads_AVA(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(Combined3dROIHeads_AVA, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, slow_features, fast_features, boxes, extras={}, part_forward=-1):
        result, loss_action, loss_weight, accuracy_action = self.action(
            slow_features, fast_features, boxes, extras, part_forward
        )

        return result, loss_action, loss_weight, accuracy_action

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + "." + key
                    weight_map[new_key] = val
        return weight_map


###AVA 特別制定模型 end


def build_3d_roi_heads(cfg, dim_in):
    roi_heads = []
    roi_heads.append(("action", build_roi_action_head(cfg, dim_in)))

    if roi_heads:
        func = registry.COMBINED_3D_ROIHEADS[cfg.MODEL.COMBINED_3D_ROIHEADS.STRUCTURE]
        roi_heads = func(cfg, roi_heads)

    return roi_heads
