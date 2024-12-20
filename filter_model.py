import torch

model = None
state_dict = model.state_dict()
del state_dict["key"]
torch.save(state_dict, "model_filtered.pth")

# stm_head.decoder_stages.0.fc_action.weight
# stm_head.decoder_stages.0.fc_action.bias
# stm_head.decoder_stages.1.fc_action.weight
# stm_head.decoder_stages.1.fc_action.bias
# stm_head.decoder_stages.2.fc_action.weight
# stm_head.decoder_stages.2.fc_action.bias
# stm_head.decoder_stages.3.fc_action.weight
# stm_head.decoder_stages.3.fc_action.bias
# stm_head.decoder_stages.4.fc_action.weight
# stm_head.decoder_stages.4.fc_action.bias
# stm_head.decoder_stages.5.fc_action.weight
# stm_head.decoder_stages.5.fc_action.bias
