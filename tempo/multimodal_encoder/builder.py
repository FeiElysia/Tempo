import copy
from pathlib import Path
from .qwen3vl_encoder import Qwen3VLTower
from .siglip_encoder import SiglipVisionTower

def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):

    vision_tower_aux_name_list = getattr(vision_tower_cfg, "mm_vision_tower_aux_list", ["Qwen/Qwen3-VL-2B-Instruct"])

    vision_tower_aux_list = []
    for vision_tower_aux_name in vision_tower_aux_name_list:
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_basename = Path(vision_tower_aux_name).name.lower()
        if "siglip" in vision_tower_basename:
            vision_tower_aux_list.append(SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "qwen3-vl" in vision_tower_basename:
            vision_tower_aux_list.append(Qwen3VLTower(vision_tower_aux_name, args=config, **kwargs))
        else:
            raise ValueError(f"Unknown vision tower: {vision_tower_basename}")

    return vision_tower_aux_list