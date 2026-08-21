import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from util.misc import get_torch_dtype


def _prelude_coda(cfg):
    """Envelope depth for middle_* strategies: number of independent (untied)
    layers at the front/back of the stack. Defaults to 1/1, the original MoR
    middle_cycle; Huginn-style P8 R12 C8 uses 8/8."""
    p = int(cfg.recursive.get("prelude_depth", 1) or 1)
    c = int(cfg.recursive.get("coda_depth", 1) or 1)
    return p, c


@torch.no_grad()
def average_initialize(cfg, model):
    """
    Initialize layers to average layer weights in looping index.
    For example, the model with 6 layers and 3 num_recursion will be initialized as
    | 1-3-5  2-4-6 | 1-3-5  2-4-6 | 1-3-5  2-4-6 | for CYCLE strategy, and
    | 1-2  1-2 | 3-4  3-4 | 5-6  5-6 | for SEQUENCE strategy.
    """
    sharing_strategy = cfg.recursive.sharing
    init_strategy = cfg.recursive.initialization
    torch_dtype = get_torch_dtype(cfg)
    p, c = _prelude_coda(cfg)

    if cfg.recursive.num_recursion == 1:
        base_depth = cfg.recursive.base_depth
        num_recursion = model.config.num_hidden_layers // base_depth
    else:
        num_recursion = cfg.recursive.num_recursion
        if sharing_strategy in ["cycle", "sequence"]:
            base_depth = int(model.config.num_hidden_layers // num_recursion)
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            base_depth = int((model.config.num_hidden_layers - p - c) // num_recursion)
    
    new_state_dict = {}
    cur_state_dict = model.state_dict()
    lora_init_dict = {}
    
    if sharing_strategy in ["middle_cycle", "middle_sequence"]:
        # First and last layer initialization
        for idx in list(range(p)) + list(range(model.config.num_hidden_layers - c, model.config.num_hidden_layers)):
            # Self Attention
            keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.self_attn.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.self_attn.{key}.weight"])
        
            # Fully Connected 
            keys = ["gate_proj", "up_proj", "down_proj"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.mlp.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.mlp.{key}.weight"])
            
            # Layer Normalization
            keys = ["input_layernorm", "post_attention_layernorm"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.{key}.weight"])

    for i in range(base_depth):  
        # Source and target indices          
        if sharing_strategy == "cycle":
            src_idxs = tar_idxs = [i + rec * base_depth for rec in range(num_recursion)]
        elif sharing_strategy == "sequence":
            src_idxs = tar_idxs = range(i * num_recursion, (i + 1) * num_recursion)
        elif sharing_strategy == "middle_cycle":
            src_idxs = tar_idxs = [p + i + rec * base_depth for rec in range(num_recursion)]  # offset by prelude depth: prelude layers are not shared
        elif sharing_strategy == "middle_sequence":
            src_idxs = tar_idxs = range(p + i * num_recursion, p + (i + 1) * num_recursion)  # offset by prelude depth: prelude layers are not shared
        else:
            raise ValueError(f"Invalid sharing strategy: {sharing_strategy}")

        # Self Attention
        keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for key in keys:
            src_weight = sum([deepcopy(cur_state_dict[f"model.layers.{idx}.self_attn.{key}.weight"]) for idx in src_idxs]) / num_recursion
            
            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.self_attn.{key}.weight"] = src_weight
                
                lora_init_dict[f"{tar_idx}_{key}"] = cur_state_dict[f"model.layers.{tar_idx}.self_attn.{key}.weight"] - src_weight
                lora_init_dict[f"{tar_idx}_{key}"] = torch.as_tensor(lora_init_dict[f"{tar_idx}_{key}"], dtype=torch_dtype)

        # Fully Connected
        keys = ["gate_proj", "up_proj", "down_proj"]
        for key in keys:
            src_weight = sum([deepcopy(cur_state_dict[f"model.layers.{idx}.mlp.{key}.weight"]) for idx in src_idxs]) / num_recursion
            
            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.mlp.{key}.weight"] = src_weight
                
                lora_init_dict[f"{tar_idx}_{key}"] = cur_state_dict[f"model.layers.{tar_idx}.mlp.{key}.weight"] - src_weight
                lora_init_dict[f"{tar_idx}_{key}"] = torch.as_tensor(lora_init_dict[f"{tar_idx}_{key}"], dtype=torch_dtype)

        # Layer Normalization
        keys = ["input_layernorm", "post_attention_layernorm"]
        for key in keys:
            src_weight = sum([deepcopy(cur_state_dict[f"model.layers.{idx}.{key}.weight"]) for idx in src_idxs]) / num_recursion

            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.{key}.weight"] = src_weight

    model.load_state_dict(new_state_dict, strict=False)
    return model, lora_init_dict


@torch.no_grad()
def selection_initialize(cfg, model):
    """
    Initialize layers to selected layer weights with certain step.
    This function supports stepwise, lower, and upper initialization.
    For example, the model with 6 layers and 3 num_recursion will be initialized as
    | 1 4 | 1 4 | 1 4 | for CYCLE strategy, and
    | 1 1 | 4 4 | 6 6 | for SEQUENCE strategy.
    """
    sharing_strategy = cfg.recursive.sharing
    init_strategy = cfg.recursive.initialization
    torch_dtype = get_torch_dtype(cfg)
    p, c = _prelude_coda(cfg)

    if cfg.recursive.num_recursion == 1:
        base_depth = cfg.recursive.base_depth
        num_recursion = model.config.num_hidden_layers // base_depth
    else:
        num_recursion = cfg.recursive.num_recursion
        if sharing_strategy in ["cycle", "sequence"]:
            base_depth = int(model.config.num_hidden_layers // num_recursion)
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            base_depth = int((model.config.num_hidden_layers - p - c) // num_recursion)

    new_state_dict = {}
    cur_state_dict = model.state_dict()
    lora_init_dict = {}
    
    if sharing_strategy in ["middle_cycle", "middle_sequence"]:
        # First and last layer initialization
        for idx in list(range(p)) + list(range(model.config.num_hidden_layers - c, model.config.num_hidden_layers)):
            # Self Attention
            keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.self_attn.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.self_attn.{key}.weight"])
        
            # Fully Connected 
            keys = ["gate_proj", "up_proj", "down_proj"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.mlp.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.mlp.{key}.weight"])
            
            # Layer Normalization
            keys = ["input_layernorm", "post_attention_layernorm"]
            for key in keys:
                new_state_dict[f"model.layers.{idx}.{key}.weight"] = deepcopy(cur_state_dict[f"model.layers.{idx}.{key}.weight"])

    # Source indices
    if init_strategy == "stepwise":
        if sharing_strategy in ["cycle", "sequence"]:
            src_idxs = [round(i) for i in np.linspace(0, model.config.num_hidden_layers - 1, base_depth)]
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            src_idxs = [round(i) for i in np.linspace(p, model.config.num_hidden_layers - c - 1, base_depth)]

    elif init_strategy == "lower":
        if sharing_strategy in ["cycle", "sequence"]:
            src_idxs = range(base_depth)
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            src_idxs = range(p, base_depth + p)

    elif init_strategy == "upper":
        if sharing_strategy in ["cycle", "sequence"]:
            src_idxs = range(model.config.num_hidden_layers - base_depth, model.config.num_hidden_layers)
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            src_idxs = range(model.config.num_hidden_layers - c - base_depth, model.config.num_hidden_layers - c)

    elif init_strategy == "random":
        if sharing_strategy in ["cycle", "sequence"]:
            src_idxs = sorted(np.random.choice(model.config.num_hidden_layers, base_depth, replace=False))
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            src_idxs = sorted(np.random.choice(range(p, model.config.num_hidden_layers - c), base_depth, replace=False))
                 
    else:
        raise ValueError(f"Invalid initialization strategy: {init_strategy}")
    
    for i, src_idx in enumerate(src_idxs):    
        # Target indices        
        if sharing_strategy == "cycle":
            tar_idxs = [i + rec * base_depth for rec in range(num_recursion)]
        elif sharing_strategy == "sequence":
            tar_idxs = range(i * num_recursion, (i + 1) * num_recursion)
        elif sharing_strategy == "middle_cycle":
            tar_idxs = [p + i + rec * base_depth for rec in range(num_recursion)]
        elif sharing_strategy == "middle_sequence":
            tar_idxs = range(p + i * num_recursion, p + (i + 1) * num_recursion)
        else:
            raise ValueError(f"Invalid sharing strategy: {sharing_strategy}")

        # Self Attention
        keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for key in keys:
            src_weight = deepcopy(cur_state_dict[f"model.layers.{src_idx}.self_attn.{key}.weight"])
            
            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.self_attn.{key}.weight"] = src_weight
                
                lora_init_dict[f"{tar_idx}_{key}"] = cur_state_dict[f"model.layers.{tar_idx}.self_attn.{key}.weight"] - src_weight
                lora_init_dict[f"{tar_idx}_{key}"] = torch.as_tensor(lora_init_dict[f"{tar_idx}_{key}"], dtype=torch_dtype)

        # Fully Connected
        keys = ["gate_proj", "up_proj", "down_proj"]
        for key in keys:
            src_weight = deepcopy(cur_state_dict[f"model.layers.{src_idx}.mlp.{key}.weight"])

            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.mlp.{key}.weight"] = src_weight
                
                lora_init_dict[f"{tar_idx}_{key}"] = cur_state_dict[f"model.layers.{tar_idx}.mlp.{key}.weight"] - src_weight
                lora_init_dict[f"{tar_idx}_{key}"] = torch.as_tensor(lora_init_dict[f"{tar_idx}_{key}"], dtype=torch_dtype)

        # Layer Normalization
        keys = ["input_layernorm", "post_attention_layernorm"]
        for key in keys:
            src_weight = deepcopy(cur_state_dict[f"model.layers.{src_idx}.{key}.weight"])

            for tar_idx in tar_idxs:
                new_state_dict[f"model.layers.{tar_idx}.{key}.weight"] = src_weight

    model.load_state_dict(new_state_dict, strict=False)
    return model, lora_init_dict


# TODO: implement picking non-uniform number of layers for each recursion
INITIALIZATION = {
    "average": average_initialize,
    "stepwise": selection_initialize,
    "lower": selection_initialize,
    "upper": selection_initialize,
    "random": selection_initialize,
}


def sharing_strategy(cfg, model):
    """
    Downscale strategy for CYCLE sharing weights in LlamaForCausalLM model.
    model argument should be LlamaForCausalLM.
    Scales trainable parameters according to num_recursion within the base depth.
    """
    sharing_strategy = cfg.recursive.sharing
    init_strategy = cfg.recursive.initialization
    p, c = _prelude_coda(cfg)

    if cfg.recursive.num_recursion == 1:
        base_depth = cfg.recursive.base_depth
        num_recursion = model.config.num_hidden_layers // base_depth
    else:
        num_recursion = cfg.recursive.num_recursion
        if sharing_strategy in ["cycle", "sequence"]:
            base_depth = int(model.config.num_hidden_layers // num_recursion)
            if base_depth * num_recursion != model.config.num_hidden_layers:
                warnings.warn("Total number of layers should be divisible by num_recursion. Adjusting the number of layers.")
                indices = [round(i) for i in np.linspace(0, model.config.num_hidden_layers - 1, base_depth * num_recursion)]
                model.model.layers = nn.ModuleList([model.model.layers[idx] for idx in indices])
                model.config.num_hidden_layers = base_depth * num_recursion
                
        elif sharing_strategy in ["middle_cycle", "middle_sequence"]:
            base_depth = int((model.config.num_hidden_layers - p - c) // num_recursion)
            if base_depth * num_recursion != model.config.num_hidden_layers - p - c:
                warnings.warn("Total number of layers should be divisible by num_recursion. Adjusting the number of layers.")
                indices = [round(i) for i in np.linspace(p, model.config.num_hidden_layers - c - 1, base_depth * num_recursion)]
                model.model.layers = nn.ModuleList(list(model.model.layers[:p]) + [model.model.layers[idx] for idx in indices] + list(model.model.layers[len(model.model.layers) - c:]))
                model.config.num_hidden_layers = base_depth * num_recursion + p + c
                
        else:
            raise ValueError(f"Invalid sharing strategy: {sharing_strategy}")

    model, lora_init_dict = INITIALIZATION[init_strategy](cfg, model)

    if cfg.recursive.num_recursion == 1:
        if sharing_strategy == "cycle":
            model.model.layers = model.model.layers[:base_depth]
        elif sharing_strategy == "sequence":
            step = model.config.num_hidden_layers // base_depth
            model.model.layers = model.model.layers[::step]
        model.config.num_hidden_layers = base_depth
    else:    
        for layer_idx in range(base_depth):

            if sharing_strategy == "cycle":
                idxs = [layer_idx + rec * base_depth for rec in range(num_recursion)]
            elif sharing_strategy == "sequence":
                idxs = range(layer_idx * num_recursion, (layer_idx + 1) * num_recursion)
            elif sharing_strategy == "middle_cycle":
                idxs = [p + layer_idx + rec * base_depth for rec in range(num_recursion)]  # offset by prelude depth: prelude layers are not shared
            elif sharing_strategy == "middle_sequence":
                idxs = range(p + layer_idx * num_recursion, p + (layer_idx + 1) * num_recursion)  # offset by prelude depth: prelude layers are not shared
            else:
                raise ValueError(f"Invalid sharing strategy: {sharing_strategy}")

            # Self Attention
            keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
            for key in keys:
                ref_weights = getattr(model.model.layers[idxs[0]].self_attn, key).weight

                for idx in idxs[1:]:
                    getattr(model.model.layers[idx].self_attn, key).weight = ref_weights # [key]

            # Fully connected
            keys = ["gate_proj", "up_proj", "down_proj"]
            for key in keys:
                ref_weights = getattr(model.model.layers[idxs[0]].mlp, key).weight

                for idx in idxs[1:]:
                    getattr(model.model.layers[idx].mlp, key).weight = ref_weights # [key]

            # Layer Normalization
            if cfg.recursive.ln_share:
                keys = ["input_layernorm", "post_attention_layernorm"]
                for key in keys:
                    ref_weights = getattr(model.model.layers[idxs[0]], key).weight

                    for idx in idxs[1:]:
                        getattr(model.model.layers[idx], key).weight = ref_weights #[key]
    
    return model, lora_init_dict