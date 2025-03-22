import torch
import os
import numpy as np
import torch.backends.cudnn as cudnn
import random
from thop import profile, clever_format
from copy import deepcopy
def slurm_dist_init(port=23456):
    
    def init_parrots(host_addr, rank, local_rank, world_size, port):
        os.environ['MASTER_ADDR'] = str(host_addr)
        os.environ['MASTER_PORT'] = str(port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    def init(host_addr, rank, local_rank, world_size, port):
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                            rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()


    def parse_host_addr(s):
        if '[' in s:
            left_bracket = s.index('[')
            right_bracket = s.index(']')
            prefix = s[:left_bracket]
            first_number = s[left_bracket+1:right_bracket].split(',')[0].split('-')[0]
            return prefix + first_number
        else:
            return s

    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    ip = parse_host_addr(os.environ['SLURM_STEP_NODELIST'])

    if torch.__version__ == 'parrots':
        init_parrots(ip, rank, local_rank, world_size, port)
        from parrots import config
        config.set_attr('engine', 'timeout', value=500)
    else:
        init(ip, rank, local_rank, world_size, port)

    return rank, local_rank, world_size

def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def random_choice(num_choice, layers):
    return list(np.random.randint(num_choice, size=layers))

def get_model_flops_params(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size)
    macs, params = profile(deepcopy(model), inputs=(input,), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    return macs, params
