# DCAEConfig(in_channels=3, latent_channels=32, encoder=EncoderConfig(in_channels=3, latent_channels=32, width_list=[128, 256, 512, 512, 1024, 1024], depth_list=[2, 2, 2, 3, 3, 3], block_type=['ResBlock', 'ResBlock', 'ResBlock', 'EViTS5_GLU', 'EViTS5_GLU', 'EViTS5_GLU'], norm='trms2d', act='silu', downsample_block_type='Conv', downsample_match_channel=True, downsample_shortcut='averaging', out_norm=None, out_act=None, out_shortcut='averaging', double_latent=False), decoder=DecoderConfig(in_channels=3, latent_channels=32, in_shortcut='duplicating', width_list=[128, 256, 512, 512, 1024, 1024], depth_list=[3, 3, 3, 3, 3, 3], block_type=['ResBlock', 'ResBlock', 'ResBlock', 'EViTS5_GLU', 'EViTS5_GLU', 'EViTS5_GLU'], norm='trms2d', act='silu', upsample_block_type='InterpolateConv', upsample_match_channel=True, upsample_shortcut='duplicating', out_norm='trms2d', out_act='relu'), use_quant_conv=False, pretrained_path=None, pretrained_source='dc-ae', scaling_factor=0.41407)

import torch

def get_vram(device, debug=False):
    """_summary_

    Args:
        function: get_vram[0] or get_vram[1]
        model or file dtype: 
            model.eval().half() # 启用半精度计算
            model.eval().to(dtype) # 启用半精度计算
            model.eval().cuda() # 模型转移到cuda
            model.eval().half().cuda() # 模型启用半精度计算再转移到cuda
            model.eval().half().to(device) # 模型启用半精度计算再转移到指定设备
            model.eval().to(device, dtype) # 模型启用指定精度计算再转移到指定设备
            image_tensor.to(device, dtype)
            mask_tensor.to(device, dtype)
            token_tensor.to(device, dtype)
            prompt_tensor.to(device, dtype)

    Returns:
        (total_memory, avialable_memory, ) int
    """
    if torch.cuda.is_available():
        if device == 'cuda' or 'cuda' in device.type:
            gpu_count = torch.cuda.device_count() # 获取可用GPU的数量
            available_gpus = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_memory = props.total_memory - torch.cuda.memory_reserved(i)
                available_gpus.append((i, free_memory))
            selected_gpu, max_free_memory = max(available_gpus, key=lambda x: x[1])
            device = torch.device(f'cuda:{selected_gpu}')
            total_memory = props.total_memory // 1024**2 # 转换为MB
            avialable_memory = max_free_memory / (1024 * 1024) # 转换为MB
            if debug:
                print('\033[93m', f'GPU数量：{gpu_count}', '\033[0m')
                print('\033[93m', f'可用GPU：{available_gpus}', '\033[0m')
                print('\033[93m', f'当前GPU：{selected_gpu}', '\033[0m')
                print('\033[93m', f'GPU信息：{props}', '\033[0m')
                print('\033[93m', f'GPU总显存VRAM：{total_memory}MB', '\033[0m')
                print('\033[93m', f'GPU已用显存VRAM：{(total_memory + 0.5 - avialable_memory)}MB', '\033[0m')
                print('\033[93m', f'GPU可用显存VRAM：{avialable_memory}MB', '\033[0m')
                print('\033[93m', f'可根据cuda设备显存大小和模型需求，选择模型初始化init_device设备并启用半精度。然后再转移到cuda设备，节省显存。', '\033[0m')
        else:
            total_memory, avialable_memory = get_ram()[0], get_ram[1]
    return (total_memory, avialable_memory, )

def get_ram(debug=False):
    """获取系统的总物理内存
        get_ram()[0] or get_ram()[1]
    Returns:
        (total_memory, available_memory, ) float
    """
    import psutil
    memory_info = psutil.virtual_memory()
    total_memory = '{:.2f}'.format(memory_info.total / (1024 ** 3))
    used_memory = '{:.2f}'.format(memory_info.used / (1024 ** 3))
    available_memory = '{:.2f}'.format(memory_info.available / (1024 ** 3))
    if debug:
        print('\033[93m', f'总内存：{total_memory}GB', '\033[0m')
        print('\033[93m', f'已用内存：{used_memory}GB', '\033[0m')
        print('\033[93m', f'可用内存：{available_memory}GB', '\033[0m')
    return (total_memory, available_memory, )