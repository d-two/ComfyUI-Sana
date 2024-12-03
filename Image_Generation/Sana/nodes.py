import os
import torch
import folder_paths
import numpy as np
from comfy.model_management import get_torch_device, soft_empty_cache, unet_offload_device, text_encoder_offload_device, vae_offload_device
from comfy.utils import load_torch_file
from PIL import Image
import comfy.model_management as mm
from .pipeline.nodes_model_config import get_vram
# from copy import deepcopy

device = get_torch_device()
is_lowvram = get_vram(device)[1] < 8888
current_dir = os.path.dirname(os.path.abspath(__file__))

class UL_SanaVAEProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "vae": ("Sana_VAE", )
            },
            "optional": {
                "latent": ("LATENT", ),
                "image": ("IMAGE", ),
            }
        }
    RETURN_TYPES = ("LATENT", "IMAGE", )
    RETURN_NAMES = ("latent", "image", )
    FUNCTION = "process"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana VAE Process"
    OUTPUT_TOOLTIPS = ("Sana VAE Outputs.", )
    DESCRIPTION = "WIP."

    def process(self, vae, latent=None, image=None):
        vae, dtype = vae['vae'], vae['dtype']
        vae.to(device)
        if latent != None and image != None:
            raise ValueError('......')
        elif latent != None: # decode
            from diffusers.image_processor import PixArtImageProcessor
            vae_scale_factor = 2 ** (len(vae.cfg.encoder.width_list) - 1)
            image_processor = PixArtImageProcessor(vae_scale_factor=vae_scale_factor)
            if 'width' in str(latent.keys()):
                width, height = latent['width'], latent['height']
            else:
                width = None
            latent = latent['samples'].to(device, dtype)
            with torch.inference_mode():
                result = vae.decode(latent.detach() / vae.cfg.scaling_factor)
            if width != None:
                result = image_processor.resize_and_crop_tensor(result, width, height)
            result = image_processor.postprocess(result.cpu())
            results = []
            for img in result:
                results.append(pil2tensor(img))
            result = torch.cat(results, dim=0)
        elif image != None: # encode
            import torchvision.transforms as transforms
            # from .diffusion.model.dc_ae.efficientvit.apps.utils.image import DMCrop
            image_np = image.squeeze().mul(255).clamp(0, 255).byte().numpy()
            image = Image.fromarray(image_np, mode='RGB')
            
            transform = transforms.Compose([
                # DMCrop(1024), # resolution
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            x = transform(image)[None].to(device, dtype)
            # image = image.permute(0, 3, 1, 2)
            with torch.inference_mode():
                # latent = vae.encode(image.to(device, dtype))
                latent = vae.encode(x)
                latent = latent * vae.cfg.scaling_factor
            result = image
        vae.to(vae_offload_device())
        soft_empty_cache(True)
        
        return ( {"samples": latent}, result, )

class UL_SanaSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
                {
                "model": ("Sana_Model", ),
                "sana_conds": ("Sana_Conditionings", ),
                "latent": ("LATENT", ),
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5, "min": 0.00, "max": 99.00, "step": 0.01}),
                "pag": ("FLOAT", {"default": 2.0, "min": 0.00, "max": 99.00, "step": 0.01}),
                "scheduler": (['flow_dpm-solver'], {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "output_type": ("BOOLEAN", {"default": False, "label_on": "latent", "label_off": "image"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量尝试释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
                },
            }

    RETURN_TYPES = ("LATENT", "IMAGE", )
    RETURN_NAMES = ("latent", "image", )
    FUNCTION = "sampler"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana Sampler"
    OUTPUT_TOOLTIPS = ("Sana Samples.", )
    DESCRIPTION = "⚡️Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer\nWe introduce Sana, a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution.\nSana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU."

    def sampler(self, model, sana_conds, steps, cfg, pag, seed, keep_model_loaded, keep_model_device, scheduler, output_type=False, latent=None):
        results = model['pipe'](
            conds=sana_conds,
            guidance_scale=cfg,
            pag_guidance_scale=pag,
            num_inference_steps=(steps+1),
            generator=torch.Generator(device=device).manual_seed(seed),
            latents=None if latent == None else latent['samples'],
            noise_scheduler=scheduler,
            output_type=output_type,
        )
        if keep_model_loaded and keep_model_device:
            model['unet'].to(unet_offload_device())
            soft_empty_cache(True)
        elif not keep_model_loaded:
            del model['unet']
            del model['text_encoder_model']
            del model['vae']
            del model['tokenizer']
            del model['text_encoder']
            del model['pipe'].model
            del model['pipe'].vae
            del model['pipe']
            del model
            soft_empty_cache(True)
            
        if not output_type:
            pil_results = []
            for img in results:
                pil_results.append(pil2tensor(img))
            pil_results = torch.cat(pil_results, dim=0)
            
            empty_latent = torch.zeros([1, 32, 768 // 32, 768 // 32], device=device) # 创建empty latent供调试。
            results = {"samples": empty_latent, "width": 768, "height": 768}
        else:
            import random
            color_list = random.sample(range(0,255),4)
            pil_results = pil2tensor(Image.new(mode='RGBA', size=[768, 768], color=(color_list[0], color_list[1], color_list[2], color_list[3])))
        
        return (results, pil_results, )
        
        
class UL_SanaModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "vae_name": (vaes, ),
                "clip_type": (["gemma-2-2b-it", "gemma-2-2b-it-bnb-4bit","Qwen2-1.5B-Instruct","T5-xxl"],{"default":"gemma-2-2b-it"}),
                "weight_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                "clip_init_device": ("BOOLEAN", {"default": True, "label_on": "device", "label_off": "cpu", "tooltip": "For ram <= 16gb and with cuda device, device is recommended for decrease ram consumption."}),
                "clip_quantize": (["None", "8-bit", "4-bit"], {"tooltip": "For non quantized llm model."}),
            },
        }

    RETURN_TYPES = ("Sana_Model", "Sana_Clip", "Sana_VAE",)
    RETURN_NAMES = ("model", "clip", "vae",)
    FUNCTION = "loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana Model Loader"
    OUTPUT_TOOLTIPS = ("Sana Models.", )
    DESCRIPTION = "If 16gb ram, it needs lot of time to init models."
    
    def loader(self, unet_name, vae_name, clip_type, weight_dtype, clip_init_device, clip_quantize):
        from .diffusion.model.builder import build_model
        from .pipeline.sana_pipeline import SanaPipeline
        from huggingface_hub import snapshot_download
        from .diffusion.model.dc_ae.efficientvit.ae_model_zoo import create_dc_ae_model_cfg
        from .diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import DCAE
        import pyrallis
        from .diffusion.utils.config import SanaConfig
        
        dtype = get_dtype_by_name(weight_dtype)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        # unet_path = r'C:\Users\pc\Desktop\New_Folder\SANA\Sana_1600M_1024px.pth'
        # unet_path = r'C:\Users\pc\Desktop\New_Folder\SANA\Sana_1600M_1024px_MultiLing.pth'
        # unet_path = r'C:\Users\pc\Desktop\New_Folder\SANA\Sana_600M_1024px_MultiLing.pth'
        
        if clip_type == 'gemma-2-2b-it':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--unsloth--gemma-2-2b-it')
            # text_encoder_dir = r'C:\Users\pc\Desktop\New_Folder\SANA\models--unsloth--gemma-2-2b-it'
            if not os.path.exists(os.path.join(text_encoder_dir, 'model.safetensors')):
                snapshot_download('unsloth/gemma-2-2b-it', local_dir=text_encoder_dir)
        elif clip_type == 'gemma-2-2b-it-bnb-4bit':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--unsloth--gemma-2-2b-it-bnb-4bit')
            # text_encoder_dir = r'C:\Users\pc\Desktop\New_Folder\SANA\models--unsloth--gemma-2-2b-it-bnb-4bit'
            if not os.path.exists(os.path.join(text_encoder_dir, 'model.safetensors')):
                snapshot_download('unsloth/gemma-2-2b-it-bnb-4bit', local_dir=text_encoder_dir)
        else:
            raise ValueError('Not implemented!')
        
        # vae = DCAE_HF.from_pretrained(vae_dir).to(dtype).eval()
        
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        # vae_path = r'C:\Users\pc\Desktop\New_Folder\SANA\models--mit-han-lab--dc-ae-f32c32-sana-1.0\model.safetensors'
        cfg = create_dc_ae_model_cfg('dc-ae-f32c32-sana-1.0')
        vae = DCAE(cfg)
        state_dict = load_torch_file(vae_path, safe_load=True)
        vae.load_state_dict(state_dict, strict=False)
        state_dict = None
        vae.to(dtype).eval()
        
        if "T5" in clip_type:
            from transformers import T5Tokenizer, T5EncoderModel
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_dir)
            llm_model = None
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_dir, torch_dtype=dtype)
        else:
            from transformers import (
                AutoTokenizer, 
                AutoModelForCausalLM, 
                BitsAndBytesConfig, 
                # Gemma2Config,
                # GemmaTokenizerFast,
                # Gemma2ForCausalLM,
                )
            # import json
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
            
            quantization_config = BitsAndBytesConfig(load_in_8bit=True) if clip_quantize=='8-bit' else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype) if clip_quantize=='4-bit' else None
            
            llm_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, quantization_config=quantization_config, torch_dtype=dtype) if clip_type != 'gemma-2-2b-it-bnb-4bit' else AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype=dtype)
            
            # llm_model_path = os.path.join(text_encoder_dir, 'model.safetensors')
            # config_path = os.path.join(text_encoder_dir, 'config.json')
            # with open(config_path, 'r') as file:
            #     config = json.load(file)
            # tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
            # llm_model = Gemma2ForCausalLM(**config) if 'bit' not in clip_type else Gemma2ForCausalLM(**config, quantization_config=quantization_config)
            # state_dict = load_torch_file(llm_model_path, safe_load=True)
            # llm_model.load_state_dict(state_dict)
            # state_dict = None
            # llm_model.to(dtype)
            
            tokenizer.padding_side = "right"
            text_encoder = llm_model.get_decoder()
        
        if clip_init_device:
            try:
                text_encoder.to(device)
            except torch.cuda.OutOfMemoryError as e:
                raise e
        
        
        state_dict = load_torch_file(unet_path, safe_load=True)
        is_1600M = state_dict['final_layer.scale_shift_table'].shape[1]==2240 # 1.6b: 2240 0.6b: 1152
        
        config_path = os.path.join(current_dir, 'configs', 'sana_config', '1024ms', 'Sana_1600M_img1024_AdamW.yaml') if is_1600M else os.path.join(current_dir, 'configs', 'sana_config', '1024ms', 'Sana_600M_img1024.yaml')
        
        config = pyrallis.load(SanaConfig, open(config_path))
        
        pred_sigma = getattr(config.scheduler, "pred_sigma", True)
        learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
        image_size = config.model.image_size
        latent_size = image_size // config.vae.vae_downsample_rate
        model_kwargs = {
            "input_size": latent_size,
            "pe_interpolation": config.model.pe_interpolation,
            "config": config,
            "model_max_length": config.text_encoder.model_max_length,
            "qk_norm": config.model.qk_norm,
            "micro_condition": config.model.micro_condition,
            "caption_channels": text_encoder.config.hidden_size, # Gemma2: 2304
            "y_norm": config.text_encoder.y_norm,
            "attn_type": config.model.attn_type,
            "ffn_type": config.model.ffn_type,
            "mlp_ratio": config.model.mlp_ratio,
            "mlp_acts": list(config.model.mlp_acts),
            "in_channels": config.vae.vae_latent_dim,
            "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
            "use_pe": config.model.use_pe,
            "pred_sigma": pred_sigma,
            "learn_sigma": learn_sigma,
            "use_fp32_attention": config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
        }
        
        unet = build_model(config.model.model, **model_kwargs)
        unet.to(dtype)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        state_dict = None
        unet.eval().to(dtype)
        pipe = SanaPipeline(config, vae, dtype, unet)
        
        clip = {
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'text_encoder_model': llm_model,
        }
        
        model = {
            'pipe': pipe,
            'unet': unet,
            'text_encoder_model': llm_model,
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'vae': vae,
        }
        
        out_vae = {
            'vae': vae,
            'dtype': dtype,
        }
        
        return (model, clip, out_vae, )

class UL_SanaTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sana_clip": ("Sana_Clip", {"tooltip": "The CLIP model used for encoding the text."}),
                "text": ("STRING", {"default": "A wide shot of (cat) wearing (jacket) with boston city in background, masterpiece, best quality, high quality, 4K, highly detailed, extremely detailed, HD, ", "multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "n_text": ("STRING", {"default": "watermark, author name, monochrome, lowres, bad anatomy, worst quality, low quality, username.", "multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "preset_styles": (STYLE_NAMES, ),
            },
        }
    RETURN_TYPES = ("Sana_Conditionings", "STRING", "STRING", )
    RETURN_NAMES = ("sana_conds", "prompt", "n_prompt", )
    FUNCTION = "encode"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana Text Encoder"
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, text, n_text, preset_styles, sana_clip=None):
        from .diffusion.model.utils import prepare_prompt_ar
        from .diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
        base_ratios = eval(f"ASPECT_RATIO_{1024}_TEST")
        tokenizer = sana_clip['tokenizer']
        text_encoder = sana_clip['text_encoder']
        
        text, n_text = apply_style(preset_styles, text, n_text)
        
        try:
            text_encoder.to(device)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        
        null_caption_token = tokenizer(
            n_text,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]
        
        prompts = []
        with torch.no_grad():
        # with torch.inference_mode():
            prompts.append(prepare_prompt_ar(text, base_ratios, device=device, show=False)[0].strip())
            chi_prompt = "\n".join(preset_te_prompt)
            prompts_all = [chi_prompt + text]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = (num_chi_prompt_tokens + 300 - 2)  # magic number 2: [bos], [_]
            caption_token = tokenizer(
                prompts_all,
                max_length=max_length_all,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            select_index = [0] + list(range(-300 + 1, 0))
            caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][:, :, select_index]
            emb_masks = caption_token.attention_mask[:, select_index]
            null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        
        text_encoder.to(text_encoder_offload_device())
        soft_empty_cache(True)
        
        return ([caption_embs, null_y, emb_masks], text, n_text, )
        
class UL_SanaVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "vae_name": (vaes, ),
                "weight_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
            },
        }

    RETURN_TYPES = ("Sana_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana VAE Loader(TestOnly)"
    OUTPUT_TOOLTIPS = ("Sana VAE: DCAE.", )
    DESCRIPTION = "For test only."
    
    def loader(self, vae_name, weight_dtype):
        from .diffusion.model.dc_ae.efficientvit.ae_model_zoo import create_dc_ae_model_cfg
        from .diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import DCAE
        
        dtype = get_dtype_by_name(weight_dtype)
        
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        # vae_path = r'C:\Users\pc\Desktop\New_Folder\SANA\models--mit-han-lab--dc-ae-f32c32-sana-1.0\model.safetensors'
        cfg = create_dc_ae_model_cfg('dc-ae-f32c32-sana-1.0')
        vae = DCAE(cfg)
        state_dict = load_torch_file(vae_path, safe_load=True)
        vae.load_state_dict(state_dict, strict=False)
        state_dict = None
        vae.to(dtype).eval()
        
        out_vae = {
            'vae': vae,
            'dtype': dtype,
        }
        
        return (out_vae, )
        
NODE_CLASS_MAPPINGS = {
    "UL_SanaSampler": UL_SanaSampler,
    "UL_SanaModelLoader": UL_SanaModelLoader,
    "UL_SanaTextEncode": UL_SanaTextEncode,
    "UL_SanaVAEProcess": UL_SanaVAEProcess,
    "UL_SanaVAELoader": UL_SanaVAELoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sana Sampler": "UL_SanaSampler",
    "Sana Model Loader": "UL_SanaModelLoader",
    "Sana Text Encoder": "UL_SanaTextEncode",
    "Sana VAE Process": "UL_SanaVAEProcess",
    "Sana VAE Loader(TestOnly)": "UL_SanaVAELoader",
}

preset_te_prompt = ['Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:', '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.', '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.', 'Here are examples of how to transform or refine prompts:', '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.', '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.', 'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:', 'User Prompt: ']

def get_dtype_by_name(dtype, debug: bool=False):
    """
    "dtype": (["auto","fp16","bf16","fp32", "fp8_e4m3fn", "fp8_e4m3fnuz", "fp8_e5m2", "fp8_e5m2fnuz"],{"default":"auto"}),返回模型精度选择。
    """
    if dtype == 'auto':
        try:
            if mm.should_use_fp16():
                dtype = torch.float16
            elif mm.should_use_bf16():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
    elif dtype== "fp16":
         dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif dtype == "fp8_e4m3fnuz":
        dtype = torch.float8_e4m3fnuz
    elif dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif dtype == "fp8_e5m2fnuz":
        dtype = torch.float8_e5m2fnuz
    if debug:
        print("\033[93mModel Precision(模型精度):", dtype, "\033[0m")
    return dtype

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[style_name])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative