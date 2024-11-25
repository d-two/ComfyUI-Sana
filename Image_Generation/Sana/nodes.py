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
            'optional': {
                'latent': ('LATENT', ),
                'image': ('IMAGE', ),
            }
        }
    RETURN_TYPES = ("LATENT", 'IMAGE', )
    RETURN_NAMES = ('latent', "image", )
    FUNCTION = "process"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana VAE Process"
    OUTPUT_TOOLTIPS = 'Sana VAE Outputs.'
    DESCRIPTION = "WIP."

    def process(self, vae, latent=None, image=None):
        vae.to(device)
        if latent != None and image != None:
            raise ValueError('......')
        elif latent != None: # decode
            from diffusers.image_processor import PixArtImageProcessor
            vae_scale_factor = 2 ** (len(vae.cfg.encoder.width_list) - 1)
            image_processor = PixArtImageProcessor(vae_scale_factor=vae_scale_factor)
            vae.to(latent['samples'].dtype)
            with torch.inference_mode():
                result = vae.decode(latent['samples'].detach() / vae.cfg.scaling_factor)
            result = image_processor.postprocess(result.cpu())
            results = []
            for img in result:
                results.append(pil2tensor(img))
            result = torch.cat(results, dim=0)
        elif image != None:
            with torch.inference_mode():
                latent = vae.encode(image)
                latent = latent.latent_dist.sample()
        vae.to(vae_offload_device())
        soft_empty_cache(True)
        
        return ( result, {"samples": latent}, )

class UL_SanaSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
                {
                "model": ("Sana_Model", ),
                'sana_conds': ('Sana_Conditionings', ),
                "seed": ("INT", {"default": 88888888, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 18, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5, "min": 0.00, "max": 99.00, "step": 0.01}),
                "pag": ("FLOAT", {"default": 2.0, "min": 0.00, "max": 99.00, "step": 0.01}),
                "width": ("INT", {"default": 1024,"min": 256, "max": 10240, "step": 1}),
                "height": ("INT", {"default": 1024,"min": 256, "max": 10240, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 960}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no", "tooltip": "Warning: do not delete model unless this node no longer needed, it will try release device_memory and ram. if checked and want to continue node generation, use ComfyUI-Manager `Free model and node cache` to reset node state or change parameter in Loader node to activate.\n注意：仅在这个节点不再需要时删除模型，将尽量尝试释放系统内存和设备专用内存。如果删除后想继续使用此节点，使用ComfyUI-Manager插件的`Free model and node cache`重置节点状态或者更换模型加载节点的参数来激活。"}),
                "keep_model_device": ("BOOLEAN", {"default": True, "label_on": "comfy", "label_off": "device", "tooltip": "Keep model in comfy_auto_unet_offload_device (HIGH_VRAM: device, Others: cpu) or device_memory after generation.\n生图完成后，模型转移到comfy自动选择设备(HIGH_VRAM: device, 其他: cpu)或者保留在设备专用内存上。"}),
                # "output_type": ("BOOLEAN", {"default": True, "label_on": "latent", "label_off": "image"}),
                },
                'optional': {
                    'latent': ('LATENT', ),
                }
            }

    RETURN_TYPES = ('LATENT', "IMAGE", )
    RETURN_NAMES = ('latent', "image", )
    FUNCTION = "sampler"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana Sampler"
    OUTPUT_TOOLTIPS = 'Sana Samples.'
    DESCRIPTION = "⚡️Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer\nWe introduce Sana, a text-to-image framework that can efficiently generate images up to 4096 × 4096 resolution.\nSana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU."

    def sampler(self, model, sana_conds, steps, cfg, pag, seed, keep_model_loaded, batch_size, keep_model_device, width, height, output_type=False, latent=None):
        results = model['pipe'](
            conds=sana_conds,
            height=height,
            width=width,
            guidance_scale=cfg,
            pag_guidance_scale=pag,
            num_inference_steps=(steps+1),
            num_images_per_prompt=batch_size,
            generator=torch.Generator(device=device).manual_seed(seed),
            latents=None if latent == None else latent['samples'],
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

            empty_latent = torch.zeros([1, 4, height // 8, width // 8], device=device) # 创建empty latent供调试。
            results = {"samples": empty_latent}
        else:
            import random
            color_list = random.sample(range(0,255),4)
            pil_results = pil2tensor(Image.new(mode='RGBA', size=[768, 768], color=(color_list[0], color_list[1], color_list[2], color_list[3])))
        
        return (results, pil_results, )
        
        
class UL_SanaModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "clip_type": (["gemma-2-2b-it","Qwen2-1.5B-Instruct","T5-xxl"],{"default":"gemma-2-2b-it"}),
                "weight_dtype": (["auto","fp16","bf16","fp32"],{"default":"auto"}),
                "clip_init_device": ("BOOLEAN", {"default": True, "label_on": "device", "label_off": "cpu", 'tooltip': 'For ram <= 16gb and with cuda device, device is recommended for decrease ram consumption.'}),
            },
        }

    RETURN_TYPES = ("Sana_Model", 'Sana_Clip', 'Sana_VAE',)
    RETURN_NAMES = ("model", 'clip', 'vae',)
    FUNCTION = "loader"
    CATEGORY = "UL Group/Image Generation"
    TITLE = "Sana Model Loader"
    OUTPUT_TOOLTIPS = 'Sana Model Loader.'
    DESCRIPTION = "If 16gb ram, it needs lot of time to init models."
    
    def loader(self, unet_name, clip_type, weight_dtype, clip_init_device):
        from .diffusion.model.builder import build_model
        from .pipeline.sana_pipeline import SanaPipeline
        from huggingface_hub import snapshot_download
        from .diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
        import pyrallis
        from .diffusion.utils.config import SanaConfig
        
        dtype = get_dtype_by_name(weight_dtype)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        vae_dir = os.path.join(folder_paths.models_dir, 'vae', 'models--mit-han-lab--dc-ae-f32c32-sana-1.0')
        # vae_dir = r'C:\Users\pc\Desktop\New_Folder\SANA\models--mit-han-lab--dc-ae-f32c32-sana-1.0'
        if not os.path.exists(os.path.join(vae_dir, 'model.safetensors')):
            snapshot_download('mit-han-lab/dc-ae-f32c32-sana-1.0', local_dir=vae_dir)
        
        if clip_type == 'gemma-2-2b-it':
            text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', 'models--google--gemma-2-2b-it')
            # text_encoder_dir = r'C:\Users\pc\Desktop\New_Folder\SANA\models--google--gemma-2-2b-it'
            if not os.path.exists(os.path.join(text_encoder_dir, 'model-00001-of-00002.safetensors')):
                snapshot_download('google/gemma-2-2b-it', local_dir=text_encoder_dir)
        else:
            raise ValueError('Not implemented!')
        
        vae = DCAE_HF.from_pretrained(vae_dir).to(dtype).eval()
        
        if "T5" in clip_type:
            from transformers import T5Tokenizer, T5EncoderModel
            tokenizer = T5Tokenizer.from_pretrained(text_encoder_dir)
            text_encoder_model = None
            text_encoder = T5EncoderModel.from_pretrained(text_encoder_dir, torch_dtype=dtype)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM#, Gemma2ForCausalLM, Gemma2Config
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
            # config = Gemma2Config.from_json_file()
            # text_encoder_model = Gemma2ForCausalLM(**config)
            # state_dict = load_torch_file(text_encoder_path)
            # text_encoder_model.load_state_dict()
            tokenizer.padding_side = "right"
            text_encoder_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype=dtype)
            text_encoder = text_encoder_model.get_decoder()
        
        if clip_init_device:
            try:
                text_encoder.to(device)
            except torch.cuda.OutOfMemoryError as e:
                raise e
        
        config_path = os.path.join(current_dir, 'configs', 'sana_config', '1024ms', 'Sana_1600M_img1024.yaml')
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
        state_dict = load_torch_file(unet_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        del state_dict
        unet.eval().to(dtype)
        pipe = SanaPipeline(config, vae, dtype, unet)
        
        clip = {
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
        }
        
        model = {
            'pipe': pipe,
            'unet': unet,
            'text_encoder_model': text_encoder_model,
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'vae': vae,
        }
        
        return (model, clip, vae, )

class UL_SanaTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sana_clip": ("Sana_Clip", {"tooltip": "The CLIP model used for encoding the text."}),
                "text": ("STRING", {"default": "A wide shot of (cat) wearing (jacket) with boston city in background, masterpiece, best quality, high quality, 4K, highly detailed, extremely detailed, HD, ", "multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "n_text": ("STRING", {"default": "watermark, author name, monochrome, lowres, bad anatomy, worst quality, low quality, username.", "multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
            },
        }
    RETURN_TYPES = ("Sana_Conditionings", "STRING", "STRING", )
    RETURN_NAMES = ("sana_conds", "prompt", "n_prompt", )
    FUNCTION = "encode"
    CATEGORY = "UL Group/Diffusers Common"
    TITLE = "Sana Text Encoder"
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, text, n_text, sana_clip=None):
        from .diffusion.model.utils import prepare_prompt_ar
        from .diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
        base_ratios = eval(f"ASPECT_RATIO_{1024}_TEST")
        tokenizer = sana_clip['tokenizer']
        text_encoder = sana_clip['text_encoder']
        
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
        with torch.inference_mode():
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
        
        
NODE_CLASS_MAPPINGS = {
    'UL_SanaSampler': UL_SanaSampler,
    'UL_SanaModelLoader': UL_SanaModelLoader,
    'UL_SanaTextEncode': UL_SanaTextEncode,
    'UL_SanaVAEProcess': UL_SanaVAEProcess,
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