import torch
import os
import folder_paths
from safetensors.torch import save_file, load_file
from datetime import datetime
from server import PromptServer
import json
import numpy as np
from PIL import Image, PngImagePlugin
import hashlib
import re
import shutil
from comfy.cli_args import args

class SaveWanVideoLatent:
    """
    Saves Wan video latents to output directory with support for subfolders.
    Saves the complete latent dictionary including samples and any metadata.
    Non an output node, may require assist of other output node. Properly works in loops.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Wan video latent tensor"}),
                "filename": ("STRING", {"default": "wan_latent.safetensors", "tooltip": "Filename without extension"}),
                "enabled": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "subdirectory": ("STRING", {"default": "", "tooltip": "Subfolder in output/ (created if needed)"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subdirectory",)
    FUNCTION = "save_latent"
    OUTPUT_NODE = False
    CATEGORY = "⛧Heresy Nodes/latent"
    DESCRIPTION = "Save Wan video latents to output folder with optional subdirectories"

    def save_latent(self, latent, filename, enabled=True, subdirectory=""):

        if not enabled:
            return (subdirectory,)
        
        # Build save path
        save_dir = self.output_dir
        if subdirectory:
            # Security: prevent directory traversal
            subdirectory = os.path.normpath(subdirectory).replace("..", "_")
            save_dir = os.path.join(save_dir, subdirectory)
            os.makedirs(save_dir, exist_ok=True)
        
        # Handle filename conflicts
        base_path = os.path.join(save_dir, filename)
        full_path = f"{base_path}"
        
        counter = 1
        while os.path.exists(full_path):
            full_path = f"{base_path}_{counter:03d}.safetensors"
            counter += 1
        
        try:
            # Convert latent dict to safetensors format
            # Safetensors only stores tensors, so we store non-tensor metadata as strings
            tensors = {}
            metadata = {}
            
            for key, value in latent.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value
                else:
                    metadata[key] = str(value)
            
            save_file(tensors, full_path, metadata=metadata)
            
            print(f"[WanVideo LatentIO] Saved to: {full_path}")
            return (subdirectory,)
            
        except Exception as e:
            print(f"[WanVideo LatentIO] Error saving: {e}")
            raise

class LoadWanVideoLatent:
    """
    Loads Wan video latents from either input or output directories.
    Supports subdirectories for organized latent storage.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "", "tooltip": "Filename with extension (e.g., latent.safetensors)"}),
                "source_directory": (["input", "output"], {"default": "input", "tooltip": "Load from ComfyUI/input or ComfyUI/output"}),
            },
            "optional": {
                "subdirectory": ("STRING", {"default": "", "tooltip": "Subfolder within selected directory"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_latent"
    CATEGORY = "⛧Heresy Nodes/latent"
    DESCRIPTION = "Load Wan video latents from input or output folders"

    def load_latent(self, filename, source_directory="output", subdirectory=""):
        # Determine base directory
        if source_directory == "input":
            base_dir = folder_paths.get_input_directory()
        else:
            base_dir = folder_paths.get_output_directory()
        
        # Build full path
        if subdirectory:
            subdirectory = os.path.normpath(subdirectory).replace("..", "_")
            base_dir = os.path.join(base_dir, subdirectory)
        
        full_path = os.path.join(base_dir, filename)
        
        # Try to find file with auto-extension
        if not os.path.exists(full_path):
            for ext in [".safetensors", ".pt", ".pth"]:
                test_path = full_path + ext
                if os.path.exists(test_path):
                    full_path = test_path
                    break
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"[WanVideo LatentIO] File not found: {full_path}")
        
        try:
            if full_path.endswith(".safetensors"):
                # Load safetensors
                tensors = load_file(full_path)
                # Try to reconstruct metadata if present
                # Note: safetensors metadata loading requires additional handling
                latent = dict(tensors)
                
                # Ensure samples key exists
                if "samples" not in latent and len(latent) > 0:
                    # If only one tensor, assume it's samples
                    if len(latent) == 1:
                        latent = {"samples": list(latent.values())[0]}
            else:
                # Load PyTorch file
                latent = torch.load(full_path, map_location="cpu")
                if not isinstance(latent, dict):
                    latent = {"samples": latent}
            
            print(f"[WanVideo LatentIO] Loaded from: {full_path}")
            return (latent,)
            
        except Exception as e:
            print(f"[WanVideo LatentIO] Error loading: {e}")
            raise

class GetDateTimeStringUpd:
    time_format = ["%Y%m%d%H%M%S","%Y%m%d%H%M","%Y%m%d","%Y-%m-%d-%H_%M_%S","%Y-%m-%d-%H_%M","%Y-%m-%d","%Y-%m-%d %H_%M_%S","%Y-%m-%d %H_%M","%Y-%m-%d","%H%M","%H%M%S","%H_%M","%H_%M_%S"]
    timestamp = ''
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "update": ("BOOLEAN", {"default": "true", "tooltip": "Update every run"}),
                "style": (s.time_format,),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("time_format",)
    FUNCTION = "get_time"

    CATEGORY = "⛧Heresy Nodes/utils"

    def get_time(self, update, style):
        now = datetime.now()
        self.timestamp = now.strftime(style)

        return (self.timestamp,)

    @classmethod
    def IS_CHANGED(self, update, style):
        if update == True:
            now = datetime.now()
            self.timestamp = now.strftime(style)
        return (self.timestamp,)
    
class SaveImageBatchNoMeta:
    """
    Saves image batches as PNG without workflow metadata by default.
    NOT an output node - works in loops.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "include_workflow": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}),
                "enabled": ("BOOLEAN", {"default": True })
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    # NOT an output node - critical for loop compatibility
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "save_images"
    CATEGORY = "⛧Heresy Nodes/image"
    OUTPUT_NODE = False
    
    def save_images(self, images, filename_prefix="ComfyUI", include_workflow=False, enabled=True, prompt=None, extra_pnginfo=None):

        if not enabled:
            return (filename_prefix,)

        # Get output path
        full_output_folder, filename, counter, subfolder, filename_prefix_out = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        saved_count = 0
        
        # Process each image in batch
        for batch_number, image in enumerate(images):
            # Convert tensor to numpy
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Generate filename
            file = f"{filename}_{counter:05}_.png"
            file_path = os.path.join(full_output_folder, file)
            
            # Prepare metadata
            metadata = None
            if include_workflow and not args.disable_metadata:
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            
            # Save PNG
            if metadata is not None:
                img.save(file_path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                img.save(file_path, compress_level=self.compress_level)
            
            saved_count += 1
            counter += 1
        
        print(f"[SaveImageBatchNoMeta] Saved {saved_count} image(s) to {full_output_folder}")
        
        # Return filename_prefix for chaining
        return (filename_prefix,)
    
class RaiseExceptionMessage:
    """
    A node that throws an exception with custom text when triggered.
    Passes through any data type by reference when not triggered.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*",),  # First input - any type, passed through by reference
                "trigger": ("BOOLEAN", {"default": False}),
                "message": ("STRING", {"default": "Exception triggered!", "multiline": True}),
            },
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "throw_exception"
    CATEGORY = "⛧Heresy Nodes/utils"
    
    def throw_exception(self, any, trigger, message):
        if trigger:
            raise Exception(f"[RaiseExceptionMessage] {message}")
        
        # Return data by reference (same object, no copy)
        return (any,)
    
class LoadSingleImage:
    """
    Load a single image by path and filename with boolean status output
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "Enter directory path (e.g., /path/to/folder or C:\\Users\\Name\\Pictures)"
                }),
                "filename": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "Enter filename (e.g., image.png)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "is_loaded")
    FUNCTION = "load_image"
    CATEGORY = "⛧Heresy Nodes/image"
    DESCRIPTION = "Load a single image by directory path and filename. Returns is_loaded=true if successful, false if file not found or error."

    def load_image(self, path, filename):
        # Default empty return
        empty_image = torch.zeros((1, 64, 64, 3))
        
        # Check if inputs are provided
        if not path or path.strip() == "":
            print(f"[LoadSingleImage] Error: No path provided")
            return (empty_image, False)
        
        if not filename or filename.strip() == "":
            print(f"[LoadSingleImage] Error: No filename provided")
            return (empty_image, False)
        
        # Clean up inputs
        path = path.strip().strip('"').strip("'")
        filename = filename.strip().strip('"').strip("'")
        
        # Remove trailing slash/backslash from path for consistency
        path = path.rstrip(os.sep)
        
        # Join path and filename
        image_path = os.path.join(path, filename)
        
        # Normalize path for current OS
        image_path = os.path.normpath(image_path)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"[LoadSingleImage] Error: File not found: {image_path}")
            return (empty_image, False)
        
        # Check if it's a file (not directory)
        if not os.path.isfile(image_path):
            print(f"[LoadSingleImage] Error: Path is not a file: {image_path}")
            return (empty_image, False)
        
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Convert to RGB
            img = img.convert('RGB')
            
            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img_array)[None,]
            
            print(f"[LoadSingleImage] Successfully loaded: {image_path} ({img.size[0]}x{img.size[1]})")
            return (image_tensor, True)
            
        except Exception as e:
            print(f"[LoadSingleImage] Error loading image: {e}")
            return (empty_image, False)
        
class LoadImagesByCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load_images"
    CATEGORY = "⛧Heresy Nodes/image"
    DESCRIPTION = "Loads image batch by condition"

    def load_images(self, folder_path, enabled=True):
        # Fast path: return empty tensor without touching disk
        if not enabled:
            return (torch.zeros((0, 64, 64, 3), dtype=torch.float32), 0)

        if not folder_path:
            raise ValueError("folder_path cannot be empty")

        # Resolve relative paths against ComfyUI input directory
        if not os.path.isabs(folder_path):
            base = folder_paths.get_input_directory()
            folder_path = os.path.join(base, folder_path)

        folder_path = os.path.abspath(os.path.normpath(folder_path))
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a valid directory: {folder_path}")

        # os.scandir is faster than os.listdir + os.path.join + os.path.isfile
        entries = [e for e in os.scandir(folder_path) if e.is_file()]
        if not entries:
            raise ValueError(f"No files found in {folder_path}")

        # Sort by name ascending
        entries.sort(key=lambda e: e.name)

        # Load every file as an image. If a file is corrupt, it is skipped.
        images = []
        for e in entries:
            try:
                img = Image.open(e.path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(np.array(img, dtype=np.float32) / 255.0)
            except Exception:
                continue

        if not images:
            raise ValueError(f"No valid images loaded from {folder_path}")

        batch = torch.from_numpy(np.stack(images, axis=0))
        return (batch, len(images))

        
class DiffModelSelect:
    """
    ComfyUI node that provides a dropdown to select base diffusion models.
    Supports both .safetensors and .gguf file extensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Combine both lists: standard + gguf
        model_list = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet_gguf")
        
        return {
            "required": {
                "model_name": (sorted(set(model_list)), {
                    "tooltip": "Select a diffusion model (.safetensors or .gguf)"
                }),
            },
        }

    RETURN_TYPES = ("STRING","BOOLEAN")
    RETURN_NAMES = ("model_name","is_gguf")
    FUNCTION = "select_model"
    CATEGORY = "⛧Heresy Nodes/utils"
    DESCRIPTION = "Select a base diffusion model from available .safetensors and .gguf files"

    def select_model(self, model_name):
        if model_name.endswith(".gguf"):
            is_gguf = True
        else:
            is_gguf = False
        # Return the selected model path as string
        return (model_name, is_gguf)
    
class CLIPModelSelector:
    """
    ComfyUI node that provides a dropdown to select CLIP models.
    Supports both .safetensors and .gguf file extensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get all CLIP models from relevant folders
        model_paths = folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")
        
        return {
            "required": {
                "model_name": (sorted(set(model_paths)), {
                    "tooltip": "Select a CLIP model (.safetensors or .gguf)"
                }),
            },
        }

    RETURN_TYPES = ("STRING","BOOLEAN")
    RETURN_NAMES = ("model_name","is_gguf")
    FUNCTION = "select_clip_model"
    CATEGORY = "⛧Heresy Nodes/utils"
    DESCRIPTION = "Select a CLIP/text encoder model from available .safetensors and .gguf files"

    def select_clip_model(self, model_name,):
        if model_name.endswith(".gguf"):
            is_gguf = True
        else:
            is_gguf = False
        # Return the selected model path as string
        return (model_name, is_gguf)

class LoadImageFolders:
    """
    Loads images from multiple indexed folders (01-99) and places them into dict with their names as keys
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_path": ("STRING",),
                "subfolder": ("STRING",),
                "folders_str": ("STRING",),
            },
            "optional": {
                "load_from_end": ("INT", {"default": 0, "tooltip": "Number of last sections that will be loaded"}),
                "dict_prefill": ("DICT", {"tooltip": "Existing dictionary of images to append to. Existing keys are skipped."}),
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("DICT",)
    FUNCTION = "load_image_folders"
    CATEGORY = "⛧Heresy Nodes/image"
    DESCRIPTION = "Loads images from multiple indexed folders (01-99) and places them into dict with their names as keys. Skipped folders return None."
    
    def load_image_folders(self, base_path, subfolder, folders_str, load_from_end=0, dict_prefill=None):
        out_dict = dict_prefill if dict_prefill is not None else {}
        img_loader = LoadImagesByCond() 
        
        # 1. Collect all valid numbered folders first
        all_valid_folders = []
        for folder in folders_str.splitlines():
            folder = folder.strip()
            if len(folder) >= 2 and folder[:2].isdigit() and int(folder[:2]) != 0:
                all_valid_folders.append(folder)
                
        # 2. Determine which folders actually need their images loaded
        if load_from_end > 0:
            folders_to_load = all_valid_folders[-load_from_end:]
        else:
            folders_to_load = all_valid_folders
            
        # 3. Process ALL valid folders
        for folder in all_valid_folders:
            key = folder[:2]
            
            # Check if the key already exists to save performance
            if key in out_dict:
                if folder in folders_to_load and out_dict[key] == None:
                    # Folder is in the load_from_end slice; load the image
                    load_path = os.path.join(base_path, folder, subfolder)
                    image, _dummy = img_loader.load_images(load_path)
                    out_dict[key] = image
            else:
                if folder in folders_to_load:
                    # Folder is in the load_from_end slice; load the image
                    load_path = os.path.join(base_path, folder, subfolder)
                    image, _dummy = img_loader.load_images(load_path)
                    out_dict[key] = image
                else:
                    # Folder was skipped by load_from_end; populate key with empty value
                    out_dict[key] = None

        return (out_dict,)
    
class GetSubfolders:
    """
    A node that lists all subfolders in a specified folder
    and returns them as a single string with \n separators.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Absolute path to folder"
                }),
            },
            "optional": {
                "sort_alphabetically": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subfolders",)
    FUNCTION = "list_subfolders"
    CATEGORY = "⛧Heresy Nodes/utils"
    OUTPUT_NODE = False

    def list_subfolders(self, folder_path, sort_alphabetically=True):
        # Expand user home directory (~) and normalize path
        expanded_path = os.path.expanduser(folder_path.strip())
        expanded_path = os.path.normpath(expanded_path)

        # Validate path exists and is a directory
        if not os.path.exists(expanded_path):
            return (f"Error: Path does not exist: {expanded_path}",)

        if not os.path.isdir(expanded_path):
            return (f"Error: Path is not a directory: {expanded_path}",)

        try:
            # Get all subdirectories
            subfolders = [
                entry.name
                for entry in os.scandir(expanded_path)
                if entry.is_dir()
            ]

            # Sort if requested
            if sort_alphabetically == True:
                subfolders.sort(key=str.lower)

            # Join with newline separator
            result = "\n".join(subfolders) if subfolders else ""

            return (result,)

        except PermissionError:
            return (f"Error: Permission denied accessing: {expanded_path}",)
        except Exception as e:
            return (f"Error: {str(e)}",)
        
        
class StitchImgMulti:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_dict": ("DICT", {"tooltip": "The source dictionary of image batches"}),
                "overlap": ("INT", {"default": 13,"min": 1, "max": 4096, "step": 1, "tooltip": "Number of overlapping frames between source and new images"}),
                "overlap_side": (["source", "new_images"], {"default": "source", "tooltip": "Which side to overlap on"}),
                "overlap_mode": (["cut", "linear_blend", "ease_in_out", "filmic_crossfade", "perceptual_crossfade"], {"default": "linear_blend", "tooltip": "Method to use for overlapping frames"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch_img_multi"
    CATEGORY = "⛧Heresy Nodes/image"
    DESCRIPTION = "Stitch multiple image batches with overlap"

    def stitch_img_multi(self, source_dict, overlap, overlap_side, overlap_mode):
        # Sorted the input dictionary by keys in ascending order before extracting values
        sorted_keys = sorted(source_dict.keys())
        batches = [source_dict[k] for k in sorted_keys]

        # Fallbacks for empty or single inputs
        if not batches:
            return (torch.zeros((1, 64, 64, 3), device="cpu"),)
        if len(batches) == 1:
            return (batches[0],)
        if overlap == 0:
            # If no overlap, just stitch them all together immediately
            return (torch.cat(batches, dim=0),)

        # Validate shapes and overlaps against the first batch
        base_shape = batches[0].shape[1:3]
        device = batches[0].device
        dtype = batches[0].dtype

        for i, b in enumerate(batches):
            if b.shape[1:3] != base_shape:
                raise ValueError(f"Batch {i} shape {b.shape[1:3]} does not match base shape {base_shape}")
            if overlap > len(b):
                raise ValueError(f"Overlap ({overlap}) cannot be greater than length of batch {i} ({len(b)})")

        # --- PRECOMPUTE ALPHA ONCE ---
        # Moving this out of the loop saves VRAM and computation time.
        alpha = None
        if overlap_mode in ["linear_blend", "filmic_crossfade", "perceptual_crossfade"]:
            alpha = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
            alpha = alpha.view(-1, 1, 1, 1)
        elif overlap_mode == "ease_in_out":
            t = torch.linspace(0, 1, overlap + 2, device=device, dtype=dtype)[1:-1]
            eased_t = 3 * t * t - 2 * t * t * t
            alpha = eased_t.view(-1, 1, 1, 1)

        # Helper function to blend just the overlapping regions of two batches
        def blend_segments(src_tail, dst_head):
            if overlap_mode == "cut":
                return src_tail if overlap_side == "new_images" else dst_head

            # Swap logic for overlap sides
            if overlap_side == "source":
                b_src, b_dst = src_tail, dst_head
            else: # "new_images"
                b_src, b_dst = dst_head, src_tail

            if overlap_mode in ["linear_blend", "ease_in_out"]:
                return (1 - alpha) * b_src + alpha * b_dst

            elif overlap_mode == "filmic_crossfade":
                gamma = 2.2
                linear_src = torch.pow(b_src, gamma)
                linear_dst = torch.pow(b_dst, gamma)
                blended = (1 - alpha) * linear_src + alpha * linear_dst
                return torch.pow(blended, 1.0 / gamma)

            elif overlap_mode == "perceptual_crossfade":
                import kornia
                src_nchw = b_src.movedim(-1, 1)
                dst_nchw = b_dst.movedim(-1, 1)
                lab_src = kornia.color.rgb_to_lab(src_nchw)
                lab_dst = kornia.color.rgb_to_lab(dst_nchw)
                blended_lab = (1 - alpha) * lab_src + alpha * lab_dst
                blended_rgb = kornia.color.lab_to_rgb(blended_lab)
                return blended_rgb.movedim(1, -1)


        # --- BATCH PROCESSING (ZERO COPY) ---
        parts = []
        
        # 1. Add the prefix of the very first batch (everything except its tail)
        parts.append(batches[0][:-overlap])

        # 2. Iterate through adjacent pairs to blend overlaps and append middle sections
        for i in range(len(batches) - 1):
            src_batch = batches[i]
            dst_batch = batches[i+1]

            # Blend the tail of the current batch with the head of the next batch
            blended = blend_segments(src_batch[-overlap:], dst_batch[:overlap])
            parts.append(blended)

            # Append the non-overlapping middle section of the destination batch.
            # If it's a middle batch, leave off its tail (it will be blended next).
            # If it's the final batch, append the whole remaining tail.
            if i < len(batches) - 2:
                parts.append(dst_batch[overlap:-overlap])
            else:
                parts.append(dst_batch[overlap:])

        # 3. SINGLE CONCATENATION
        # PyTorch only allocates memory for the final tensor once here.
        return (torch.cat(parts, dim=0),)
    

class DeleteNumberedFolders:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Must specify a subfolder (e.g., "my_renders" or "test_batch/images")
                "relative_target_path": ("STRING", {"default": "", "multiline": False}),
                # 0 deletes everything matching the pattern. 5 keeps 01, 02, 03, 04, but deletes 05+.
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1, "tooltip": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_log",)
    FUNCTION = "del_num_folders"
    CATEGORY = "⛧Heresy Nodes/utils"
    OUTPUT_NODE = True 

    def del_num_folders(self, relative_target_path, start_index):
        # 1. Clean the input to prevent whitespace bypasses
        relative_target_path = relative_target_path.strip()
        
        # 2. CHECK: Forbid empty strings outright
        if not relative_target_path:
            error_msg = "Error: Target path cannot be empty. You must specify a nested folder."
            print(f"[DeleteNumberedFolders] {error_msg}")
            return (error_msg,)

        # 3. Get the absolute path to ComfyUI's standard output directory
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        
        # 4. Construct the target directory and resolve its absolute path (cleans up any ../..)
        target_dir = os.path.abspath(os.path.join(output_dir, relative_target_path))
        
        # 5. ROOT FOLDER CHECK: Ensure they aren't targeting the root output folder itself (e.g., via ".")
        if target_dir == output_dir:
            error_msg = "Error: Deleting directly from the root output folder is forbidden. Please specify a nested folder."
            print(f"[DeleteNumberedFolders] {error_msg}")
            return (error_msg,)

        # 6. SECURITY CHECK: Ensure the target directory strictly resides within the ComfyUI output directory
        if os.path.commonpath([output_dir]) != os.path.commonpath([output_dir, target_dir]):
            error_msg = f"Security Violation: Target path '{target_dir}' is outside the ComfyUI output directory. Aborting."
            print(f"[DeleteNumberedFolders] {error_msg}")
            return (error_msg,)
            
        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            warning_msg = f"Skipped: Target directory '{target_dir}' does not exist."
            print(f"[DeleteNumberedFolders] {warning_msg}")
            return (warning_msg,)

        # 7. Regex Pattern: Exactly two digits at the start, optionally followed by "_L" at the end
        pattern = re.compile(r"^(\d{2})(?:_L)?$")
        deleted_dirs = []

        # 8. Iterate and conditionally delete
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    # Extract the numeric string (e.g., "03") and convert to integer (3)
                    num = int(match.group(1))
                    
                    if num >= start_index:
                        try:
                            shutil.rmtree(item_path)
                            deleted_dirs.append(item)
                        except Exception as e:
                            print(f"[DeleteNumberedFolders] Failed to delete {item_path}: {e}")

        # 9. Format the output log
        if deleted_dirs:
            # Sort for a cleaner log readout
            deleted_dirs.sort() 
            log = f"Deleted {len(deleted_dirs)} folders: {', '.join(deleted_dirs)}"
        else:
            log = "No matching folders found to delete based on current criteria."
            
        print(f"[DeleteNumberedFolders] {log}")
        
        return (log,)
    
class SaveDictLatentAndImages:
    """
    Saves a Wan video latent and batches of images from a dictionary 
    by routing them through SaveWanVideoLatent and SaveImageBatchNoMeta instances.
    Skips saving if the corresponding numbered folder already exists.
    """
    def __init__(self):
        # Instantiate the classes defined earlier in the file
        self.latent_saver = SaveWanVideoLatent()
        self.image_saver = SaveImageBatchNoMeta()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Wan video latent tensor"}),
                "img_dict": ("DICT", {"tooltip": "Dictionary mapping keys (e.g., '01') to image batches"}),
                "project_subpath": ("STRING", {"tooltip": "Subpath in the output folder"}),
                "folders_str": ("STRING", {"multiline": True, "tooltip": "Newline separated string of existing folders"}),
                "latent_filename": ("STRING", {"tooltip": "Filename without extension for the latent"}),
                "latents_subfolder": ("STRING", {"tooltip": "Subfolder name for the safetensor"}),
                "images_subfolder": ("STRING", {"tooltip": "Subfolder name for the image batch"}),
            },
            "optional": {
                "any": ("*", {"tooltip": "For chaining"}),
            }
        }
    
    RETURN_TYPES = ("STRING","*")
    RETURN_NAMES = ("save_log","any")
    FUNCTION = "save_data"
    OUTPUT_NODE = False
    CATEGORY = "⛧Heresy Nodes/utils"
    DESCRIPTION = "Routes dictionary data to the single latent and image batch saving nodes."

    def save_data(self, latent, img_dict, project_subpath, folders_str, latent_filename, latents_subfolder, images_subfolder, any=None):
        
        # Clean the base project path
        safe_project_subpath = os.path.normpath(project_subpath).replace("..", "_")
        existing_folders = [f.strip() for f in folders_str.split("\n") if f.strip()]
        log = []

        for key, img_batch in img_dict.items():
            key_str = str(key)
            
            # 1. Determine Target Folder Name & Check if it exists
            if key_str in existing_folders or f"{key_str}_L" in existing_folders:
                skip_msg = f"[Skipped] Folder for '{key_str}' already exists. No data saved."
                print(f"[SaveDictIO] {skip_msg}")
                log.append(skip_msg)
                continue # Skip processing this dictionary entry entirely
            else:
                # If neither exists, we are creating a new folder. 
                # Since we are saving a latent, it always gets the _L addition.
                target_folder_name = f"{key_str}_L"
            
            # 2. Setup and trigger Latent Saving
            # Combine paths: project_path / 01_L / latents
            latent_subdir = os.path.join(safe_project_subpath, target_folder_name, latents_subfolder)
            # Ensure forward slashes for ComfyUI subpath compatibility cross-platform
            latent_subdir = latent_subdir.replace("\\", "/")
            
            try:
                # Call the instance method from SaveWanVideoLatent
                self.latent_saver.save_latent(
                    latent=latent,
                    filename=latent_filename,
                    enabled=True,
                    subdirectory=latent_subdir
                )
                log.append(f"[Latent] Saved successfully to {latent_subdir}/{latent_filename}.safetensors")
            except Exception as e:
                err_msg = f"[Latent] Error saving for {target_folder_name}: {e}"
                print(f"[SaveDictIO] {err_msg}")
                log.append(err_msg)

            # 3. Setup and trigger Image Batch Saving
            # Combine paths: project_path / 01_L / images / i
            img_prefix = os.path.join(safe_project_subpath, target_folder_name, images_subfolder, "i")
            # Ensure forward slashes so Comfy's `get_save_image_path` correctly creates the subdirectories
            img_prefix = img_prefix.replace("\\", "/")
            
            try:
                # Call the instance method from SaveImageBatchNoMeta
                self.image_saver.save_images(
                    images=img_batch,
                    filename_prefix=img_prefix,
                    include_workflow=False,
                    enabled=True
                )
                log.append(f"[Images] Saved batch using prefix: {img_prefix}")
            except Exception as e:
                err_msg = f"[Images] Error saving batch for {target_folder_name}: {e}"
                print(f"[SaveDictIO] {err_msg}")
                log.append(err_msg)

        # Return formatted multiline string as log
        return ("\n".join(log), any)
    