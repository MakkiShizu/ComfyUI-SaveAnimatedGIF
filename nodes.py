import folder_paths
from comfy.cli_args import args

# import cv2
import os
import json
from PIL import Image
import numpy as np


class SaveAnimatedWEBPRevise:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    formats = {
        "webp": "WEBP",
        "gif": "GIF",
        # "mp4": "MP4"
    }  # Added support for gif and mp4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                "method": (list(s.methods.keys()),),
                "format": (list(s.formats.keys()),),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/animation"

    def save_images(
        self,
        images,
        fps,
        filename_prefix,
        lossless,
        quality,
        method,
        format="webp",
        num_frames=0,
        prompt=None,
        extra_pnginfo=None,
    ):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        pil_images = []

        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010F
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(
                        x, json.dumps(extra_pnginfo[x])
                    )
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        c = len(pil_images)

        if format == "webp":
            # Saving as animated WEBP
            for i in range(0, c, num_frames):
                file = f"{filename}_{counter:05}_.webp"
                pil_images[i].save(
                    os.path.join(full_output_folder, file),
                    save_all=True,
                    duration=int(1000.0 / fps),
                    append_images=pil_images[i + 1 : i + num_frames],
                    exif=metadata,
                    lossless=lossless,
                    quality=quality,
                    method=method,
                )
                results.append(
                    {"filename": file, "subfolder": subfolder, "type": self.type}
                )
                counter += 1
        elif format == "gif":
            # Saving as GIF (a single animated GIF file)
            gif_filename = f"{filename}_{counter:05}_.gif"
            pil_images[0].save(
                os.path.join(full_output_folder, gif_filename),
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000.0 / fps),
                loop=0,
                optimize=True,
                lossless=lossless,
            )
            results.append(
                {"filename": gif_filename, "subfolder": subfolder, "type": self.type}
            )
            counter += 1
        # elif format == "mp4":
        #     # Saving as MP4 (using OpenCV)
        #     mp4_filename = f"{filename}_{counter:05}_.mp4"
        #     height, width = pil_images[0].size
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
        #     video_writer = cv2.VideoWriter(os.path.join(full_output_folder, mp4_filename), fourcc, fps, (width, height))

        #     for img in pil_images:
        #         frame = np.array(img)
        #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        #         video_writer.write(frame)

        #     video_writer.release()
        #     results.append({
        #         "filename": mp4_filename,
        #         "subfolder": subfolder,
        #         "type": self.type
        #     })
        #     counter += 1

        animated = num_frames != 1
        return {"ui": {"images": results, "animated": (animated,)}}


class SaveAnimatedGIF:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "lossless": ("BOOLEAN", {"default": True}),
                # "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                "method": (list(s.methods.keys()),),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/animation"

    def save_images(
        self,
        images,
        fps,
        filename_prefix,
        lossless,
        # quality,
        method,
        num_frames=0,
        prompt=None,
        extra_pnginfo=None,
    ):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        pil_images = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010F
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(
                        x, json.dumps(extra_pnginfo[x])
                    )
                    inital_exif -= 1

        gif_filename = f"{filename}_{counter:05}_.gif"
        pil_images[0].save(
            os.path.join(full_output_folder, gif_filename),
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000.0 / fps),
            loop=0,
            optimize=True,
            lossless=lossless,
        )
        results.append(
            {"filename": gif_filename, "subfolder": subfolder, "type": self.type}
        )
        counter += 1

        animated = num_frames != 1
        return {"ui": {"images": results, "animated": (animated,)}}


NODE_CLASS_MAPPINGS = {
    "SaveAnimatedWEBPRevise": SaveAnimatedWEBPRevise,
    "SaveAnimatedGIF": SaveAnimatedGIF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAnimatedWEBPRevise": "SaveAnimatedWEBPRevise",
    "SaveAnimatedGIF": "SaveAnimatedGIF",
}
