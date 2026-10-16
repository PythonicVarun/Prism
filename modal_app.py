from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "gradio~=6.30.0",
        "opencv-python-headless~=5.0.0.93",
        "numpy~=2.5.4",
        "pillow~=12.3.0",
    )
    .add_local_dir(Path(__file__).parent / "examples", remote_path="/root/examples")
    .add_local_file(Path(__file__).parent / "favicon.ico", remote_path="/root/favicon.ico")
)

volume = modal.Volume.from_name("prism-weights", create_if_missing=True)
VOLUME_PATH = "/weights"

app = modal.App("prism-colorizer")

with image.imports():
    import math
    import os

    import cv2
    import gradio as gr
    import numpy as np
    import torch
    from fastapi import FastAPI
    from PIL import Image
    from torch import nn


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    cpu=2,
    memory=16 * 1024,
    timeout=600,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=3)
@modal.asgi_app()
def serve():
    device = torch.device("cpu")

    NumHeads = 8
    HeadDims = 32
    EmbedDim = 256
    DefaultTemperature = 0.38

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.QueryMatrix = nn.Linear(EmbedDim, EmbedDim)
            self.KeyMatrix = nn.Linear(EmbedDim, EmbedDim)
            self.ValueMatrix = nn.Linear(EmbedDim, EmbedDim)
            self.OutputMatrix = nn.Linear(EmbedDim, EmbedDim)
            self.Softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            BatchSize, SeqLen, _ = x.shape
            Q = self.QueryMatrix(x).view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(1, 2)
            K = self.KeyMatrix(x).view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(1, 2)
            V = self.ValueMatrix(x).view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(1, 2)
            attn = self.Softmax((Q @ K.transpose(2, 3)) / math.sqrt(HeadDims))
            return self.OutputMatrix((attn @ V).transpose(1, 2).contiguous().view(BatchSize, SeqLen, EmbedDim))

    class PrismBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.PseudoQuery = nn.Parameter(torch.randn(EmbedDim) * 0.02)
            self.Attention = Attention()
            self.MultiLayerPerceptron = nn.Sequential(
                nn.Linear(EmbedDim, EmbedDim * 4), nn.GELU(), nn.Linear(EmbedDim * 4, EmbedDim)
            )
            self.LayerNorm1 = nn.LayerNorm(EmbedDim)
            self.LayerNorm2 = nn.LayerNorm(EmbedDim)
            self.LayerNorm3 = nn.LayerNorm(EmbedDim)
            self.Softmax = nn.Softmax(dim=0)

        def forward(self, x, HistoryList):
            hist = torch.stack(HistoryList, dim=0)
            weights = self.Softmax((self.LayerNorm1(hist) * self.PseudoQuery).sum(dim=-1) / math.sqrt(EmbedDim)).unsqueeze(-1)
            x = x + self.Attention(self.LayerNorm2((hist * weights).sum(dim=0)))
            x = x + self.MultiLayerPerceptron(self.LayerNorm3(x))
            return x

    class PrismModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.HighResConv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.GELU())
            self.InputLayer = nn.Conv2d(64, 256, kernel_size=16, stride=16)
            self.Flatten = nn.Flatten(start_dim=2)
            self.LearnParam = nn.Parameter(torch.randn((1, 256, 256)) * 0.02)
            self.ModuleList = nn.ModuleList([PrismBlock() for _ in range(8)])
            self.FinalNorm = nn.LayerNorm(EmbedDim)
            self.Decoder1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(32, 256), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(32, 256), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(32, 256), nn.GELU(),
            )
            self.Decoder2 = nn.Sequential(
                nn.Conv2d(320, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
                nn.Conv2d(128, 313, 3, 1, 1),
            )

        def forward(self, x):
            skip = self.HighResConv(x)
            x = self.Flatten(self.InputLayer(skip)).transpose(1, 2) + self.LearnParam
            history = [x]
            for block in self.ModuleList:
                x = block(x, history)
                history.append(x)
            x = self.FinalNorm(x).transpose(1, 2).reshape(-1, 256, 16, 16)
            x = self.Decoder1(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return self.Decoder2(torch.cat([x, skip], dim=1))

    def load_resources():
        mdl = PrismModel().to(device)
        mdl.eval()
        try:
            ckpt = torch.load(f"{VOLUME_PATH}/PrismModel.pth", map_location=device, weights_only=False)
            state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.get("ModelState", ckpt).items()}
            mdl.load_state_dict(state)
        except Exception as e:
            print(f"Model load error: {e}")
        buckets = None
        try:
            buckets = torch.from_numpy(np.load(f"{VOLUME_PATH}/CoordBuckets.npy")).float().to(device)
        except Exception as e:
            print(f"Buckets load error: {e}")
        return mdl, buckets

    model, buckets_tensor = load_resources()

    @torch.no_grad()
    def colorize_image(input_image, temperature=DefaultTemperature):
        if input_image is None:
            return None
        arr = np.array(input_image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        l_orig = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[:, :, 0] if arr.ndim == 3 else arr
        orig_h, orig_w = l_orig.shape[:2]

        if buckets_tensor is None:
            return Image.fromarray(np.stack([l_orig] * 3, axis=-1).astype(np.uint8))

        t = (torch.from_numpy(cv2.resize(l_orig, (256, 256), interpolation=cv2.INTER_CUBIC)).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            logits = model(t)

        probs = torch.nn.functional.softmax(logits.float() / max(float(temperature), 1e-4), dim=1)
        pred_ab = cv2.resize(
            torch.einsum("bchw,cd->bdhw", probs, buckets_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.merge([l_orig, np.clip(pred_ab[..., 0] + 128.0, 0, 255).astype(np.uint8), np.clip(pred_ab[..., 1] + 128.0, 0, 255).astype(np.uint8)])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def gradio_colorize(image, temperature=DefaultTemperature):
        try:
            return colorize_image(image, temperature)
        except Exception as e:
            print(f"Colorize error: {e}")
            return None

    with gr.Blocks(title="Prism - Image Colorization") as demo:
        gr.Markdown("""
        # 🎨 Prism - Image Colorization
        Transform grayscale images into vibrant color photos using deep learning!

        **How to use:**
        1. Upload a grayscale or color image
        2. Adjust the **Temperature** slider to control color vibrancy
        3. Click **Colorize**

        **Model:** Prism (Transformer routing blocks with High-Res skip decoder)
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Image")
                input_img = gr.Image(label="Upload Image", type="pil", image_mode="RGB")
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=DefaultTemperature, step=0.01,
                    label="Color Temperature",
                    info="Lower = conservative colors; Higher = vibrant, diverse colors.",
                )
                with gr.Row():
                    submit_btn = gr.Button("🎨 Colorize", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")
            with gr.Column():
                gr.Markdown("### Output Image")
                output_img = gr.Image(label="Colorized Output", type="pil")

        example_dir = "/root/examples"
        example_files = (
            sorted(
                os.path.join(example_dir, f)
                for f in os.listdir(example_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if os.path.isdir(example_dir)
            else []
        )
        if example_files:
            gr.Markdown("### Try an Example")
            gr.Examples(examples=example_files, inputs=[input_img], label="Examples")

        submit_btn.click(fn=gradio_colorize, inputs=[input_img, temp_slider], outputs=[output_img])
        clear_btn.click(fn=lambda: (None, DefaultTemperature, None), outputs=[input_img, temp_slider, output_img])

        gr.Markdown("""
        ---
        ### About
        - Grayscale luminance input (`L` channel) → predicts `ab` chroma over 313 LAB buckets
        - Multi-head attention routing blocks with high-res skip connections
        - Temperature-scaled softmax for color diversity control

        ### Authors

        - **Nishant Kalaichelvan** - [@nkminion](https://github.com/nkminion)
        - **Varun Agnihotri** - [@PythonicVarun](https://github.com/PythonicVarun) - [hello@pythonicvarun.me](mailto:hello@pythonicvarun.me)

        ### License
        See the [LICENSE](https://github.com/nkminion/Prism/blob/main/LICENSE) file for more details.
        """)

    return gr.mount_gradio_app(
        app=FastAPI(),
        blocks=demo,
        path="/",
        show_error=True,
        theme=gr.themes.Base(),
        favicon_path="/root/favicon.ico"
    )
