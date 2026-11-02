import io
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
    .add_local_file(
        Path(__file__).parent / "favicon.ico", remote_path="/root/favicon.ico"
    )
)

volume = modal.Volume.from_name("prism-weights", create_if_missing=True)
VOLUME_PATH = "/weights"

app = modal.App("prism-colorizer")

with image.imports():
    import math
    import os
    import time

    import cv2
    import gradio as gr
    import numpy as np
    import torch
    from fastapi import FastAPI
    from PIL import Image
    from torch import nn


NumHeads = 8
HeadDims = 32
EmbedDim = 256
DefaultTemperature = 0.38


@app.cls(
    image=image,
    volumes={VOLUME_PATH: volume},
    cpu=8,
    memory=16 * 1024,
    timeout=600,
    min_containers=0,
    max_containers=2,
)
class Colorizer:
    @modal.enter()
    def load(self):
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        device = torch.device("cpu")
        self.device = device

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.QueryMatrix = nn.Linear(EmbedDim, EmbedDim)
                self.KeyMatrix = nn.Linear(EmbedDim, EmbedDim)
                self.ValueMatrix = nn.Linear(EmbedDim, EmbedDim)
                self.OutputMatrix = nn.Linear(EmbedDim, EmbedDim)
                self.Softmax = nn.Softmax(dim=-1)

            def forward(self, x):
                # SeqLen = 256 currently
                BatchSize, SeqLen, _ = x.shape

                QueryOutput = self.QueryMatrix(x)
                KeyOutput = self.KeyMatrix(x)
                ValueOutput = self.ValueMatrix(x)

                # 8 Heads
                QueryOutput = QueryOutput.view(
                    BatchSize, SeqLen, NumHeads, HeadDims
                ).transpose(1, 2)  # B,NumHeads,256,HeadDims
                KeyOutput = KeyOutput.view(
                    BatchSize, SeqLen, NumHeads, HeadDims
                ).transpose(1, 2)  # B,NumHeads,256,HeadDims
                ValueOutput = ValueOutput.view(
                    BatchSize, SeqLen, NumHeads, HeadDims
                ).transpose(1, 2)  # B,NumHeads,256,HeadDims

                # Attention(Q, K, V) = (softmax((QK^T)/sqrt(dk)))*V, dk = HeadDims
                AttentionOutput = torch.nn.functional.scaled_dot_product_attention(
                    QueryOutput, KeyOutput, ValueOutput
                )
                AttentionOutput = (
                    AttentionOutput.transpose(1, 2)
                    .contiguous()
                    .view(BatchSize, SeqLen, EmbedDim)
                )

                FinalOutput = self.OutputMatrix(AttentionOutput)

                return FinalOutput

        class PrismBlock(nn.Module):
            def __init__(self):
                super().__init__()

                self.PseudoQuery = nn.Parameter(torch.randn(EmbedDim) * 0.02)
                self.Attention = Attention()
                self.MultiLayerPerceptron = nn.Sequential(
                    nn.Linear(in_features=EmbedDim, out_features=EmbedDim * 4),
                    nn.GELU(),
                    nn.Linear(in_features=EmbedDim * 4, out_features=EmbedDim),
                )
                self.LayerNorm1 = nn.LayerNorm(EmbedDim)
                self.LayerNorm2 = nn.LayerNorm(EmbedDim)
                self.LayerNorm3 = nn.LayerNorm(EmbedDim)
                self.Softmax = nn.Softmax(dim=0)

            def forward(self, x, HistoryList):
                HistoryList = torch.stack(HistoryList, dim=0)  # D,B,SeqLen,EmbedDim
                HistoryListNorm = self.LayerNorm1(HistoryList)

                Scores = (HistoryListNorm * self.PseudoQuery).sum(dim=-1) / math.sqrt(
                    EmbedDim
                )  # D,B,SeqLen
                Weights = self.Softmax(Scores).unsqueeze(-1)

                XRouted = (HistoryList * Weights).sum(dim=0)

                XRoutedNorm = self.LayerNorm2(XRouted)
                x = x + self.Attention(XRoutedNorm)

                xNorm = self.LayerNorm3(x)
                x = x + self.MultiLayerPerceptron(xNorm)

                return x

        class PrismModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.Stem = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                )
                self.Stage1 = nn.Sequential(
                    nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),  # 256 -> 128
                    nn.GroupNorm(16, 96),
                    nn.GELU(),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 96),
                    nn.GELU(),
                )
                self.Stage2 = nn.Sequential(
                    nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                    nn.GroupNorm(16, 128),
                    nn.GELU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 128),
                    nn.GELU(),
                )
                self.Stage3 = nn.Sequential(
                    nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                    nn.GroupNorm(16, 192),
                    nn.GELU(),
                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 192),
                    nn.GELU(),
                )
                self.PatchEmbed = nn.Sequential(
                    nn.Conv2d(192, 256, kernel_size=2, stride=2),  # 32 -> 16
                )

                self.Flatten = nn.Flatten(start_dim=2)  # B, 256, 256
                self.LearnParam = nn.Parameter(torch.randn((1, 256, 256)) * 0.02)

                self.ModuleList = nn.ModuleList([PrismBlock() for _ in range(8)])
                self.FinalNorm = nn.LayerNorm(EmbedDim)

                self.Decoder3 = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    ),  # 16 -> 32
                    nn.Conv2d(256 + 192, 192, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 192),
                    nn.GELU(),
                )
                self.Decoder2 = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    ),  # 32 -> 64
                    nn.Conv2d(192 + 128, 128, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 128),
                    nn.GELU(),
                )
                self.Decoder1 = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    ),  # 64 -> 128
                    nn.Conv2d(128 + 96, 96, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(16, 96),
                    nn.GELU(),
                )
                self.Decoder0 = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    ),  # 128 -> 256
                    nn.Conv2d(96 + 64, 64, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 313, kernel_size=1, stride=1),
                )

            def forward(self, x):
                Skip0 = self.Stem(x)  # B, 64, 256, 256
                Skip1 = self.Stage1(Skip0)  # B, 96, 128, 128
                Skip2 = self.Stage2(Skip1)  # B, 128, 64, 64
                Skip3 = self.Stage3(Skip2)  # B, 192, 32, 32

                Tokens = self.PatchEmbed(Skip3)  # B, 256, 16, 16
                Tokens = self.Flatten(Tokens)  # B, 256, 256
                Tokens = Tokens.transpose(1, 2) + self.LearnParam  # B, 256, 256

                HistoryList = [Tokens]
                for module in self.ModuleList:
                    Tokens = module(Tokens, HistoryList)
                    HistoryList.append(Tokens)

                Tokens = self.FinalNorm(Tokens)
                x = Tokens.transpose(1, 2).reshape(-1, 256, 16, 16)

                x = torch.cat([self.Decoder3[0](x), Skip3], dim=1)
                x = self.Decoder3[1:](x)

                x = torch.cat([self.Decoder2[0](x), Skip2], dim=1)
                x = self.Decoder2[1:](x)

                x = torch.cat([self.Decoder1[0](x), Skip1], dim=1)
                x = self.Decoder1[1:](x)

                x = torch.cat([self.Decoder0[0](x), Skip0], dim=1)
                x = self.Decoder0[1:](x)

                return x

        def load_resources():
            mdl = PrismModel().to(device)
            mdl.eval()
            try:
                ckpt = torch.load(
                    f"{VOLUME_PATH}/PrismModel.pth",
                    map_location=device,
                    weights_only=False,
                )
                state = {
                    k.replace("_orig_mod.", "").replace("module.", ""): v
                    for k, v in ckpt.get("ModelState", ckpt).items()
                }
                mdl.load_state_dict(state)
            except Exception as e:
                print(f"Model load error: {e}")
            buckets = None
            try:
                buckets = (
                    torch.from_numpy(np.load(f"{VOLUME_PATH}/CoordBuckets.npy"))
                    .float()
                    .to(device)
                )
            except Exception as e:
                print(f"Buckets load error: {e}")
            return mdl, buckets

        self.model, self.buckets_tensor = load_resources()

    @modal.method()
    def colorize(
        self, png_bytes: bytes, temperature: float = DefaultTemperature
    ) -> bytes:
        t0 = time.time()
        print(f"threads={torch.get_num_threads()}")
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        with torch.no_grad():
            out = self._colorize_image(img, temperature)

        print(f"colorize took {time.time() - t0:.1f}s, input {img.size}")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    def _colorize_image(self, input_image, temperature=DefaultTemperature):
        if input_image is None:
            return None

        arr = np.array(input_image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        l_orig = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[:, :, 0] if arr.ndim == 3 else arr
        orig_h, orig_w = l_orig.shape[:2]

        if self.buckets_tensor is None:
            return Image.fromarray(np.stack([l_orig] * 3, axis=-1).astype(np.uint8))

        t = (
            (
                torch.from_numpy(
                    cv2.resize(l_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
                ).float()
                / 255.0
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        logits = self.model(t)

        probs = torch.nn.functional.softmax(
            logits.float() / max(float(temperature), 1e-4), dim=1
        )
        pred_ab = cv2.resize(
            torch.einsum("bchw,cd->bdhw", probs, self.buckets_tensor)
            .squeeze(0)
            .cpu()
            .numpy()
            .transpose(1, 2, 0),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        lab = cv2.merge(
            [
                l_orig,
                np.clip(pred_ab[..., 0] + 128.0, 0, 255).astype(np.uint8),
                np.clip(pred_ab[..., 1] + 128.0, 0, 255).astype(np.uint8),
            ]
        )
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


@app.function(
    image=image,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def serve():
    colorizer = Colorizer()

    def gradio_colorize(image, temperature=DefaultTemperature):
        if image is None:
            return None
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out = colorizer.colorize.remote(buf.getvalue(), float(temperature))
            return Image.open(io.BytesIO(out))
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
                    minimum=0.1,
                    maximum=1.0,
                    value=DefaultTemperature,
                    step=0.01,
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

        submit_btn.click(
            fn=gradio_colorize,
            inputs=[input_img, temp_slider],
            outputs=[output_img],
            concurrency_limit=2,
        )
        clear_btn.click(
            fn=lambda: (None, DefaultTemperature, None),
            outputs=[input_img, temp_slider, output_img],
        )

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
        favicon_path="/root/favicon.ico",
    )
