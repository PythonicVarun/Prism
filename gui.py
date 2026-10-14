import math
import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        QueryOutput = self.QueryMatrix(x)
        KeyOutput = self.KeyMatrix(x)
        ValueOutput = self.ValueMatrix(x)

        QueryOutput = QueryOutput.view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(
            1, 2
        )
        KeyOutput = KeyOutput.view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(
            1, 2
        )
        ValueOutput = ValueOutput.view(BatchSize, SeqLen, NumHeads, HeadDims).transpose(
            1, 2
        )

        KeyOutput = KeyOutput.transpose(2, 3)
        MatMulOutput = QueryOutput @ KeyOutput
        MatMulOutput /= math.sqrt(HeadDims)
        MatMulOutput = self.Softmax(MatMulOutput)
        AttentionOutput = MatMulOutput @ ValueOutput

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
        HistoryList = torch.stack(HistoryList, dim=0)
        HistoryListNorm = self.LayerNorm1(HistoryList)

        Scores = (HistoryListNorm * self.PseudoQuery).sum(dim=-1) / math.sqrt(EmbedDim)
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
        self.HighResConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.GELU(),
        )
        self.InputLayer = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=16, stride=16
        )
        self.Flatten = nn.Flatten(start_dim=2)
        self.LearnParam = nn.Parameter(torch.randn((1, 256, 256)) * 0.02)
        self.ModuleList = nn.ModuleList([PrismBlock() for _ in range(8)])
        self.FinalNorm = nn.LayerNorm(EmbedDim)
        self.Decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.Decoder2 = nn.Sequential(
            nn.Conv2d(
                in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(
                in_channels=128, out_channels=313, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        SkipFeatures = self.HighResConv(x)
        x = self.InputLayer(SkipFeatures)
        x = self.Flatten(x)
        x = x.transpose(1, 2)
        x = x + self.LearnParam

        HistoryList = [x]
        for module in self.ModuleList:
            x = module(x, HistoryList)
            HistoryList.append(x)

        x = self.FinalNorm(x)
        x = x.transpose(1, 2).reshape(-1, 256, 16, 16)
        x = self.Decoder1(x)
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = torch.cat([x, SkipFeatures], dim=1)
        x = self.Decoder2(x)

        return x


def CleanCheckpointStateDict(RawStateDict):
    CleanStateDict = {}
    for key, value in RawStateDict.items():
        CleanKey = key.replace("_orig_mod.", "").replace("module.", "")
        CleanStateDict[CleanKey] = value
    return CleanStateDict


def load_model_and_resources(
    model_path="PrismModel.pth", buckets_path="CoordBuckets.npy"
):
    model = PrismModel().to(device)
    model.eval()

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("ModelState", checkpoint)
        model.load_state_dict(CleanCheckpointStateDict(state_dict))
        print("Model loaded successfully")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model: {e}")
        print("Using untrained model")

    buckets_tensor = None
    try:
        coord_buckets = np.load(buckets_path)
        buckets_tensor = torch.from_numpy(coord_buckets).float().to(device)
        print("Coordinate buckets loaded successfully")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading coordinate buckets: {e}")

    return model, buckets_tensor


print("Loading Prism Colorization Model...")
model, buckets_tensor = load_model_and_resources()


@torch.no_grad()
def colorize_image(input_image, temperature=DefaultTemperature):
    """
    Colorize a grayscale or color image using the Prism Model.
    """
    if input_image is None:
        return None

    if isinstance(input_image, Image.Image):
        input_array = np.array(input_image)
    else:
        input_array = input_image

    if len(input_array.shape) == 3 and input_array.shape[2] == 4:
        input_array = cv2.cvtColor(input_array, cv2.COLOR_RGBA2RGB)

    if len(input_array.shape) == 3:
        img_lab = cv2.cvtColor(input_array, cv2.COLOR_RGB2LAB)
        l_orig = img_lab[:, :, 0]
    else:
        l_orig = input_array

    orig_h, orig_w = l_orig.shape[:2]

    if buckets_tensor is None:
        print("Using fallback grayscale output (coordinate buckets not available)")
        fallback = np.stack([l_orig, l_orig, l_orig], axis=-1).astype(np.uint8)
        return Image.fromarray(fallback)

    l_resized = cv2.resize(l_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    input_tensor = (
        (torch.from_numpy(l_resized).float() / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16

    with torch.amp.autocast(amp_device, dtype=amp_dtype):
        logits = model(input_tensor)

    logits = logits.float()
    temperature = max(float(temperature), 1e-4)
    probs = torch.nn.functional.softmax(logits / temperature, dim=1)
    pred_ab = torch.einsum("b c h w, c d -> b d h w", probs, buckets_tensor)

    pred_ab_numpy = pred_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_ab_resized = cv2.resize(
        pred_ab_numpy, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
    )

    pred_a = np.clip(pred_ab_resized[..., 0] + 128.0, 0, 255).astype(np.uint8)
    pred_b = np.clip(pred_ab_resized[..., 1] + 128.0, 0, 255).astype(np.uint8)

    merged_lab = cv2.merge([l_orig, pred_a, pred_b])
    colorized_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(colorized_rgb)


def gradio_colorize(image, temperature=DefaultTemperature):
    if image is None:
        return None
    try:
        return colorize_image(image, temperature)
    except Exception as e:  # noqa: BLE001
        print(f"Error during colorization: {e}")
        return None


def create_interface():
    with gr.Blocks(title="Prism - Image Colorization") as demo:
        gr.Markdown("""
        # 🎨 Prism - Image Colorization
        
        Transform grayscale images into vibrant color photos using deep learning!
        
        **How to use:**
        1. Upload a grayscale or color image (or choose an example below)
        2. Adjust the **Temperature** slider to control color vibrancy and diversity
        3. Click **Colorize** to view the restored colors
        
        **Model:** Prism (Transformer routing blocks with High-Res skip decoder)
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Image")
                input_img = gr.Image(
                    label="Upload Image (grayscale or color)",
                    type="pil",
                    image_mode="RGB",
                )
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=DefaultTemperature,
                    step=0.01,
                    label="Color Temperature",
                    info="Lower values produce more conservative colors; higher values yield more vibrant, diverse colors.",
                )
                with gr.Row():
                    submit_btn = gr.Button("🎨 Colorize", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")

            with gr.Column():
                gr.Markdown("### Output Image")
                output_img = gr.Image(
                    label="Colorized Output",
                    type="pil",
                )

        submit_btn.click(
            fn=gradio_colorize,
            inputs=[input_img, temp_slider],
            outputs=[output_img],
        )

        clear_btn.click(
            fn=lambda: (None, DefaultTemperature, None),
            outputs=[input_img, temp_slider, output_img],
        )

        gr.Markdown("### Try Examples")
        examples_list = []
        if os.path.isdir("examples/"):
            for f in sorted(os.listdir("examples/")):
                if f.endswith((".png", ".jpg", ".jpeg")):
                    examples_list.append([f"examples/{f}", DefaultTemperature])

        if examples_list:
            gr.Examples(
                examples=examples_list,
                inputs=[input_img, temp_slider],
                outputs=[output_img],
                fn=gradio_colorize,
                cache_examples=False,
            )

        gr.Markdown("""
        ---
        ### About
        This model was trained on image colorization using the latest Prism architecture:
        - Grayscale luminance input (`L` channel, normalized to `[0, 1]`)
        - High-resolution skip connections with multi-head attention routing blocks
        - Soft color assignment over 313 quantized LAB coordinate buckets using annealed temperature softmax
        - Preserves input resolution by merging predicted chroma (`ab`) channels with original luminance (`L`)
        """)

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎨 PRISM Image Colorization - Gradio App")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 60 + "\n")

    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
