import torch
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Dummy input (IMPORTANT: 3 channels)
dummy = torch.randn(1, 3, 224, 224)

# Wrapper to isolate image path
class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)

image_model = ImageEncoder(model)

torch.onnx.export(
    image_model,
    dummy,
    "../../mdls/clip_image.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    opset_version=17,
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "image_embeddings": {0: "batch"}
    }
)