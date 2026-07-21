from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

MODEL = "../../models/SmolVLM-500M-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    local_files_only=True
)
model.to("cuda")

print("Opening image...")
image = Image.open("publichmarket.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text",
             "text": "Describe this image in detail."}
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Running inference...")

output = model.generate(
    **inputs,
    max_new_tokens=128
)

answer = processor.decode(
    output[0],
    skip_special_tokens=True
)

print()
print(answer)
