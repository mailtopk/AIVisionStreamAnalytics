from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq

MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForVision2Seq.from_pretrained(MODEL)
model.save_pretrained("../../models")
processor.save_pretrained("../../models")

print("Model loaded successfully!")
