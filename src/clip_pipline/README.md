## Extract video .engine

```
trtexec \
  --onnx=clip_image.onnx \
  --fp16 \
  --shapes=pixel_values:1x3x224x224 \
  --saveEngine=clip.engine
```