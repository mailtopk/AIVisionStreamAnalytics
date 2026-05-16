import onnx
model = onnx.load('../../mdls/model.onnx') #OpenAI clip model
for inp in model.graph.input:
    print(inp)


# ame: "input_ids"
# type {
#   tensor_type {
#     elem_type: 7
#     shape {
#       dim {
#         dim_param: "text_batch_size"
#       }
#       dim {
#         dim_param: "sequence_length"
#       }
#     }
#   }
# }
# name: "pixel_values"
# type {
#   tensor_type {
#     elem_type: 1
#     shape {
#       dim {
#         dim_param: "image_batch_size"
#       }
#       dim {
#         dim_param: "num_channels"
#       }
#       dim {
#         dim_param: "height"
#       }
#       dim {
#         dim_param: "width"
#       }
#     }
#   }
# }
# name: "attention_mask"
# type {
#   tensor_type {
#     elem_type: 7
#     shape {
#       dim {
#         dim_param: "text_batch_size"
#       }
#       dim {
#         dim_param: "sequence_length"
#       }
#     }
#   }
# }