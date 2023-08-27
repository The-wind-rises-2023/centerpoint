import onnx
from onnxsim import simplify

# Load the original ONNX model
original_model_path = '/home/jing/Workspace/mmdeploy/end2end.onnx'
original_model = onnx.load(original_model_path)
import ipdb;ipdb.set_trace()
# Simplify the model
simplified_model,check = simplify(original_model)

# Save the simplified model
simplified_model_path = '/home/jing/Workspace/mmdeploy/end2end_sim.onnx'
onnx.save(simplified_model, simplified_model_path)

