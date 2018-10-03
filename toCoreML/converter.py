# converter.py
import tfcoreml

# Supply a dictionary of input tensors' name and shape (with batch axis)
input_tensor_shapes = {"input:0": [1, None, None, 3]} # batch size is 1
# TF graph definition
tf_model_path = './HousePlantIdentifier.pb'
# Output CoreML model path
coreml_model_file = './HousePlantIdentifier.mlmodel'
# The TF model's ouput tensor name
output_tensor_names = ['final_result:0']

# Call the converter. This may take a while
coreml_model = tfcoreml.convert (
tf_model_path=tf_model_path,
mlmodel_path=coreml_model_file,
input_name_shape_dict=input_tensor_shapes,
output_feature_names=output_tensor_names,
image_input_names = ['input:0'],
red_bias = -1,
green_bias = -1,
blue_bias = -1,
image_scale = 2.0/255.0)