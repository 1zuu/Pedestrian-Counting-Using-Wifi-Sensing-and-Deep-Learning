import platform
import numpy as np

if 'windows' in platform.architecture()[1].lower():
	from tensorflow import lite as tflite
else:
	import tflite_runtime.interpreter as tflite

class PedestrianCounting(object):
	def __init__(self, model_converter):
		self.model_converter = model_converter
		
	def TFinterpreter(self):
		self.interpreter = tflite.Interpreter(model_path=self.model_converter) # Load tflite model
		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details() # Get input details of the model
		self.output_details = self.interpreter.get_output_details() # Get output details of the model

	def Inference(self, features):
		features = features.astype(np.float32)
		input_shape = self.input_details[0]['shape']
		assert np.array_equal(input_shape, features.shape), "Input tensor hasn't correct dimension"

		self.interpreter.set_tensor(self.input_details[0]['index'], features)

		self.interpreter.invoke() # set the inference

		output_data = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
		return output_data