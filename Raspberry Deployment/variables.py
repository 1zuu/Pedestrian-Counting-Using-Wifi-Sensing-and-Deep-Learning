tflite_file = 'docs/dnn_model_converter.tflite'

input_shape_inf = [1,103]
output_shape_inf = [1,13]

port = 8050
# host = '10.10.10.100'
host = 'localhost'
inf_url = 'http://{}:{}/test/'.format(host,port)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}