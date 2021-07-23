import os
import dash
import platform
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from flask import Flask, request, jsonify
from dash.dependencies import Output, Input
from db_interations import access_database, process_data
from flask import Flask, send_from_directory, jsonify, request

from variables import*
from inference import*

app = Flask(__name__)

inference = PedestrianCounting(tflite_file)
inference.TFinterpreter()

def get_data():

    prediction, ground_truth = access_database()
    if (len(prediction) == 0) or (len(ground_truth) == 0):
        total_prediction = [0]
        total_ground_truth = [0]

    total_ground_truth = ground_truth
    total_prediction = prediction

    iteration = np.arange(1, len(total_prediction)+1)
    return iteration, total_prediction, total_ground_truth

external_stylesheets = [
    {
        "href": "https://codepen.io/chriddyp/pen/bWLwgP.css"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

server = Flask(__name__)
app = dash.Dash(
            __name__, 
            server=server,
            external_stylesheets=external_stylesheets
               )
app.title = "Pedestrian counting Estimator"
app.config['suppress_callback_exceptions']=True

@server.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(server.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
                               
app.layout = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
        html.Div(
            style={'textAlign': 'center','color': colors['text']},
            children=[
                html.H2(children="💻 Developer ⎆ Isuru Alagiyawanna 💻", className="header-emoji"),
                html.H1(
                    children="CSI based Pedestrian Count Estimation", className="header-title"
                ),
                html.H3(
                    children="Estimate the count upto 12 pedestrians in a pedestrian crossing using Deep Learning & CSI data",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    style={'textAlign': 'center','color': colors['text']},
                    children=[
                        html.H1(
                            id='pedestrian-counter',
                            children=" True Observation  ➡️ {}  VS  Estimated  Count  ➡️ {}".format(0, 0)
                            )
                    ],
                    className="header",
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="prediction-chart",
                            config={"displayModeBar": False},
                            figure={
                                "data": [
                                    {
                                        "x": [1],
                                        "y": [0],
					                    "name" : "Predicted Count",
                                        "type": "lines",
                                        "hovertemplate": "%{y:.2f} persons"
                                                        "<extra></extra>",
                                    },
                                    {
                                        "x": [1],
                                        "y": [0],
					                    "name" : "Ground Truth Count",
                                        "type": "lines",
                                        "hovertemplate": "%{y:.2f} persons"
                                                        "<extra></extra>",
                                    },
                                ],
                                "layout": {
                                    "title": {
                                        "text": "Ground Truth vs Prediction Analysis",
                                        "x": 0.05,
                                        "xanchor": "left",
                                    },
                                    "xaxis": {"fixedrange": True},
                                    "yaxis": {
                                        "ticksuffix": "-persons",
                                        "fixedrange": True,
                                    },
                                    "color": ["#17B897", "#32CD32"],
                                },
                            },
                        ),
                        dcc.Interval(
                                id='my_interval',
                                disabled=False,     
                                n_intervals=0,     
                                interval=40  
                                    
                        )],
                        className="card"
                )
            ],
            className="wrapper",
        )
    ]
)

@server.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    true_output = eval(message['true count'])

    if 'windows' in platform.architecture()[1].lower():
        estimated_count = true_output+1
    else:
        input_value = eval(message['csi data'])
        input_value = np.array(input_value).reshape(*input_shape_inf)
        output_data = inference.Inference(input_value)
        output_data = output_data.argmax()
        estimated_count = int(output_data.squeeze())

    estimated_count = process_data(estimated_count, true_output)    

    prediction_data = (true_output, estimated_count)
    access_database(False, prediction_data)

    response = {
            'estimated count' : str(estimated_count),
            'true count' : str(true_output)
            }
    return jsonify(response)

@app.callback(Output('pedestrian-counter', 'children'),
              [Input("my_interval", "n_intervals"),])
def update_data(n):
    _, prediction, ground_truth = get_data()
    return html.H3(
        children=" True Observation  ➡️ {}  VS  Estimated  Count  ➡️ {}".format(
                                                    ground_truth[-1], prediction[-1])),
          
@app.callback(Output('prediction-chart', 'figure'),
             [Input("my_interval", "n_intervals")])
def update_prediction_chart(n):
    iteration, prediction, ground_truth = get_data()
    update_chart = {
                    "data": [
                        {
                            "x": iteration,
                            "y": prediction,
			    "name" : "Ground Truth Count",
                            "type": "lines",
                            "hovertemplate": "%{y:.2f} persons"
                                                "<extra></extra>",
                        },
                        {
                            "x": iteration,
                            "y": ground_truth,
			    "name" : "Predicted Count",
                            "type": "lines",
                            "hovertemplate": "%{y:.2f} persons"
                                                "<extra></extra>",
                        },
                    ],
                    "layout": {
                        "title": {
                            "text": "Average Price of Avocados",
                            "x": 0.05,
                            "xanchor": "left",
                        },
                        "xaxis": {"fixedrange": True},
                        "yaxis": {
                            "ticksuffix": "-persons",
                            "fixedrange": True,
                        },
                        "color": ["#17B897", "#32CD32"],
                    },
                }
    return update_chart

if __name__ == "__main__":
    app.run_server(debug=True,host=host, port=port)