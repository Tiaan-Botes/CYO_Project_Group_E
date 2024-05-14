import dash
import plotly.graph_objs as go
import numpy as np
from dash import dcc, html
import pandas as pd

# Generate some example data
x = np.linspace(0, 10, 100)
y = np.sin(x)
# Read the snapshot data
data_url = 'https://raw.githubusercontent.com/Tiaan-Botes/CYO_Project_Group_E/52663ba4e6833e232f1bbd7a0ab48edb23f52b91/data/data.csv'
df = pd.read_csv(data_url)
df.drop(columns=['Unnamed: 0'])
df = df.round(2)
head_data = df.head()
app = dash.Dash(__name__)


# Define app layout
app.layout = html.Div(style={'backgroundColor': '#537d90', 'padding': '2rem'},
                      children=[
                          html.H1(children='Stock Prediction Model',
                                  style={'textAlign': 'center', 'color': 'white', 'textDecoration': 'underline', 'fontWeight': 'bold', 'fontSize': '2.5em'}),
                          html.H3(children='Head of Data',
                                  style={'textAlign': 'center', 'color': 'white', 'fontSize': '1.5em'}),
                          html.Div([
                              html.Table([
                                  html.Thead(html.Tr([html.Th(col) for col in head_data.columns], style={'color': 'white', 'width': '20%'})),
                                  html.Tbody([
                                      html.Tr([
                                          html.Td(head_data.iloc[i][col]) for col in head_data.columns
                                      ], style={'border': '1px solid white', 'color': 'white', 'width': '20%'}) for i in range(len(head_data))
                                  ])
                              ], style={'margin': '0 auto', 'color': 'white'})
                          ]),
                          dcc.Graph(
                              id='example-graph',
                              figure={
                                  'data': [
                                      {'x': x, 'y': y, 'type': 'scatter', 'name': 'sin(x)'},
                                  ],
                                  'layout': go.Layout(
                                      title='Stock Prediction',
                                      xaxis={'title': 'Time'},
                                      yaxis={'title': 'Value'},
                                      margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                                      legend={'x': 0, 'y': 1},
                                      plot_bgcolor='#ffffff',
                                      paper_bgcolor='#ffffff',
                                      width=1500,
                                  )
                              }
                          )

                      ])

if __name__ == '__main__':
    app.run_server(debug=True)