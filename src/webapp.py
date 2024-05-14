import dash
from dash import dcc, html
from GRUmodel import prepare_plot_data, stock_price_plot, daily_returns_plot, prepare_data, sudden_drop, calculate_growth, testY, gru_predictions

app = dash.Dash(__name__)

# Generate some example data
df = prepare_data()
head_data = df.head()
data = prepare_plot_data(df)
stock_price = stock_price_plot(data)
daily_returns = daily_returns_plot(data)

def check_sudden_drop():
    message = sudden_drop()
    if "Alert" in message:
        return html.Div(message, style={'background-color': 'red', 'color': 'black', 'padding': '10px'})
    else:
        return None

def format_growth_table():
    day_growth, month_growth, year_growth = calculate_growth()
    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Period"),
            html.Th("Growth (%)")
        ])),
        html.Tbody([
            html.Tr([
                html.Td("Daily"),
                html.Td(f"{day_growth:.2f}")
            ]),
            html.Tr([
                html.Td("Monthly"),
                html.Td(f"{month_growth:.2f}")
            ]),
            html.Tr([
                html.Td("Yearly"),
                html.Td(f"{year_growth:.2f}")
            ])
        ])
    ], style={'margin': '20px 0', 'border': '1px solid white', 'border-collapse': 'collapse', 'color': 'white', 'font-size': '1.5em', 'textAlign': 'left'})
    return table

# Define app layout
app.layout = html.Div(style={'backgroundColor': '#537d90', 'padding': '2rem', 'border': '2rem solid white'},
children=[
    html.H1(children='Stock Prediction Model',
            style={'textAlign': 'center', 'color': 'white', 'textDecoration': 'underline', 'fontWeight': 'bold', 'fontSize': '2.5em', 'marginBottom': '20px'}),
    html.H3(children='Head of Data',
            style={'textAlign': 'center', 'color': 'white', 'fontSize': '1.5em', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in head_data.columns], style={'color': 'white', 'width': '200px'})),
                html.Tbody([
                    html.Tr([
                        html.Td(head_data.iloc[i][col], style={'border': '1px solid white', 'color': 'white', 'padding': '8px', 'font-size': '1.2em'}) for col in head_data.columns
                    ]) for i in range(len(head_data))
                ])
            ], style={'margin': '0 auto', 'color': 'white', 'marginBottom': '20px'})
        ]),
    ]),
    html.H1("Growth Analysis", style={'textAlign': 'left', 'color': 'white', 'marginBottom': '20px'}),
    format_growth_table(),
    dcc.Graph(id='stock-price-plot', figure=stock_price, style={'marginBottom': '20px'}),
    dcc.Graph(id='daily-returns-plot', figure=daily_returns),
    html.H2("Model Plot", style={'textAlign': 'left', 'color': 'white', 'marginBottom': '20px'}),
    dcc.Graph(
        id='actual-vs-predicted',
        figure={
            'data': [
                {'x': list(range(len(testY))), 'y': testY, 'type': 'line', 'name': 'Actual'},
                {'x': list(range(len(gru_predictions))), 'y': gru_predictions.flatten(), 'type': 'line', 'name': 'Predicted'}
            ],
            'layout': {
                'title': 'Actual vs Predicted',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Value'},
                'hovermode': 'closest'
            }
    }
)
])

if __name__ == '__main__':
    app.run_server(debug=True)
