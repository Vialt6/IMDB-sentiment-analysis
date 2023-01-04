import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns

import jobs.imdb_sentiment_analysis

app = dash.Dash()

df = jobs.prova.df.toPandas()

fig = px.bar(
    df,
    x="sentiment",
    size_max=60
)

app.layout=html.Div(children = [
    html.H1(children='IMDB Dashboard'),
    dcc.Graph(id="provaprova", figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)