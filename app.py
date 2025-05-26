import os
import io
import json
import base64
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import plotly.graph_objects as go

from kedro_datasets.json import JSONDataset
from kedro.io import DataCatalog

# Load environment variables
load_dotenv()

# ---------- S3 JSON LOADERS ----------
def parse_json_content(content):
    """Try to parse content into a DataFrame or fallback to raw JSON"""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        elif isinstance(parsed, dict):
            if all(isinstance(v, list) for v in parsed.values()):
                lengths = [len(v) for v in parsed.values()]
                if len(set(lengths)) == 1:
                    return pd.DataFrame(parsed)
                else:
                    print("Returning flattened list of keyphrases")
                    return pd.DataFrame([
                        {"project_id": k, "term": phrase}
                        for k, phrases in parsed.items()
                        for phrase in phrases
                    ])
            else:
                return parsed
        else:
            return parsed
    except Exception as e:
        print(f"Failed to parse content: {e}")
        return None

def read_json_from_s3(key):
    bucket_name = 'climate-upload'
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        response = s3.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read().decode('utf-8')
        return parse_json_content(content)
    except Exception as e:
        print(f"Error loading {key} from bucket {bucket_name}: {e}")
        return None

def load_data():
    country_scores = read_json_from_s3('artefacts/country_scores.json')
    project_keyphrases = read_json_from_s3('artefacts/project_keyphrases.json')
    feature_importance = read_json_from_s3('artefacts/feature_importances.json')
    what_if_results = read_json_from_s3('artefacts/what_if_results.json')
    return country_scores, project_keyphrases, feature_importance, what_if_results

country_scores, project_keyphrases, feature_importance, what_if_results = load_data()

# ---------- WORDCLOUD ----------
def flatten_keyphrases_to_counts(nested_dict):
    term_counts = nested_dict['term'].value_counts().reset_index()
    term_counts.columns = ['term', 'count']
    return term_counts

def generate_wordcloud_base64(df, term_col="term", count_col="count"):
    try:
        word_dict = dict(zip(df[term_col], df[count_col]))
        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color='#2a2a2a',
            colormap='Set2',
            max_words=100
        ).generate_from_frequencies(word_dict)

        buf = io.BytesIO()
        plt.figure(figsize=(16, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#2a2a2a')
        plt.close()
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

wordcloud_img = generate_wordcloud_base64(flatten_keyphrases_to_counts(project_keyphrases))

# ---------- CHARTS ----------
def create_feature_importance_chart():
    top_features = feature_importance.head(5).copy()
    other_importance = feature_importance.iloc[5:]['importance'].sum()
    top_features = pd.concat(
        [top_features, pd.DataFrame({'feature': ['Other'], 'importance': [other_importance]})],
        ignore_index=True
    )

    fig = px.pie(top_features, values='importance', names='feature',
                 color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=70)
    fig.update_layout(
        legend=dict(
            orientation="v", yanchor="middle", y=0.5,
            xanchor="center", x=1.05, font=dict(size=45)
        ),
        margin=dict(l=20, r=150, t=40, b=20),
        height=400,
        font=dict(size=14),
        hoverlabel=dict(font_size=30)
    )
    return fig

def create_country_contributions_chart(country_scores):
    top_countries = country_scores.sort_values('Weighted_Score', ascending=False).head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_countries['country'],
        y=top_countries['Weighted_Score'],
        name='Weighted Score',
        marker=dict(color=top_countries['color']),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        font=dict(size=14),
        xaxis=dict(title='Country', tickfont=dict(size=30)),
        yaxis=dict(title='Weighted Score', tickfont=dict(size=30)),
        legend=dict(font=dict(size=20)),
        margin=dict(l=20, r=20, t=40, b=100),
        height=500,
        hoverlabel=dict(font_size=30)
    )
    return fig

def create_performance_comparison_chart():
    if not what_if_results:
        countries = ['Country A', 'Country B', 'Country C', 'Country D']
        current = [80, 75, 85, 70]
        potential = [95, 90, 100, 85]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(countries))), y=current, mode='lines',
                                 name='Current Performance', line=dict(color='teal', width=3)))
        fig.add_trace(go.Scatter(x=list(range(len(countries))), y=potential, mode='lines',
                                 name='Potential with New Metrics', line=dict(color='orange', width=3)))
        fig.update_layout(
            title='Performance Comparison: Current vs. Potential with New Metrics',
            xaxis=dict(tickmode='array', tickvals=list(range(len(countries))), ticktext=countries),
            yaxis_title='Performance Score',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
    else:
        fig = go.Figure()  # Placeholder
    return fig

# ---------- DASH SETUP ----------
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=DynaPuff&display=swap"
])
server = app.server
load_figure_template('darkly')

# ---------- DASH LAYOUT ----------
app.layout = html.Div([
    html.H1("Impact on EU Climate Neutrality Goal", style={
        'color': 'limegreen', 'textAlign': 'center',
        'marginTop': '5px', 'marginBottom': '10px',
        'fontFamily': 'DynaPuff, cursive', 'letterSpacing': '1px'
    }),

    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Countries by contribution", className="text-center", style={'fontFamily': 'DynaPuff, cursive'}),
                            dcc.Graph(figure=create_country_contributions_chart(country_scores), config={"displayModeBar": False}, style={"height": "40vh"})
                        ], className="bg-dark text-light shadow p-3 rounded-4", style={"backgroundColor": "#2a2a2a"})
                    ], width=12)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Most frequent focus areas:", className="text-center", style={'fontFamily': 'DynaPuff, cursive'}),
                            html.Img(src=wordcloud_img, style={'width': '100%', 'maxHeight': '35vh', 'objectFit': 'contain'})
                        ], className="bg-dark text-light shadow p-3 rounded-4 h-100", style={"backgroundColor": "#2a2a2a", "height": "100%"})
                    ], width=6),

                    dbc.Col([
                        html.Div([
                            html.H4("Most important features for success of a project", className="text-center", style={'fontFamily': 'DynaPuff, cursive'}),
                            dcc.Graph(figure=create_feature_importance_chart(), config={"displayModeBar": False}, style={"height": "35vh"})
                        ], className="bg-dark text-light shadow p-3 rounded-4 h-100", style={"backgroundColor": "#2a2a2a", "height": "100%"})
                    ], width=6)
                ], className="gy-2")
            ], fluid=True, style={"height": "85vh"})
        ],
        style={'backgroundColor': '#2a2a2a', 'color': 'white', 'border': 'none',
               'fontSize': '1.1rem', 'letterSpacing': '0.5px', 'fontFamily': 'DynaPuff, cursive'},
        selected_style={'backgroundColor': '#2a2a2a', 'color': '#28a745', 'fontWeight': 'bold',
                        'border': 'none', 'fontSize': '1.1rem', 'letterSpacing': '0.5px'}),

        dcc.Tab(label="Contribution improvement report", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Graph(figure=create_performance_comparison_chart(), config={"displayModeBar": False}, style={"height": "40vh"})
                        ], className="bg-dark text-light shadow p-3 rounded-4", style={"backgroundColor": "#2a2a2a"})
                    ], width=6),

                    dbc.Col([
                        html.Div([
                            html.P("Over the past three years, Country A and Country B have persistently underperformed...", style={'fontSize': '14px', 'fontFamily': 'DynaPuff, cursive'}),
                            html.Br(),
                            html.P("Our analysis shows that by improving their COS through more diverse partnerships...", style={'fontSize': '14px', 'fontFamily': 'DynaPuff, cursive'})
                        ], className="bg-dark text-light shadow p-3 rounded-4", style={"backgroundColor": "#2a2a2a"})
                    ], width=6)
                ])
            ], fluid=True, style={"height": "85vh"})
        ],
        style={'backgroundColor': '#2a2a2a', 'color': 'white', 'border': 'none',
               'fontSize': '1.1rem', 'letterSpacing': '0.5px', 'fontFamily': 'DynaPuff, cursive'},
        selected_style={'backgroundColor': '#2a2a2a', 'color': '#28a745', 'fontWeight': 'bold',
                        'border': 'none', 'fontSize': '1.1rem', 'letterSpacing': '0.5px'})
    ],
    style={
        'backgroundColor': '#2a2a2a',
        'borderRadius': '10px',
        'padding': '0.5rem',
        'color': 'white',
        'border': 'none',
        'marginBottom': '10px',
        'fontFamily': 'DynaPuff, cursive'
    }),
], style={
    "backgroundColor": "#121212",
    "padding": "1rem",
    "height": "100vh",
    "overflow": "hidden",
    "display": "flex",
    "flexDirection": "column"
})

# ---------- RUN ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

