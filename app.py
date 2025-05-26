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

# ---------- GLOBALS ----------
country_name_map = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czech Republic",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain",
    "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia"
}

shocks = {
    2030: "Global recession",
    2036: "Policy withdrawal",
    2043: "Technological disruption"
}

# ---------- S3 JSON LOADERS ----------
def parse_json_content(content):
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
    future_projections = read_json_from_s3('artefacts/country_projections.json')
    return country_scores, project_keyphrases, feature_importance, future_projections

country_scores, project_keyphrases, feature_importance, future_projections = load_data()

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
    top_features = pd.concat([
        top_features,
        pd.DataFrame({'feature': ['Other'], 'importance': [other_importance]})
    ], ignore_index=True)

    fig = px.pie(top_features, values='importance', names='feature',
                 color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=24)
    fig.update_layout(
        showlegend=False,
        #legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, font=dict(size=16)),
        margin=dict(l=20, r=100, t=30, b=10),
        height=380,
        font=dict(size=16),
        hoverlabel=dict(font_size=20)
    )
    return fig

def create_country_contributions_chart(country_scores):
    country_scores = pd.DataFrame(country_scores)
    top_countries = country_scores.sort_values('Weighted_Score', ascending=False).head(20)

    # Add full country names using the dictionary
    top_countries['full_name'] = top_countries['country'].map(country_name_map).fillna(top_countries['country'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_countries['country'],
        y=top_countries['Weighted_Score'],
        name='Weighted Score',
        marker=dict(color=top_countries['color']),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "Normalized score: %{y:.2f}<br>" +
            "<i>This score has been normalized using GDP and project participation.</i><extra></extra>"
        ),
        customdata=top_countries[['full_name']]
    ))

    fig.update_layout(
        font=dict(size=16),
        xaxis=dict(title='Country', tickfont=dict(size=16)),
        yaxis=dict(title='Weighted Score', tickfont=dict(size=16)),
        margin=dict(l=20, r=20, t=30, b=50),
        height=380,
        hoverlabel=dict(font_size=16)
    )
    return fig


# def create_performance_comparison_chart(country_code="MT", what_if_data=None):
#     if not what_if_data or country_code not in what_if_data:
#         countries = ['Country A', 'Country B', 'Country C', 'Country D']
#         current = [80, 75, 85, 70]
#         potential = [95, 90, 100, 85]
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=countries, y=current, mode='lines', name='Current Performance',
#                                  line=dict(color='teal', width=3)))
#         fig.add_trace(go.Scatter(x=countries, y=potential, mode='lines', name='Potential with New Metrics',
#                                  line=dict(color='orange', width=3)))
#         fig.update_layout(
#             yaxis_title='Performance Score',
#             legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
#             margin=dict(l=20, r=20, t=30, b=10),
#             height=380,
#             font=dict(size=16)
#         )
#         return fig

#     country_data = what_if_data[country_code]
#     years = country_data["years"]
#     current_scores = country_data["score_current"]
#     improved_scores = country_data["score_improved"]

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=years, y=current_scores, mode='lines+markers',
#                              name='Current Performance', line=dict(color='teal', width=3)))
#     fig.add_trace(go.Scatter(x=years, y=improved_scores, mode='lines+markers',
#                              name='Potential with New Metrics', line=dict(color='orange', width=3)))
#     fig.update_layout(
#         xaxis_title="Year",
#         yaxis_title="Performance Score",
#         legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
#         margin=dict(l=20, r=20, t=30, b=10),
#         height=380,
#         font=dict(size=14)
#     )
#     return fig

def create_performance_comparison_chart(country_code="MT", what_if_data=None):

    if not what_if_data or country_code not in what_if_data:
        countries = ['Country A', 'Country B', 'Country C', 'Country D']
        current = [80, 75, 85, 70]
        potential = [95, 90, 100, 85]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=countries, y=current, mode='lines', name='Current Performance',
                                 line=dict(color='teal', width=3)))
        fig.add_trace(go.Scatter(x=countries, y=potential, mode='lines', name='Potential with New Metrics',
                                 line=dict(color='orange', width=3)))
        fig.update_layout(
            yaxis_title='Performance Score',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=30, b=10),
            height=380,
            font=dict(size=16)
        )
        return fig

    country_data = what_if_data[country_code]
    years = country_data["years"]
    current_scores = country_data["score_current"]
    improved_scores = country_data["score_improved"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=current_scores, mode='lines+markers',
                             name='Current Performance', line=dict(color='teal', width=3)))
    fig.add_trace(go.Scatter(x=years, y=improved_scores, mode='lines+markers',
                             name='Potential with New Metrics', line=dict(color='orange', width=3)))

    # Add built-in shocks
    for yr, label in shocks.items():
        if years[0] <= yr <= years[-1]:
            fig.add_vline(x=yr, line_dash="dot", line_color="gray", opacity=0.6)
            fig.add_annotation(
                x=yr,
                y=max(max(current_scores), max(improved_scores)),
                text=label,
                showarrow=False,
                yshift=10,
                font=dict(size=16, color="grey")
            )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Performance Score",
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(size=16)
    )

    return fig

# ---------- DASH APP ----------
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=DynaPuff&display=swap"
])
server = app.server
load_figure_template('darkly')

app.layout = html.Div([
    html.H1("Impact on EU Climate Neutrality Goal", style={
        'color': '#51A356', 'textAlign': 'center',
        'margin': '0.5vh 0',
        'fontFamily': 'DynaPuff, cursive',
        'letterSpacing': '1px',
        'fontSize': '4vh'
    }),

    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Countries by contribution", className="text-center", style={
                                'fontFamily': 'DynaPuff, cursive',
                                'fontSize': '2.5vh','marginBottom': '0.2vh'
                            }),
                            dcc.Graph(
                                figure=create_country_contributions_chart(country_scores),
                                config={"displayModeBar": False},
                                style={"height": "35vh"}
                            )
                        ], className="bg-dark text-light shadow p-3 rounded-4")
                    ], className="mb-3")
                ]),

                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Most frequent focus areas:", className="text-center", style={
                                'fontFamily': 'DynaPuff, cursive',
                                'fontSize': '2.5vh'
                            }),
                            html.Img(src=wordcloud_img, style={
                                'width': '100%',
                                'height': '30vh',
                                'objectFit': 'cover'
                            })
                        ], className="bg-dark text-light shadow p-3 rounded-4 h-100")
                    ], width=6),

                    dbc.Col([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Most important features for success of a project", className="text-end", style={
                                        'fontFamily': 'DynaPuff, cursive',
                                        'fontSize': '2.5vh'
                                    })
                                ], width=5, className="d-flex align-items-center"),
                                dbc.Col([
                                    dcc.Graph(
                                        figure=create_feature_importance_chart(),
                                        config={"displayModeBar": False},
                                        style={"height": "33vh"}
                                    )
                                ], width=7)
                            ],align="center")
                        ], className="bg-dark text-light shadow p-3 rounded-4 h-100")
                    ], width=6)
                ])
            ], style={"height": "85vh", "overflow": "hidden", "padding": "1vh"})
        ],
        style={
            'backgroundColor': '#2a2a2a',
            'color': 'white',
            'border': 'none',
            'fontSize': '2.1vh',
            'letterSpacing': '0.5px',
            'fontFamily': 'DynaPuff, cursive',
            'padding': '6px 12px'
        },
        selected_style={
            'backgroundColor': '#2a2a2a',
            'color': '#51A356',
            'fontWeight': 'bold',
            'border': 'none',
            'fontSize': '2.1vh',
            'letterSpacing': '0.5px',
            'padding': '6px 12px'
        }),

        dcc.Tab(label="Contribution improvement report", children=[
            html.Div([

                # Unified Row 1
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                figure=create_performance_comparison_chart("MT", future_projections),
                                config={"displayModeBar": False},
                                style={"height": "36vh"}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.P(future_projections["MT"]["explanation"], style={
                                    'fontSize': '2.5vh',
                                    'fontFamily': 'DynaPuff, cursive',
                                    'marginBottom': '1vh'
                                })
                            ], style={
                                "display": "flex",
                                "flexDirection": "column",
                                "justifyContent": "center",
                                "height": "100%"
                            })
                        ], width=6)
                    ])
                ], className="bg-dark text-light shadow p-3 rounded-4 mb-3"),

                # Unified Row 2
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.P(future_projections["HR"]["explanation"], style={
                                    'fontSize': '2.5vh',
                                    'fontFamily': 'DynaPuff, cursive',
                                    'marginBottom': '1vh'
                                })
                            ],  style={
                                "display": "flex",
                                "flexDirection": "column",
                                "justifyContent": "center",
                                "height": "100%"
                            })
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(
                                figure=create_performance_comparison_chart("HR", future_projections),
                                config={"displayModeBar": False},
                                style={"height": "36vh"}
                            )
                        ], width=6)
                    ])
                ], className="bg-dark text-light shadow p-3 rounded-4")

            ], style={"padding": "1vh"})
        ],
        style={
            'backgroundColor': '#2a2a2a',
            'color': 'white',
            'border': 'none',
            'fontSize': '2.1vh',
            'letterSpacing': '0.5px',
            'fontFamily': 'DynaPuff, cursive',
            'padding': '6px 12px'
        },
        selected_style={
            'backgroundColor': '#2a2a2a',
            'color': '#51A356',
            'fontWeight': 'bold',
            'border': 'none',
            'fontSize': '2.1vh',
            'letterSpacing': '0.5px',
            'padding': '6px 12px'
        })


    ],
    style={
        'backgroundColor': '#2a2a2a',
        'borderRadius': '10px',
        'padding': '0.2vh',
        'color': 'white',
        'border': 'none',
        'marginBottom': '0px',
        'fontFamily': 'DynaPuff, cursive'
    }),

], style={
    "backgroundColor": "#121212",
    "padding": "1vh",
    "minHeight": "100vh",
    "width": "100%",
    "overflow": "hidden"
})


# ---------- RUN ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
