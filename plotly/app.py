import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

FILE_PATH = "data_synthetic_.csv"
df = pd.read_csv(FILE_PATH)

df = df.drop_duplicates()

missing_cols = ["Age", "Income Level", "Premium Amount", "Credit Score"]
for col in missing_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], format="mixed", errors="coerce")
df["Policy Renewal Date"] = pd.to_datetime(df["Policy Renewal Date"], format="mixed", errors="coerce")

df["Policy Start Year"] = df["Policy Start Date"].dt.year
df["Policy Tenure Days"] = (df["Policy Renewal Date"] - df["Policy Start Date"]).dt.days

if "Customer ID" in df.columns:
    df = df.drop(columns=["Customer ID"])

df = df.dropna(subset=["Policy Start Year"])
df["Policy Start Year"] = df["Policy Start Year"].astype(int)

def cap_outliers_iqr(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data[col] = np.where(data[col] < lower, lower,
                         np.where(data[col] > upper, upper, data[col]))

for col in ["Premium Amount", "Income Level", "Credit Score"]:
    if col in df.columns:
        cap_outliers_iqr(df, col)

segment_colors = {
    "Segment1": "#4C78A8",
    "Segment2": "#F58518",
    "Segment3": "#54A24B",
    "Segment4": "#E45756",
    "Segment5": "#B279A2"
}

segment_options = sorted(df["Segmentation Group"].dropna().unique())
policy_options = sorted(df["Policy Type"].dropna().unique())
risk_options = sorted(df["Risk Profile"].dropna().unique())

min_year = int(df["Policy Start Year"].min())
max_year = int(df["Policy Start Year"].max())

app = Dash(__name__)
app.title = "Customer Segmentation Dashboard"

BACKGROUND = "#F4F7FB"
CARD = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#6B7280"
HEADER = "#111827"
ACCENT = "#2563EB"
BORDER = "#E5E7EB"

card_style = {
    "backgroundColor": CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "18px",
    "padding": "16px",
    "boxShadow": "0 6px 20px rgba(17,24,39,0.06)"
}

graph_card_style = {
    "backgroundColor": CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "18px",
    "padding": "10px",
    "boxShadow": "0 6px 20px rgba(17,24,39,0.06)"
}

app.layout = html.Div(
    style={
        "backgroundColor": BACKGROUND,
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "color": TEXT
    },
    children=[
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {HEADER}, #1D4ED8)",
                "color": "white",
                "borderRadius": "22px",
                "padding": "24px",
                "marginBottom": "22px",
                "boxShadow": "0 10px 30px rgba(37,99,235,0.20)"
            },
            children=[
                html.H1("Customer Segmentation Dashboard", style={"margin": "0 0 8px 0"}),
                html.P(
                    "Interactive view of segment distribution, premium behavior, credit profile, policy mix, risk-based premium patterns, and time trends.",
                    style={"margin": 0, "opacity": 0.92}
                )
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "16px",
                "marginBottom": "18px"
            },
            children=[
                html.Div([
                    html.Label("Segmentation Group", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                    dcc.Dropdown(
                        id="segment_filter",
                        options=[{"label": s, "value": s} for s in segment_options],
                        value=segment_options,
                        multi=True,
                        placeholder="Select segments"
                    )
                ], style=card_style),

                html.Div([
                    html.Label("Policy Type", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                    dcc.Dropdown(
                        id="policy_filter",
                        options=[{"label": p, "value": p} for p in policy_options],
                        value=policy_options,
                        multi=True,
                        placeholder="Select policy types"
                    )
                ], style=card_style),

                html.Div([
                    html.Label("Risk Profile", style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"}),
                    dcc.Dropdown(
                        id="risk_filter",
                        options=[{"label": r, "value": r} for r in risk_options],
                        value=risk_options,
                        multi=True,
                        placeholder="Select risk profiles"
                    )
                ], style=card_style),
            ]
        ),

        html.Div(
            style={**card_style, "marginBottom": "18px"},
            children=[
                html.Label("Policy Start Year Range", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                dcc.RangeSlider(
                    id="year_filter",
                    min=min_year,
                    max=max_year,
                    step=1,
                    value=[min_year, max_year],
                    marks={year: str(year) for year in range(min_year, max_year + 1)}
                )
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "16px",
                "marginBottom": "18px"
            },
            children=[
                html.Div(id="kpi_total", style=card_style),
                html.Div(id="kpi_premium", style=card_style),
                html.Div(id="kpi_credit", style=card_style),
                html.Div(id="kpi_tenure", style=card_style),
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px"
            },
            children=[
                html.Div(dcc.Graph(id="segment_count_chart"), style=graph_card_style),
                html.Div(dcc.Graph(id="policy_donut_chart"), style=graph_card_style),
                html.Div(dcc.Graph(id="premium_box_chart"), style=graph_card_style),
                html.Div(dcc.Graph(id="credit_hist_chart"), style=graph_card_style),
                html.Div(dcc.Graph(id="premium_heatmap"), style=graph_card_style),
                html.Div(dcc.Graph(id="year_line_chart"), style=graph_card_style),
            ]
        )
    ]
)

@app.callback(
    Output("kpi_total", "children"),
    Output("kpi_premium", "children"),
    Output("kpi_credit", "children"),
    Output("kpi_tenure", "children"),
    Output("segment_count_chart", "figure"),
    Output("policy_donut_chart", "figure"),
    Output("premium_box_chart", "figure"),
    Output("credit_hist_chart", "figure"),
    Output("premium_heatmap", "figure"),
    Output("year_line_chart", "figure"),
    Input("segment_filter", "value"),
    Input("policy_filter", "value"),
    Input("risk_filter", "value"),
    Input("year_filter", "value"),
)
def update_dashboard(selected_segments, selected_policies, selected_risks, selected_years):
    filtered_df = df[
        df["Segmentation Group"].isin(selected_segments) &
        df["Policy Type"].isin(selected_policies) &
        df["Risk Profile"].isin(selected_risks) &
        df["Policy Start Year"].between(selected_years[0], selected_years[1])
    ].copy()

    if filtered_df.empty:
        empty_fig = px.scatter(title="No data available for selected filters")
        empty_card = html.Div([
            html.H3("0", style={"margin": 0}),
            html.P("No matching records", style={"margin": "6px 0 0 0", "color": MUTED})
        ])
        return empty_card, empty_card, empty_card, empty_card, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    kpi_total = html.Div([
        html.Div("Filtered Customers", style={"color": MUTED, "fontSize": "14px"}),
        html.H2(f"{len(filtered_df):,}", style={"margin": "8px 0 0 0", "color": TEXT})
    ])

    kpi_premium = html.Div([
        html.Div("Average Premium", style={"color": MUTED, "fontSize": "14px"}),
        html.H2(f"{filtered_df['Premium Amount'].mean():,.2f}", style={"margin": "8px 0 0 0", "color": TEXT})
    ])

    kpi_credit = html.Div([
        html.Div("Average Credit Score", style={"color": MUTED, "fontSize": "14px"}),
        html.H2(f"{filtered_df['Credit Score'].mean():,.2f}", style={"margin": "8px 0 0 0", "color": TEXT})
    ])

    kpi_tenure = html.Div([
        html.Div("Average Policy Tenure (Days)", style={"color": MUTED, "fontSize": "14px"}),
        html.H2(f"{filtered_df['Policy Tenure Days'].mean():,.0f}", style={"margin": "8px 0 0 0", "color": TEXT})
    ])

    common_layout = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT),
        margin=dict(l=40, r=20, t=60, b=40),
        title_font=dict(size=18),
    )

    seg_counts = filtered_df["Segmentation Group"].value_counts().reset_index()
    seg_counts.columns = ["Segmentation Group", "Count"]

    fig1 = px.bar(
        seg_counts,
        x="Segmentation Group",
        y="Count",
        color="Segmentation Group",
        color_discrete_map=segment_colors,
        text="Count",
        title="Customer Count by Segment"
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(showlegend=False, **common_layout)
    fig1.update_xaxes(title="Segmentation Group")
    fig1.update_yaxes(title="Count")

    policy_counts = filtered_df["Policy Type"].value_counts().reset_index()
    policy_counts.columns = ["Policy Type", "Count"]

    fig2 = px.pie(
        policy_counts,
        names="Policy Type",
        values="Count",
        hole=0.55,
        title="Policy Type Share"
    )
    fig2.update_traces(textinfo="percent+label", pull=[0.02] * len(policy_counts))
    fig2.update_layout(**common_layout)

    fig3 = px.box(
        filtered_df,
        x="Segmentation Group",
        y="Premium Amount",
        color="Segmentation Group",
        color_discrete_map=segment_colors,
        points="outliers",
        title="Premium Amount by Segment"
    )
    fig3.update_layout(showlegend=False, **common_layout)
    fig3.update_xaxes(title="Segmentation Group")
    fig3.update_yaxes(title="Premium Amount")

    fig4 = px.histogram(
        filtered_df,
        x="Credit Score",
        color="Segmentation Group",
        color_discrete_map=segment_colors,
        nbins=25,
        barmode="overlay",
        opacity=0.65,
        title="Credit Score Distribution by Segment"
    )
    fig4.update_layout(**common_layout)
    fig4.update_xaxes(title="Credit Score")
    fig4.update_yaxes(title="Frequency")

    heatmap_data = filtered_df.pivot_table(
        values="Premium Amount",
        index="Risk Profile",
        columns="Segmentation Group",
        aggfunc="mean"
    )

    fig5 = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Average Premium by Risk Profile and Segment"
    )
    fig5.update_layout(**common_layout)
    fig5.update_xaxes(title="Segmentation Group")
    fig5.update_yaxes(title="Risk Profile")

    year_counts = (
        filtered_df.groupby("Policy Start Year")
        .size()
        .reset_index(name="Count")
        .sort_values("Policy Start Year")
    )

    fig6 = px.line(
        year_counts,
        x="Policy Start Year",
        y="Count",
        markers=True,
        title="Policies Started Over Time"
    )
    fig6.update_traces(line=dict(width=4, color=ACCENT), marker=dict(size=9, color=ACCENT))
    fig6.update_layout(**common_layout)
    fig6.update_xaxes(title="Policy Start Year")
    fig6.update_yaxes(title="Number of Policies")

    return kpi_total, kpi_premium, kpi_credit, kpi_tenure, fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == "__main__":
    app.run(debug=True)