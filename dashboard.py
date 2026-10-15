"""
dashboard.py  -  Interactive Dash dashboard for the AcademicNotesAI fused dataset
Run:   python dashboard.py     then open  http://127.0.0.1:8050
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from data_prep import load_data, WEEKDAYS

DF = load_data()

# ------------------------------------------------------------------ palette
ACT_COLORS = {"AI Chat": "#4F46E5", "YouTube Search": "#EF4444", "Image Study": "#F59E0B",
              "Quiz": "#10B981", "Saved Note": "#06B6D4", "Sign-up": "#94A3B8"}
TOPIC_PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
                 "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#94A3B8"]
DIFF_COLORS = {"Easy": "#10B981", "Medium": "#F59E0B", "Hard": "#EF4444"}
SEQ = [[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#312E81"]]
FONT = dict(family="Inter, Segoe UI, Arial", size=12, color="#1E293B")
TOPIC_COLORS = {t: TOPIC_PALETTE[i % 10] for i, t in enumerate(sorted(DF.topic_group.unique()))}


def style(fig, title, h=340, legend=True):
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=15)),
                      template="plotly_white", height=h, font=FONT,
                      margin=dict(l=45, r=20, t=55, b=40), showlegend=legend,
                      paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)))
    return fig


def empty(title, h=340, msg="No data for the selected filters"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#94A3B8"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style(fig, title, h, False)


# ------------------------------------------------------------------ charts
def c1_donut(d):
    t = "1 · Activity Mix (Donut)"
    if d.empty: return empty(t)
    g = d.activity.value_counts().reset_index()
    fig = px.pie(g, names="activity", values="count", hole=.58, color="activity",
                 color_discrete_map=ACT_COLORS)
    fig.update_traces(textinfo="percent+label", textposition="outside", sort=False)
    fig.add_annotation(text=f"<b>{len(d)}</b><br>records", showarrow=False, font=dict(size=18))
    return style(fig, t, legend=False)


def c2_area(d):
    t = "2 · Weekly Activity Trend (Stacked Area)"
    if d.empty: return empty(t)
    g = d.groupby(["week", "activity"]).size().reset_index(name="n")
    fig = px.area(g, x="week", y="n", color="activity", color_discrete_map=ACT_COLORS,
                  line_shape="spline")
    fig.update_layout(xaxis_title="Week starting", yaxis_title="Records")
    return style(fig, t)


def c3_heat(d):
    t = "3 · Weekday × Activity (Heatmap)"
    if d.empty: return empty(t)
    p = (d.pivot_table(index="activity", columns="weekday", values="record_id",
                       aggfunc="count", observed=False).reindex(columns=WEEKDAYS).fillna(0))
    fig = go.Figure(go.Heatmap(z=p.values, x=p.columns.astype(str), y=p.index, colorscale=SEQ,
                               text=p.values.astype(int), texttemplate="%{text}",
                               hovertemplate="%{y} on %{x}: %{z}<extra></extra>"))
    return style(fig, t, legend=False)


def c4_users(d):
    t = "4 · Records per User (Stacked Bar)"
    if d.empty: return empty(t)
    g = d.groupby(["user_short", "activity"]).size().reset_index(name="n")
    order = g.groupby("user_short").n.sum().sort_values(ascending=False).index
    fig = px.bar(g, x="user_short", y="n", color="activity", color_discrete_map=ACT_COLORS,
                 category_orders={"user_short": list(order)})
    fig.update_layout(xaxis_title="", yaxis_title="Records")
    return style(fig, t)


def c5_topics(d):
    t = "5 · Most Studied Topics (Horizontal Bar)"
    dd = d[d.topic_group != "General / Unspecified"]
    if dd.empty: return empty(t)
    g = dd.topic_group.value_counts().sort_values().reset_index()
    fig = px.bar(g, x="count", y="topic_group", orientation="h", color="topic_group",
                 color_discrete_map=TOPIC_COLORS, text="count")
    fig.update_layout(xaxis_title="Records", yaxis_title="")
    fig.update_traces(textposition="outside")
    return style(fig, t, legend=False)


def c6_sun(d):
    t = "6 · Activity → Topic Breakdown (Sunburst)"
    dd = d[d.topic_group != "General / Unspecified"]
    if dd.empty: return empty(t)
    g = dd.groupby(["activity", "topic_group"]).size().reset_index(name="n")
    fig = px.sunburst(g, path=["activity", "topic_group"], values="n", color="activity",
                      color_discrete_map=ACT_COLORS)
    fig.update_traces(insidetextorientation="radial")
    return style(fig, t, legend=False)


def c7_box(d):
    t = "7 · Prompt Length by Activity (Box Plot)"
    dd = d[d.prompt_words > 0]
    if dd.empty: return empty(t)
    fig = px.box(dd, x="activity", y="prompt_words", color="activity", points="all",
                 color_discrete_map=ACT_COLORS, hover_data=["prompt_text"])
    fig.update_layout(xaxis_title="", yaxis_title="Words in prompt")
    return style(fig, t, legend=False)


def c8_polar(d):
    t = "8 · Time-of-Day Usage (Polar Bar Clock)"
    dd = d[d.hour.notna()]
    if dd.empty: return empty(t)
    g = dd.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    g.columns = ["hour", "n"]
    fig = go.Figure(go.Barpolar(r=g.n, theta=g.hour * 15, width=14, marker_color=g.n,
                                marker_colorscale=SEQ, hovertext=g.hour.astype(str) + ":00",
                                hovertemplate="%{hovertext}  →  %{r} records<extra></extra>"))
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90,
                      tickvals=list(range(0, 360, 45)), ticktext=[f"{h}:00" for h in range(0, 24, 3)]),
                      radialaxis=dict(showticklabels=True, gridcolor="#E2E8F0")))
    return style(fig, t, legend=False)


def c9_scatter(d):
    t = "9 · Question vs Answer Length – Saved Notes (Scatter + Trend)"
    dd = d[(d.is_saved_note == 1) & (d.answer_length > 0)]
    if dd.empty: return empty(t)
    fig = px.scatter(dd, x="question_length", y="answer_length", color="topic_group",
                     color_discrete_map=TOPIC_COLORS, hover_data=["prompt_text"])
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    if len(dd) >= 3 and dd.question_length.nunique() > 1:
        m, b = np.polyfit(dd.question_length, dd.answer_length, 1)
        xs = np.array([dd.question_length.min(), dd.question_length.max()])
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                 line=dict(color="#64748B", dash="dash")))
    fig.update_layout(xaxis_title="Question length (characters)", yaxis_title="Answer length (characters)")
    return style(fig, t)


def c10_quiz(d):
    t = "10 · Quiz Scores (Bar + 60% Pass Line)"
    dd = d[d.is_quiz == 1].sort_values("date_parsed")
    if dd.empty: return empty(t)
    dd = dd.assign(label=dd.topic.str.replace(" tutorial", "") + "<br>" + dd.date_parsed.dt.strftime("%d %b"))
    fig = px.bar(dd, x="label", y="percentage", color="difficulty", color_discrete_map=DIFF_COLORS,
                 text=dd.score.astype(str) + "/" + dd.total_questions.astype(str))
    fig.add_hline(y=60, line_dash="dash", line_color="#64748B", annotation_text="Pass 60%")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Score %", yaxis_range=[0, 115])
    return style(fig, t)


def c11_timeline(d):
    t = "11 · Event Timeline per User (Strip / Scatter)"
    if d.empty: return empty(t, 340)
    fig = px.scatter(d, x="date_parsed", y="user_short", color="activity",
                     color_discrete_map=ACT_COLORS, hover_data=["prompt_text", "topic_group"])
    fig.update_traces(marker=dict(size=11, opacity=.85, line=dict(width=1, color="white")))
    fig.update_layout(xaxis_title="Date", yaxis_title="")
    return style(fig, t)


NUM_COLS = ["prompt_words", "prompt_chars", "question_length", "answer_length", "message_count",
            "results_count", "score", "total_questions", "percentage"]


def c12_corr(d):
    t = "12 · Correlation Matrix (Heatmap)"
    if len(d) < 3: return empty(t)
    c = d[NUM_COLS].loc[:, d[NUM_COLS].std() > 0].corr()
    if c.shape[0] < 2: return empty(t)
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1,
                               colorscale="RdBu_r", text=np.round(c.values, 2),
                               texttemplate="%{text}", textfont=dict(size=10)))
    fig.update_xaxes(tickangle=-40)
    return style(fig, t, h=420, legend=False)


def kpis(d):
    quiz = d[d.is_quiz == 1]
    notes = d[(d.is_saved_note == 1) & (d.answer_length > 0)]
    tops = d[d.topic_group != "General / Unspecified"].topic_group
    return [
        (f"{len(d):,}", "Total records", "#4F46E5"),
        (f"{d.user_uid.nunique()}", "User accounts", "#06B6D4"),
        (f"{int((d.is_chat == 1).sum())}", "AI chats", "#8B5CF6"),
        (f"{quiz.percentage.mean():.0f}%" if len(quiz) else "–", "Avg quiz score", "#10B981"),
        (f"{notes.answer_length.mean():,.0f}" if len(notes) else "–", "Avg note length (chars)", "#F59E0B"),
        (tops.mode().iat[0] if len(tops) else "–", "Top topic", "#EF4444"),
    ]


def build_all(d):
    return [c1_donut(d), c2_area(d), c3_heat(d), c4_users(d), c5_topics(d), c6_sun(d),
            c7_box(d), c8_polar(d), c9_scatter(d), c10_quiz(d), c11_timeline(d), c12_corr(d)]


# ------------------------------------------------------------------ layout
app = Dash(__name__, title="AcademicNotesAI Dashboard")
server = app.server


def card(gid, note=None):
    kids = [html.Div(dcc.Graph(id=gid, config={"displaylogo": False}), className="card")]
    if note: kids.append(html.Div("💡 " + note, className="note"))
    return html.Div(kids)


def section(title, sub):
    return [html.Div(title, className="section"), html.Div(sub, className="section-sub")]


dmin, dmax = DF.date_parsed.min().date(), DF.date_parsed.max().date()
opts = lambda s: [{"label": v, "value": v} for v in sorted(s.unique())]

app.layout = html.Div([
    html.Div([html.H1("📚 AcademicNotesAI · Fused Dataset Dashboard"),
              html.P("Exploratory analysis of how students use an LLM-powered study assistant "
                     "(chat · YouTube · image study · quizzes · saved notes)")], className="header"),
    html.Div([
        html.Div([html.Label("Date range"),
                  dcc.DatePickerRange(id="f-date", min_date_allowed=dmin, max_date_allowed=dmax,
                                      start_date=dmin, end_date=dmax, display_format="DD MMM YYYY")],
                 className="fbox date"),
        html.Div([html.Label("User"), dcc.Dropdown(id="f-user", options=opts(DF.user), multi=True,
                                                   placeholder="All users")], className="fbox"),
        html.Div([html.Label("Activity type"), dcc.Dropdown(id="f-act", options=opts(DF.activity), multi=True,
                                                            placeholder="All activities")], className="fbox"),
        html.Div([html.Label("Topic"), dcc.Dropdown(id="f-topic", options=opts(DF.topic_group), multi=True,
                                                    placeholder="All topics")], className="fbox"),
        html.Button("Reset", id="f-reset", n_clicks=0, className="btn"),
    ], className="filters"),

    html.Div([
        html.Div(id="kpis", className="kpis"),

        *section("Overview", "What kind of activity dominates, and how does it change over time?"),
        html.Div([card("g1", "Chat is the biggest slice – students mostly ask questions rather than quiz themselves."),
                  card("g2", "Weekly stacked area shows bursts of use (exam / assignment weeks) separated by quiet periods.")],
                 className="row r12"),
        html.Div([card("g3", "Darker cells = more activity on that weekday."),
                  card("g4", "Compares total engagement and the mix of features each user prefers.")],
                 className="row r2"),

        *section("Content & Topics", "Which subjects are students studying and with which feature?"),
        html.Div([card("g5", "Cloud Computing and Machine Learning lead; 'General' rows (sign-ups, unnamed images) are excluded."),
                  card("g6", "Click a slice to zoom: inner ring = feature, outer ring = topic.")],
                 className="row r2"),

        *section("Distributions & Behaviour", "How long are prompts, and when do people study?"),
        html.Div([card("g7", "Box = middle 50 %, line = median, dots = every record."),
                  card("g8", "Only records that carry a real clock time are shown (chat-history stores date only).")],
                 className="row r2"),

        *section("Relationships & Performance", "Quiz outcomes, relationships between variables and an event timeline."),
        html.Div([card("g9", "Dashed line = linear trend. Only 8 saved notes, so treat it as indicative."),
                  card("g10", "Bars below the dashed line need revision; labels show score / total questions.")],
                 className="row r2"),
        html.Div([card("g11", "Each dot is one event – see who was active when."),
                  card("g12", "Red = positive, blue = negative correlation. Structural zeros in a fused dataset can inflate values.")],
                 className="row r2"),
    ], className="container"),
    html.Div("Built with Python · Pandas · Plotly · Dash", className="footer"),
])


@app.callback(
    [Output("kpis", "children")] + [Output(f"g{i}", "figure") for i in range(1, 13)],
    Input("f-date", "start_date"), Input("f-date", "end_date"),
    Input("f-user", "value"), Input("f-act", "value"), Input("f-topic", "value"))
def update(start, end, users, acts, topics):
    d = DF.copy()
    if start: d = d[d.date_parsed >= pd.to_datetime(start)]
    if end:   d = d[d.date_parsed < pd.to_datetime(end) + pd.Timedelta(days=1)]
    if users:  d = d[d.user.isin(users)]
    if acts:   d = d[d.activity.isin(acts)]
    if topics: d = d[d.topic_group.isin(topics)]
    cards = [html.Div([html.Div(v, className="v"), html.Div(l, className="l")],
                      className="kpi", style={"borderLeftColor": c}) for v, l, c in kpis(d)]
    return [cards] + build_all(d)


@app.callback(Output("f-date", "start_date"), Output("f-date", "end_date"),
              Output("f-user", "value"), Output("f-act", "value"), Output("f-topic", "value"),
              Input("f-reset", "n_clicks"), prevent_initial_call=True)
def reset(_):
    return dmin, dmax, None, None, None


if __name__ == "__main__":
    app.run(debug=True)
