"""predictions page: trend lines, what drives a value, and what-if predictions."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from charts import series_colors, style_figure
from predict import default_target, fit_drivers, fit_quality, fit_trend, usable_features
from ui import page_header, require_dataset

CAUTION = ("These are patterns in your data, not proof of cause and effect, and predictions "
           "are only as good as the data behind them.")


def trend_tab(df, numeric_columns, date_columns):
    x_options = numeric_columns + date_columns
    controls = st.columns(3)
    x = controls[0].selectbox("X (what you know)", x_options, key="trend_x")
    y_options = [c for c in numeric_columns if c != x]
    y_default = default_target(y_options, df) if y_options else None
    y = controls[1].selectbox("Y (what to predict)", y_options, key="trend_y",
                              index=y_options.index(y_default) if y_default else 0)
    shape = controls[2].segmented_control("Shape", ["Straight line", "Curve"], default="Straight line",
                                          required=True, key="trend_shape")
    if not y:
        st.info("This needs at least two numeric columns.", icon=":material/info:")
        return
    try:
        trend, xs, ys = fit_trend(df, x, y, degree=1 if shape == "Straight line" else 2)
    except ValueError as error:
        st.warning(str(error))
        return

    colors = series_colors()
    line_x = np.linspace(xs.min(), xs.max(), 200)
    fig = px.scatter(x=df[x], y=ys, labels={"x": x, "y": y}, color_discrete_sequence=colors,
                     title=f"{y} vs {x}")
    line_axis = (df[x].min() + pd.to_timedelta(line_x, unit="D")) if trend.x_is_date else line_x
    fig.add_trace(go.Scatter(x=line_axis, y=trend.predict(line_x), mode="lines", name="Trend",
                             line=dict(color=colors[1], width=2)))
    st.plotly_chart(style_figure(fig), width="stretch")

    metrics = st.container(horizontal=True)
    metrics.metric("Fit (R²)", f"{trend.r2:.2f}", border=True,
                   help="0 means the line explains nothing, 1 means it explains every point.")
    metrics.metric("Relationship", fit_quality(trend.r2).capitalize(), border=True)
    st.caption(f"Equation: `{trend.equation(x, y)}`")

    if not trend.x_is_date:
        with st.container(border=True):
            st.markdown("##### Predict")
            value = st.number_input(f"If {x} is", value=float(xs.median()), key="trend_input")
            st.markdown(f"…then **{y}** is expected to be about **{trend.predict([value])[0]:.4g}**.")
            if not xs.min() <= value <= xs.max():
                st.caption(f"Careful: {value:g} is outside the range in your data "
                           f"({xs.min():g} to {xs.max():g}), so this guess is less reliable.")


def drivers_tab(df, numeric_columns):
    controls = st.columns([1, 2])
    target = controls[0].selectbox("What do you want to explain?", numeric_columns, key="driver_target",
                                   index=numeric_columns.index(default_target(numeric_columns, df)))
    candidates = usable_features(df, target)
    features = controls[1].multiselect("Using these columns", candidates, default=candidates[:8],
                                       key="driver_features")
    try:
        model = fit_drivers(df, target, features)
    except ValueError as error:
        st.info(str(error), icon=":material/info:")
        return

    importance = model.importance()
    top = importance.iloc[0]
    quality = fit_quality(model.r2)
    st.markdown(
        f"Together these columns explain **{model.r2:.0%}** of the differences in **{target}** "
        f"({quality} relationship, based on {model.rows_used:,} complete rows). "
        f"The biggest factor is **{top['Column']}**"
        + (f" ({top['Direction'].lower()})." if top["Direction"] != "Depends on the category" else ".")
    )

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        ordered = importance.sort_values("Influence")
        fig = px.bar(ordered, x="Influence", y="Column", orientation="h",
                     title=f"What is most linked to {target}", color_discrete_sequence=series_colors())
        fig.update_layout(xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Share of influence")
        st.plotly_chart(style_figure(fig), width="stretch")
    with table_col:
        st.dataframe(
            importance, hide_index=True, width="stretch",
            column_config={"Influence": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                        max_value=1)},
        )
    st.session_state.prediction_model = model


def what_if_tab(df):
    model = st.session_state.get("prediction_model")
    if model is None or model.target not in df.columns:
        st.info("Set up a model on the 'What drives it' tab first.", icon=":material/info:")
        return
    st.caption(f"Change the values to see the predicted **{model.target}**. "
               "Starting values are the typical (middle or most common) ones in your data.")
    values = {}
    inputs = st.columns(3)
    for position, feature in enumerate(model.features):
        series = df[feature].dropna()
        box = inputs[position % 3]
        if pd.api.types.is_numeric_dtype(series):
            values[feature] = box.number_input(feature, value=float(series.median()), key=f"whatif_{feature}")
        else:
            options = series.astype(str).value_counts().index.tolist()
            values[feature] = box.selectbox(feature, options, key=f"whatif_{feature}")
    prediction = model.predict_one(values)
    st.metric(f"Predicted {model.target}", f"{prediction:.4g}", border=True,
              help=f"Based on a model that explains {model.r2:.0%} of the differences in {model.target}.")


page_header("Predictions", "Find trends, see what drives a number, and try what-if scenarios.")
df = require_dataset()
numeric_columns = df.select_dtypes("number").columns.tolist()
date_columns = df.select_dtypes("datetime").columns.tolist()
if not numeric_columns:
    st.info("Predictions need at least one numeric column.", icon=":material/info:")
    st.stop()
st.caption(CAUTION)

tab_trend, tab_drivers, tab_what_if = st.tabs([
    ":material/trending_up: Trend",
    ":material/account_tree: What drives it",
    ":material/tune: What if",
])
with tab_trend:
    trend_tab(df, numeric_columns, date_columns)
with tab_drivers:
    drivers_tab(df, numeric_columns)
with tab_what_if:
    what_if_tab(df)
