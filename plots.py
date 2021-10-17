
# ----------------------------------------- #
#                                           #
#  TITLE:   FRANK APP EVALUATION DASHBOARD  #
#  PURPOSE: Data Visualizatoins             #
#  AUTHOR:  Craig Hansen                    #
#  DATE:    15 Oct 2021                     #
# ----------------------------------------- #


# -- Import Python Libraries -- #
# from getdata import *
# from functions import *

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np


# -- Hourly charts -- #

def n_hourly_plots(data, x, y, title):
    fig = px.line(data, x=x, y=y, title=title)
    fig.update_xaxes(nticks=24)
    fig.update_layout(width=1500, height=300)
    st.plotly_chart(fig)


def mean_hourly_plots(data, x, y, title):
    fig = px.line(data, x=x, y=y, title=title)
    fig.update_xaxes(nticks=24)
    fig.update_layout(width=1600, height=400)
    st.plotly_chart(fig)

# -- Scatter plot with regression line BY user group -- #


def ScatterReg(data):
    fig = px.scatter(data,
                     x="Survey",
                     y="Algorithm",
                     facet_col="Emotion",
                     trendline="ols")
    fig.update_layout(title='Scatter plot of Survey vs. Algorithm')
    fig.update_layout(width=1600, height=400)
    for a in fig.layout.annotations:
        a.text = a.text.split("=")[1]
    st.plotly_chart(fig)


def ScatterRegUser(data):
    fig = px.scatter(data,
                     x="Survey",
                     y="Algorithm",
                     facet_col="Emotion",
                     color='userGroup',
                     trendline="ols")
    fig.update_layout(title='Scatter plot of Survey vs. Algorithm')
    fig.update_layout(width=1600, height=400)
    for a in fig.layout.annotations:
        a.text = a.text.split("=")[1]
    st.plotly_chart(fig)

# -- Correlation plot of the emotions -- #


def CorrHeatmap(data, emotionslist):
    corr = data[emotionslist].corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_mask = corr.mask(mask)
    ticklist = emotionslist
    ticklabels = emotionslist
    fig = go.Figure(data=go.Heatmap(
        z=corr_mask.values,
        y=corr_mask.index.values,
        x=corr_mask.columns.values,
        colorscale=px.colors.diverging.RdBu,
        zmin=-1, zmax=1,
        xgap=3,
        ygap=3))
    fig.update_xaxes(tickmode='array',
                     tickvals=ticklist,
                     ticktext=ticklabels)
    fig.update_yaxes(tickmode='array',
                     tickvals=ticklist,
                     ticktext=ticklabels)
    fig.update_layout(title="Correlation Heatmap",
                      yaxis_autorange='reversed', template='plotly_white')
    st.plotly_chart(fig)

# -- Stacked bar chart comparing the tertile membership of Algorithm vs. Survey  -- #


def stacked_bar(data, title):
    fig = px.bar(data, x="Algorithm", y='%', color="Survey",
                 title=title,
                 barmode='relative')
    st.plotly_chart(fig)
