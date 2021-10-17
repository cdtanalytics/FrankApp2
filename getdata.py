# ----------------------------------------- #
#                                           #
#  TITLE:   FRANK APP EVALUATION DASHBOARD  #
#  PURPOSE: Data Preparation                #
#  AUTHOR:  Craig Hansen                    #
#  DATE:    15 OCt 2021                     #
# ----------------------------------------- #


# ---- IMPORTS ---- #

import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from functools import reduce
from functions import *


# ---- INITIALISE THE SCALER ---- #

scaler = MinMaxScaler()


# ---- INPUT TEXT DATA PREPARATION (RAW) ---- #

# Read in the keyboard input data
df = pd.read_csv("Pilot Keyboard Input Cleaned 5 emotions.csv")

# Every second row was missing, remove these
df.dropna(subset=["inputTime"], inplace=True)

# Get the date and time
df['inputTime'] = pd.to_datetime(df['inputTime'], format='%Y-%m-%d-%H-%M-%S')

# Count the number of words in each input
df['word_count'] = df['inputText'].str.split().str.len()

# Make all text lowercase
df['inputText'] = df['inputText'].str.lower()

# Sentiment analysis - this is crude without stopwords removed or lemmitization etc
df['polarity'] = df['inputText'].apply(detect_polarity)
df['subjectivity'] = df['inputText'].apply(detect_subjectivity)

# Normalise the emotions using Min Max scaling (values will be from 0 - 1)
df[emotions_txt_sc] = scaler.fit_transform(df[emotions_txt])

# Normalise the emotions using percentile rank
df[emotions_txt_rk] = df[emotions_txt].rank(pct=True)
# Put into 3 groups (Low, Medium, High)
for i in emotions_txt_rk:
    colname = i + '_grp'
    df[colname] = bins3(df, i)

# Create final dataframe to use in matching
text_df = df.copy()

# Get the number of text inputs per hour
text_by_hr = text_df.groupby(text_df['inputTime'].dt.hour).size(
).reset_index().rename(columns={0: 'Inputs (n)', 'inputTime': 'Hour'})
text_hravg_txt = text_df.groupby(text_df['inputTime'].dt.hour)[
    emotions_txt].mean().reset_index().rename(columns={'inputTime': 'Hour'})
text_hravg_txt_sc = text_df.groupby(text_df['inputTime'].dt.hour)[
    emotions_txt_sc].mean().reset_index().rename(columns={'inputTime': 'Hour'})

del df

# ---- INPUT TEXT DATA PREPARATION (ROLLED UP ACROSS THE INPUT TIME FOR EACH RECORD) ---- #

# Roll up the emotions
emot_rollup = text_df.groupby(['userId', 'userGroup', 'inputTime'])[
    ['Anger', 'Sadness', 'Fear', 'Joy', 'Disgust', 'polarity', 'subjectivity']].max().reset_index()
# Roll up the text input
words_rollup = text_df.groupby(['userId', 'userGroup', 'inputTime'])[
    'inputText'].apply('; '.join).reset_index()
# Get the word count
words_rollup['word_count'] = words_rollup['inputText'].str.split().str.len()
# Merge together
df = pd.merge(emot_rollup, words_rollup, how='left',
              on=['userId', 'userGroup', 'inputTime'])

# Normalise the emotions using Min Max scaling (values will be from 0 - 1)
df[emotions_txt_sc] = scaler.fit_transform(df[emotions_txt])

# Normalise the emotions using percentile rank
df[emotions_txt_rk] = df[emotions_txt].rank(pct=True)
# Put into 3 groups (Low, Medium, High)
for i in emotions_txt_rk:
    colname = i + '_grp'
    df[colname] = bins3(df, i)

# Create final dataframe to use in matching
text_rollup_df = df.copy()

del df


# ---- DAILY MOODS DATA PREPARATION ---- #

# Read in the keyboard input data
df = pd.read_csv("Pilot Mood Survey.csv")

# Get the date and time
df['surveyTime'] = pd.to_datetime(df['surveyTime'], format='%Y-%m-%d-%H-%M-%S')

# Normalise the emotions using Min Max scaling (values will be from 0 - 1)
df[emotions_svy_sc] = scaler.fit_transform(df[emotions_svy])

# Normalise the emotions using percentile rank
df[emotions_svy_rk] = df[emotions_svy].rank(pct=True)
# Put into 3 groups (Low, Medium, High)
for i in emotions_svy_rk:
    colname = i + '_grp'
    df[colname] = bins3_dm(df, i)

# Create final dataframe to use in matching
dm_df = df[user_time + emotions_svy + emotions_svy_sc +
           emotions_svy_rk + emotions_svy_rk_grp].copy()

# Get the number of daily mood inputs per hour
dm_by_hr = dm_df.groupby(dm_df['surveyTime'].dt.hour).size(
).reset_index().rename(columns={0: 'Inputs (n)', 'surveyTime': 'Hour'})
dm_hr_svy = dm_df.groupby(dm_df['surveyTime'].dt.hour)[emotions_svy].mean(
).reset_index().rename(columns={'surveyTime': 'Hour'})
dm_hr_svy_sc = dm_df.groupby(dm_df['surveyTime'].dt.hour)[
    emotions_svy_sc].mean().reset_index().rename(columns={'surveyTime': 'Hour'})

hours = text_by_hr['Hour'].reset_index()
dfs = [hours, dm_by_hr, dm_hr_svy, dm_hr_svy_sc]
dm_hourly = reduce(lambda left, right: pd.merge(
    left, right, on=['Hour'], how='left'), dfs)

del df


# ---- DEQ DATA PREPARATION ---- #

# Read in the keyboard input data
df = pd.read_csv("Pilot DEQ Data.csv")

# Get the date and time
df['surveyTime'] = pd.to_datetime(df['surveyTime'], format='%Y-%m-%d-%H-%M-%S')

# Fix up lonely as the "5" has been recorded as "Quite a bit" - needs to be numeric;
df['Lonely-S'] = df['Lonely-S'].replace("Quite a bit", "5").astype('int64')

# Calcuate the DEQ scores for each emotion
df['Anger_Survey'] = df[['Anger-Ag', 'Rage-Ag',
                         'Pissed-Off-Ag', 'Mad-Ag']].sum(axis=1)
df['Sadness_Survey'] = df[['Sad-S', 'Grief-S',
                           'Lonely-S', 'Empty-S']].sum(axis=1)
df['Fear_Survey'] = df[['Terror-F', 'Panic-F',
                        'Scared-F', 'Fear-F']].sum(axis=1)
df['Joy_Survey'] = df[['Happy-H', 'Satisfaction-H',
                       'Enjoyment-H', 'Linking-H']].sum(axis=1)
df['Disgust_Survey'] = df[['Grossed-Out-Dg', 'Sickened-Dg',
                           'Nausea-Dg', 'Revulsion-Dg']].sum(axis=1)

# Normalise the emotions using Min Max scaling (values will be from 0 - 1)
df[emotions_svy_sc] = scaler.fit_transform(df[emotions_svy])

# Normalise the emotions using percentile rank
df[emotions_svy_rk] = df[emotions_svy].rank(pct=True)
# Put into 3 groups (Low, Medium, High)
for i in emotions_svy_rk:
    colname = i + '_grp'
    df[colname] = bins3(df, i)

# Create final dataframe to use in matching
deq_df = df[user_time + emotions_svy + emotions_svy_sc +
            emotions_svy_rk + emotions_svy_rk_grp].copy()

# Get the number of DEQ inputs per hour
deq_by_hr = deq_df.groupby(deq_df['surveyTime'].dt.hour).size(
).reset_index().rename(columns={0: 'Inputs (n)', 'surveyTime': 'Hour'})
deq_hr_svy = deq_df.groupby(deq_df['surveyTime'].dt.hour)[
    emotions_svy].mean().reset_index().rename(columns={'surveyTime': 'Hour'})
deq_hr_svy_sc = deq_df.groupby(deq_df['surveyTime'].dt.hour)[
    emotions_svy_sc].mean().reset_index().rename(columns={'surveyTime': 'Hour'})

hours = text_by_hr['Hour'].reset_index()
dfs = [hours, deq_by_hr, deq_hr_svy, deq_hr_svy_sc]
deq_hourly = reduce(lambda left, right: pd.merge(
    left, right, on=['Hour'], how='left'), dfs)

del df
