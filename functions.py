
# ----------------------------------------- #
#                                           #
#  TITLE:   FRANK APP EVALUATION DASHBOARD  #
#  PURPOSE: Functions                       #
#  AUTHOR:  Craig Hansen                    #
#  DATE:    15 Oct 2021                     #
# ----------------------------------------- #


# ---- IMPORTS ---- #

import pandas as pd
import numpy as np
from textblob import TextBlob

# from getdata import *
# from plots import *


# ---- VARIOUS LISTS FOR ANALYSES ---- #


# Userid, inputTime, userGroup
user_time_grp = ['userId', 'userGroup', 'inputTime']
user_time = ['userId', 'surveyTime']

words = ['word_count', 'polarity', 'subjectivity']

# Emotions from input text file
emotions_txt = ['Anger', 'Sadness', 'Fear', 'Joy', 'Disgust']
emotions_txt_sc = ['txt_Anger_scaled', 'txt_Sadness_scaled',
                   'txt_Fear_scaled', 'txt_Joy_scaled', 'txt_Disgust_scaled']
emotions_txt_rk = ['txt_Anger_rank', 'txt_Sadness_rank',
                   'txt_Fear_rank', 'txt_Joy_rank', 'txt_Disgust_rank']
emotions_txt_rk_grp = ['txt_Anger_rank_grp', 'txt_Sadness_rank_grp',
                       'txt_Fear_rank_grp', 'txt_Joy_rank_grp', 'txt_Disgust_rank_grp']

text_cols = user_time_grp + emotions_txt + emotions_txt_sc + \
    emotions_txt_rk + emotions_txt_rk_grp + words

# Emotions from input survey files
emotions_svy = ['Anger_Survey', 'Sadness_Survey',
                'Fear_Survey', 'Joy_Survey', 'Disgust_Survey']
emotions_svy_grp = ['Anger_Survey_grp', 'Sadness_Survey_grp',
                    'Fear_Survey_grp', 'Joy_Survey_grp', 'Disgust_Survey_grp']
emotions_svy_sc = ['Anger_scaled', 'Sadness_scaled',
                   'Fear_scaled', 'Joy_scaled', 'Disgust_scaled']
emotions_svy_rk = ['Anger_rank', 'Sadness_rank',
                   'Fear_rank', 'Joy_rank', 'Disgust_rank']
emotions_svy_rk_grp = ['Anger_rank_grp', 'Sadness_rank_grp',
                       'Fear_rank_grp', 'Joy_rank_grp', 'Disgust_rank_grp']

svy_cols = user_time + emotions_svy + emotions_svy_sc + \
    emotions_svy_rk + emotions_svy_rk_grp

# Scaled variables (algorithm vs. survey)
corr_raw_list = emotions_txt + emotions_svy
corr_scaled_list = emotions_txt_sc + emotions_svy_sc


# ---- SENTIMENT ANALYSES ---- #
def detect_polarity(text):
    return TextBlob(text).sentiment.polarity


def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# ---- CUT THE EMOTION SCORES INTO TERTILES (LOW, MEDIUM, HIGH) ---- #

def bins3(df, col):
    tertiles = df[col].describe(percentiles=[0.33, 0.66])
    return pd.cut(df[col], bins=[0, tertiles[4], tertiles[6], 1.0], duplicates='raise', labels=['1. Low', '2. Medium', '3. High'])


def bins3_dm(df, col):
    tertiles = df[col].describe(percentiles=[0.33, 0.66])
    if col == 'Disgust_rank':
        return pd.cut(df[col], bins=[0, tertiles[4], tertiles[6], 1.0], duplicates='drop', labels=['1. Low/2. Medium', '3. High'])
    else:
        return pd.cut(df[col], bins=[0, tertiles[4], tertiles[6], 1.0], labels=['1. Low', '2. Medium', '3. High'])

# ---- DATA LINKAGE BETWEEN INPUT TEXT AND EITHER DAILY MOODS OR DEQ ---- #


def linkage(main_df, linkdf, mins):

    # Calculate a time period for matching
    linkdf['before'] = linkdf['surveyTime'] - pd.Timedelta(minutes=mins)
    linkdf['after'] = linkdf['surveyTime'] + pd.Timedelta(minutes=mins)

    ba = ['before', 'after']

    # Matching Step 1: Merge on the userId
    merge1 = pd.merge(main_df[text_cols],
                      linkdf[svy_cols + ba],
                      how='inner',
                      left_on='userId',
                      right_on='userId')
    # Matching Step 2: Filter out only records where the text input time is between the selected survey times
    merge2 = merge1[(merge1['inputTime'] >= merge1['before'])
                    & (merge1['inputTime'] <= merge1['after'])]

    return merge2


# ---- PREPARE LONG FILES FOR PLOTS WITH USERGROUPS ---- #

# To create a multi-column plot you have to make the data long

def make_long_raw(df):

    # Raw Data
    algo = df.melt(id_vars=['userId', 'userGroup'],
                   value_vars=emotions_txt,
                   var_name='Emotion',
                   value_name='Algorithm')

    algo["Emotion"].replace({'Anger': 'txt_Anger',
                             'Disgust': 'txt_Disgust',
                             'Fear': 'txt_Fear',
                             'Joy': 'txt_Joy',
                             'Sadness': 'txt_Sadness'}, inplace=True)

    svy = df.melt(id_vars=['userId', 'userGroup'],
                  value_vars=emotions_svy,
                  var_name='Emotion',
                  value_name='Survey')

    matched_raw_long = pd.merge(algo, svy['Survey'],
                                left_index=True, right_index=True)
    del algo, svy

    return matched_raw_long


def make_long_scaled(df):
    # Scaled
    algo = df.melt(id_vars=['userId', 'userGroup'],
                   value_vars=emotions_txt_sc,
                   var_name='Emotion',
                   value_name='Algorithm')

    svy = df.melt(id_vars=['userId', 'userGroup'],
                  value_vars=emotions_svy_sc,
                  var_name='Emotion',
                  value_name='Survey')

    matched_scaled_long = pd.merge(algo, svy['Survey'],
                                   left_index=True, right_index=True)
    del algo, svy

    return matched_scaled_long


# ---- CROSSTAB OF THE LOW/MEDIUM/IGH FOR ALGORITH VS. SURVEY ---- #

def consistency_counts(df, textCol, svyCol):
    res = df.groupby([textCol, svyCol]).size().reset_index().rename(
        columns=({textCol: 'Algorithm', svyCol: 'Survey', 0: 'Count'}))

    res['%'] = round((res['Count'] / res.groupby(['Algorithm'])
                     ['Count'].transform('sum')) * 100, 1)
    return res


# ---- SPEARMAN CORRELATIONS ---- #

def spearman_corr(data, emotions_list):
    corr = data[emotions_list].corr(method='spearman')
    return corr


def corr_by_subjectivity(df, textlist, svylist):

    corr_results = pd.DataFrame()

    for t, s in zip(textlist, svylist):
        for time in np.arange(0.1, 1.0, 0.1):
            time = time.round(3)
            temp = df[df['subjectivity'] >= time]
            corr = temp[[t, s]].corr(method='spearman')
            result = corr.iloc[0, 1].round(4)
            res = pd.DataFrame({'Algorithm': [t], 'Survey': [s], 'Correlation': [
                               result], 'Subjectivity': [time]})
            corr_results = corr_results.append(res)
    return corr_results.pivot(index=['Algorithm', 'Survey'], columns='Subjectivity', values='Correlation')


def corr_by_polarity(df, textlist, svylist):

    corr_results = pd.DataFrame()

    for t, s in zip(textlist, svylist):
        for time in np.arange(-1, 1.2, 0.2):
            time = time.round(3)
            temp = df[df['polarity'] >= time]
            corr = temp[[t, s]].corr(method='spearman')
            result = corr.iloc[0, 1].round(4)
            res = pd.DataFrame({'Algorithm': [t], 'Survey': [
                               s], 'Correlation': [result], 'Polarity': [time]})
            corr_results = corr_results.append(res)
    return corr_results.pivot(index=['Algorithm', 'Survey'], columns='Polarity', values='Correlation')


def corr_by_words(df, textlist, svylist):

    corr_results = pd.DataFrame()

    for t, s in zip(textlist, svylist):
        for time in np.arange(0, 50, 5):
            time = time.round(3)
            temp = df[df['word_count'] >= time]
            corr = temp[[t, s]].corr(method='spearman')
            result = corr.iloc[0, 1].round(4)
            res = pd.DataFrame({'Algorithm': [t], 'Survey': [s], 'Correlation': [
                               result], 'Word Count': [time]})
            corr_results = corr_results.append(res)
    return corr_results.pivot(index=['Algorithm', 'Survey'], columns='Word Count', values='Correlation')
