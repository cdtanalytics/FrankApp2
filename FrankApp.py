
# ----------------------------------------- #
#                                           #
#  TITLE:   FRANK APP EVALUATION DASHBOARD  #
#  PURPOSE: Main App                        #
#  AUTHOR:  Craig Hansen                    #
#  DATE:    14 Oct 2021                     #
#                                           #
# ----------------------------------------- #


# -- IMPORT LIBRARIES -- #

import streamlit as st
import pandas as pd
import seaborn as sns

from getdata import *
from functions import *
from plots import *

# -- SET STYLES -- #

st.set_page_config(page_title="FRANK App2", layout="wide")

sns.set(style='darkgrid', font_scale=1.1)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# -- IMPORT LOCAL LIBRARIES -- #


# ---------- MAIN PAGE ---------- #


def main():

    # - HEADING - #
    st.markdown("<h1 style='text-align: center; color: black; font-size:60px;'>FRANK APP (ROUND 2)</h1>",
                unsafe_allow_html=True)

    st.write('---')

    # Raw Data
    st.header('1. Import the raw text input data')
    st.subheader('1a. Calculate the scaled data and the percentile groupings')
    st.write(text_df.head(20).astype('object'))
    st.subheader('1b. Summarise the raw data')
    st.write(text_df[emotions_txt].describe().round(3).T.astype('object'))
    st.subheader('1c. Summarise the data by daily hours')
    n_hourly_plots(data=text_by_hr, x="Hour",
                   y="Inputs (n)", title='Number of text inputs by hour')
    mean_hourly_plots(data=text_hravg_txt, x="Hour",
                      y=emotions_txt, title='Mean emotion by hour (raw values)')

    st.write('---')

    # Raw Data Rolled up
    st.header('2. Roll up the input text for each user')
    st.write('e.g. look at records for userId 19g68kmexxfoh')
    st.subheader('2a. Calculate the scaled data and the percentile groupings')
    st.write(text_rollup_df.head(20).astype('object'))
    st.subheader('2b. Summarise the rolled up data')
    st.write(text_rollup_df[emotions_txt].describe().round(
        3).T.astype('object'))
    st.write('Note: You will notice that the mean is now higher because the data were rolled up by selecting the maximum scores')

    st.write('---')

    # Daily Moods Data
    st.header('3. Import the raw Daily Moods data')
    st.subheader('3a. Calculate the scaled data and the percentile groupings')
    st.write(dm_df.head(20).astype('object'))
    st.subheader('3b. Summarise the Daily Moods data')
    st.write(dm_df[emotions_svy].describe().round(3).T.astype('object'))
    st.subheader('3c. Summarise the data by daily hours')
    n_hourly_plots(data=dm_hourly, x="Hour", y="Inputs (n)",
                   title='Number of Daily Moods inputs by hour')
    mean_hourly_plots(data=dm_hourly, x="Hour", y=emotions_svy,
                      title='Mean Daily Moods emotion by hour (raw values)')

    st.write('---')

    # DEQ Data
    st.header('4. Import the raw DEQ data')
    st.subheader('4a. Calculate the scaled data and the percentile groupings')
    st.write(deq_df.head(20).astype('object'))
    st.subheader('4b. Summarise the DEQ data')
    st.write(deq_df[emotions_svy].describe().round(3).T.astype('object'))
    st.subheader('4c. Summarise the data by daily hours')
    n_hourly_plots(data=deq_hourly, x="Hour",
                   y="Inputs (n)", title='Number of DEQ inputs by hour')
    mean_hourly_plots(data=deq_hourly, x="Hour", y=emotions_svy,
                      title='Mean DEQ emotion by hour (raw values)')

    st.write('---')

    st.header(
        '5. Link input text data (Algorithm) with survey data (Daily Moods or DEQ)')

    st.subheader(
        'Step 1. Select the text input data (Raw or Rolled up) to be linked to the survey data')
    main_file = st.selectbox('', ('Raw', 'Rolled up'))

    st.subheader(
        'Step 2. Select the survey data (Daily Moods or DEQ) to be linked to the text input data')
    linkage_file = st.selectbox('', ('Daily Moods', 'DEQ'))
    st.subheader(
        'Step 3. Select a window (minutes) for linking the input Text with surveys input timing')
    linkage_period = st.selectbox('', (30, 60, 90, 120, 150, 180))
    st.write(
        f'The {main_file} text file is linked with the {linkage_file} survey file within {linkage_period} mins either side of the timing of the survey input')

    # ---- RUN THE DATA LINKAGE ---- #

    # Select the linkage file and time window in the drop down box

    if main_file == 'Raw':
        main_txt_file = text_df
    elif main_file == "Rolled up":
        main_txt_file = text_rollup_df

    if linkage_file == 'DEQ':
        svy_file = deq_df
    elif linkage_file == "Daily Moods":
        svy_file = dm_df

    # Run the linkage
    matched = linkage(main_df=main_txt_file,
                      linkdf=svy_file,
                      mins=linkage_period)

    # Get the number of matched records
    n_records = matched.shape[0]

    # Prepare long files for plots
    if linkage_file == "Daily Moods":
        matched_raw_long = make_long_raw(matched)
        matched_scaled_long = make_long_scaled(matched)
    elif linkage_file == "DEQ":
        matched_raw_long = make_long_raw(matched)
        matched_scaled_long = make_long_scaled(matched)

    # View the matched data

    st.subheader(
        f'5a. View the matched data, there are {n_records} matched records')
    st.write('inputTime = The time the text was submitted. surveyTime = The time the survey was submitted.')
    st.write(matched.astype('object'))

    # -- Raw -- #
    st.subheader('5b. Correlations (Raw Data)')
    st.write(spearman_corr(matched, corr_raw_list))
    # Scatter plot
    st.subheader('5c. Scatter Plot with Regression Line (Raw Data)')
    ScatterReg(matched_raw_long)
    # Scatter plot BY ueserGroup
    st.subheader(
        '5d. Scatter Plot with Regression Line BY User Group (Raw Data)')
    ScatterRegUser(matched_raw_long)

  # -- Scaled -- #
    st.subheader('5e. Correlations (Scaled Data)')
    st.write(spearman_corr(matched, corr_scaled_list))
    # Scatter plot
    st.subheader('5f. Scatter Plot with Regression Line (Scaled Data)')
    ScatterReg(matched_scaled_long)
    # Scatter plot BY ueserGroup
    st.subheader(
        '5g. Scatter Plot with Regression Line BY User Group (Scaled Data)')
    ScatterRegUser(matched_scaled_long)

    st.write('---')

    # Comparison of tertile membership
    st.header('6. Membership in Low/Medium/High groups (Algorithm vs. Survey)')

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.subheader('6a. Anger')
        anger = consistency_counts(
            matched, 'txt_Anger_rank_grp', 'Anger_rank_grp')
        st.write(anger.astype('object'))
    with col2:
        st.subheader('6b. Sadness')
        sad = consistency_counts(
            matched, 'txt_Sadness_rank_grp', 'Sadness_rank_grp')
        st.write(sad.astype('object'))
    with col3:
        st.subheader('6c. Fear')
        fear = consistency_counts(
            matched, 'txt_Fear_rank_grp', 'Fear_rank_grp')
        st.write(fear.astype('object'))

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.subheader('6d. Joy')
        joy = consistency_counts(
            matched, 'txt_Joy_rank_grp', 'Joy_rank_grp')
        st.write(joy.astype('object'))
    with col2:
        st.subheader('6e. Disgust')
        disgust = consistency_counts(
            matched, 'txt_Disgust_rank_grp', 'Disgust_rank_grp')
        st.write(disgust.astype('object'))

    st.write('---')

    st.header(
        '7. Correlations between Algorithm and Survey based on subjectivity/polarity/word count of text')
    st.write('Note: Each column represents >=. For example, Subjectivity column "0.10" refers to filtering the data where subjectivity >=0.1')

    st.subheader(
        '7a. By Subjectivity (columns represent increasing subjectivity 0 - 10)')
    st.write(corr_by_subjectivity(
        matched, emotions_txt, emotions_svy).astype('object'))
    st.subheader(
        '7b. By Polarity (columns represent increasing sentiment -1.0 to 1.0)')
    st.write(corr_by_polarity(matched, emotions_txt,
             emotions_svy).astype('object'))
    st.subheader('7c. By Word Count (columns represent increasing word count)')
    st.write(corr_by_words(matched, emotions_txt, emotions_svy).astype('object'))

    st.write('---')

    st.header(
        '8. AMONG FRANK+KEYBOARD GROUP: Correlations between Algorithm and Survey based on subjectivity/polarity/word count of text')
    st.write('Note: Each column represents >=. For example, Subjectivity column "0.10" refers to filtering the data where subjectivity >=0.1')

    st.subheader(
        '8a. By Subjectivity (columns represent increasing subjectivity 0 - 10)')
    st.write(corr_by_subjectivity(
        matched[matched['userGroup']="FrankKeyboard"], emotions_txt, emotions_svy).astype('object'))
    st.subheader(
        '8b. By Polarity (columns represent increasing sentiment -1.0 to 1.0)')
    st.write(corr_by_polarity(matched[matched['userGroup']="FrankKeyboard"], emotions_txt,
             emotions_svy).astype('object'))
    st.subheader('8c. By Word Count (columns represent increasing word count)')
    st.write(corr_by_words(matched[matched['userGroup']="FrankKeyboard"], emotions_txt, emotions_svy).astype('object'))
    
    st.write('---')

    st.header(
        '9. AMONG FRANK GROUP: Correlations between Algorithm and Survey based on subjectivity/polarity/word count of text')
    st.write('Note: Each column represents >=. For example, Subjectivity column "0.10" refers to filtering the data where subjectivity >=0.1')

    st.subheader(
        '9a. By Subjectivity (columns represent increasing subjectivity 0 - 10)')
    st.write(corr_by_subjectivity(
        matched[matched['userGroup']="Frank"], emotions_txt, emotions_svy).astype('object'))
    st.subheader(
        '9b. By Polarity (columns represent increasing sentiment -1.0 to 1.0)')
    st.write(corr_by_polarity(matched[matched['userGroup']="Frank"], emotions_txt,
             emotions_svy).astype('object'))
    st.subheader('9c. By Word Count (columns represent increasing word count)')
    st.write(corr_by_words(matched[matched['userGroup']="Frank"], emotions_txt, emotions_svy).astype('object'))
    
    st.write('---')

    st.header(
        '10. AMONG KEYBOARD GROUP: Correlations between Algorithm and Survey based on subjectivity/polarity/word count of text')
    st.write('Note: Each column represents >=. For example, Subjectivity column "0.10" refers to filtering the data where subjectivity >=0.1')

    st.subheader(
        '10a. By Subjectivity (columns represent increasing subjectivity 0 - 10)')
    st.write(corr_by_subjectivity(
        matched[matched['userGroup']="Keyboard"], emotions_txt, emotions_svy).astype('object'))
    st.subheader(
        '10b. By Polarity (columns represent increasing sentiment -1.0 to 1.0)')
    st.write(corr_by_polarity(matched[matched['userGroup']="Keyboard"], emotions_txt,
             emotions_svy).astype('object'))
    st.subheader('10c. By Word Count (columns represent increasing word count)')
    st.write(corr_by_words(matched[matched['userGroup']="Keyboard"], emotions_txt, emotions_svy).astype('object'))
    
if __name__ == "__main__":
    main()
