
import streamlit as st

BEFORE_TEXT = """Welcome to the experiment!
\nWe will start with few questions about your demographics, and you daily use in of LLMS."""


def next_page():
    st.session_state.cur_page = 'instructions'


def before():
    st.markdown(BEFORE_TEXT)
    st.selectbox('Age', key='age', options=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], index=None)
    st.radio('Gender', options=['male', 'female', 'other', 'prefer not to say'], key='gender', index=None)
    st.radio('How often do you use LLMs in your daily life?',
             options=['1 - never', '2', '3', '4', '5 - all the time'],
             key='llms_usage', index=None)
    st.radio('How much do you trust the output that your favorite LLM generates?',
             options=['1 - poorly, I always edit what the model generates', '2', '3', '4', '5 - greatly, I use its output as is'],
             key='llms_trust_before', index=None)
    st.radio('How satisfied are you with the experiment of using LLMs?',
             options=['1 - not at all', '2', '3', '4', '5 - completely satisfied'],
             key='llms_satisfy_before', index=None)
    st.selectbox('Do you pay subscription to any LLM service?', key='llms_subscription', options=['yes', 'no'], index=None)
    st.text_input('Which tasks do you use LLMs for?')
    st.button('Submit and continue', key='next_button1', on_click=next_page)
