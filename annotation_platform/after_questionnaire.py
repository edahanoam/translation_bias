
import streamlit as st
import time

AFTER_TEXT = """Please take a moment to answer the following questionnaire on your experience with our models' translation capabilities.\n"""

DEMOGRAPHICS =  '''
<p><font size="4"> Please answer the following questions: </font> <br>
           <br> Note:
        <ul>
        <li>The data will not be used for identification in any way.</li>
        <li>Answering these questions is voluntary.</li>
        <li>You can skip any question you do not wish to answer.</li>
        </ul>
'''




# def after():
#     st.markdown(AFTER_TEXT)
#     st.radio('How much did you trust the LLM that generated the translations?',
#              options=['1 - poorly, I always edited what the model generates', '2', '3', '4',
#                       '5 - greatly, I use its output as is'],
#              key='llm_trust_after', index=None)
#     st.radio('How satisfied are you with the LLM that generated the translations?',
#              options=['1 - not at all', '2', '3', '4', '5 - completely satisfied'],
#              key='llm_satisfy_after', index=None)
#     st.radio('How gender biased do you think that this LLM is?',
#              options=['1 - not biased at all', '2', '3', '4', '5 - 100% biased'],
#              key='llm_bias', index=None)
#     st.button('Finish', key='next_button1', on_click=next_page)
#
#
# def after_with_all_survey():
#     st.markdown(AFTER_TEXT)
#
#
#     st.radio('How much did you trust the LLM that generated the translations?',
#              options=['1 - poorly, I always edited what the model generates', '2', '3', '4',
#                       '5 - greatly, I use its output as is'],
#              key='llm_trust_after', index=None)
#     st.radio('How satisfied are you with the LLM that generated the translations?',
#              options=['1 - not at all', '2', '3', '4', '5 - completely satisfied'],
#              key='llm_satisfy_after', index=None)
#     st.radio('How gender biased do you think that this LLM is?',
#              options=['1 - not biased at all', '2', '3', '4', '5 - 100% biased'],
#              key='llm_bias', index=None)
#
#     st.radio('How often do you use LLMs in your daily life?',
#              options=['1 - never', '2', '3', '4', '5 - all the time'],
#              key='llms_usage', index=None)
#     st.radio('How much do you trust the output that your favorite LLM generates?',
#              options=['1 - poorly, I always edit what the model generates', '2', '3', '4', '5 - greatly, I use its output as is'],
#              key='llms_trust_before', index=None)
#     st.radio('How satisfied are you with the experiment of using LLMs?',
#              options=['1 - not at all', '2', '3', '4', '5 - completely satisfied'],
#              key='llms_satisfy_before', index=None)
#     st.selectbox('Do you pay a subscription to any LLM service?', key='llms_subscription', options=['yes', 'no'], index=None)
#     st.text_input('Which tasks do you use LLMs for?')
#
#     st.selectbox('Age', key='age', options=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], index=None)
#     st.radio('Gender', options=['male', 'female', 'other', 'prefer not to say'], key='gender', index=None)
#
#     st.button('Finish', key='next_button1', on_click=next_page)



def demographics():
    with st.columns([1, 2, 1])[1]:
    #the demographics
        st.markdown(DEMOGRAPHICS, unsafe_allow_html=True)
        age = st.selectbox('what is your age?', key='age', options=['18-24', '25-29','30-34','35-39', '40-44','45-49', '50-54','55-59','60-64','65-69','70-74','75-79','Above 79'], index=None)
        gender = st.selectbox('What is your gender?', key='gender', options=['Female','Male','Non-binary','Other','Prefer not to say'], index=None)
        degree= st.selectbox("""What is the highest degree or level of school you have completed? If currently enrolled, highest degree received.""" ,key='degree',options=['Less than a high school diploma','High school degree or equivalent','Bachelor\'s degree (e.g. BA, BS) ','Master\'s degree (e.g. MA, MS, MEd)','Doctorate','Other'], index=None)
        employment= st.selectbox("""What is your current employment status?""" ,key='employment',options=['Full-time employment','Part-time employment','Unemployed','Self-employed','Home-maker','Student','Retired'], index=None)
    #LLMs

        usage = st.radio('How often do you use Large Language Models (LLMs) such as ChatGPT, GPT-4, or other AI tools in your daily life?',
             options=['1 - never', '2', '3', '4', '5 - all the time'],
             key='llms_usage', index=None)
        satisfaction = st.radio('How satisfied are you with the translations produced by the model?',
             options=['1 - not at all', '2', '3', '4', '5 - completely satisfied'],
             key='llm_satisfy_after', index=None)
        biased = st.radio('How gender biased do you think that this LLM is?',
             options=['1 - not biased at all', '2', '3', '4', '5 - 100% biased'],
             key='llm_bias', index=None)
        comments = st.text_area("Anything you want to tell us?", placeholder="Feel free to write any additional feedback or comments here...")
        st.button('Finish', key='next_button1', on_click=lambda:next_page(age, gender, degree, employment, usage, satisfaction, biased, comments))



def next_page(age, gender, degree, employment, usage, satisfaction, biased, comments):
    st.session_state.ws.append_row([age,gender,degree,employment,usage, satisfaction,biased,comments])
    st.session_state.ws.append_row([f"Time took to all examples: {time.time() - st.session_state.start_time} seconds"])

    st.session_state.cur_page = 'finish'
