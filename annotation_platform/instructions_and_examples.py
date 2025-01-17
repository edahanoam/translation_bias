
import streamlit as st

INSTRUCTIONS = """You should carry out a light post-editing (e.g. for internal communication). In light post-editing you need to ensure that the correct information is conveyed while accepting a more relaxed standard of fluency and style. For this reason you should post-edit the machine translation suggestion by applying the minimal edits required to transform the system output into a fluent sentence with the same meaning as the source sentence: 
make only the necessary edits 
do not make “preferential” edits aimed to adjust the style.

Note that the sentences may contain ambiguous references. In such cases, simply ignore them and accept the output generated by the MT system.

Please carry out the translation task “as naturally as possible”, at your usual working pace, but without taking breaks. 


\n"""
EXAMPLES = """<Examples for the task>.\n"""


def next_page():
    st.session_state.cur_page = 'training'
    #st.session_state.cur_page = 'experiment'


def instructions_page():
    st.markdown(INSTRUCTIONS)
    #st.markdown(EXAMPLES)
    st.button('Start training', key='next_button2', on_click=next_page)
    #st.button('Start', key='next_button2', on_click=next_page)

