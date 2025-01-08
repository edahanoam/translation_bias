
import streamlit as st
from experiment_helper import display_single_example, save_annotation
import csv
import random



def csv_to_format():
    data_array = []

    # Open the CSV file containing your data
    with open('short_translation_mine_translations.csv', newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        reader = csv.DictReader(csvfile)

        # Loop through rows in the CSV file
        for row in reader:
            # Each row is a dictionary with keys as column headers
            # Append a new dictionary to the list with the required keys
            data_array.append({
                'input': row['sentence_text'],
                'output': row['model_translations'],
                #'gold': row['gold']
            })
    random.shuffle(data_array)
    return data_array[:20]


def load_all_test_data():
    st.session_state.test_sample_index = 0
    #sample_1 = {'input': 'I am a teacher', 'output': 'אני מורה', 'gold': 'אני מורה'}
    data = csv_to_format()
    #return [sample_1]
    return data


def load_all_test_data_from_spreadsheet():
    st.session_state.test_sample_index = 0
    data_array = []

    worksheet = st.session_state.ws

    # Fetch all data from the worksheet
    data = worksheet.get_all_records()
    # Loop through rows in the fetched data
    for row in data:

        # Each row is a dictionary with keys as column headers from your Google Sheet
        data_array.append({
            'input': row['sentence_text'].rstrip('.'),  # Ensure your column name matches the Google Sheets column name
            'output': row['model_translations'].rstrip('.'),  # Ensure your column name matches the Google Sheets column name
            #'gold': row['gold']  # Ensure your column name matches the Google Sheets column name
        })


    # Shuffle the data array to randomize
    #random.shuffle(data_array)
    #TODO: add all of the correction, and make sure that each annotator get a seperated spread sheet. also i think that each of the annotators will get the sheet with the translation on each on

    # Return only the first 20 itemsK
    return data_array[:20]



def load_data_from_spreadsheet():
    # Get all the data from the worksheet
    data = st.session_state.ws.get_all_values()



def next_page():
    st.session_state.cur_page = 'after'


def next_sample(translation):
    worksheet = st.session_state.ws
    # +2 as the index starts at 1 in spreadsheets and 1 is the header
    place = 'F'+str(st.session_state.test_sample_index+2)
    worksheet.update(place, [[translation]])
    st.session_state.test_sample_index += 1



def experiment():
    if 'test_data' not in st.session_state:
        st.session_state.test_data = load_all_test_data_from_spreadsheet()
    if st.session_state.test_sample_index >= len(st.session_state.test_data):
        with st.columns([1, 2, 1])[1]:
            st.write('Testing is over!')
            st.button('Continue', key='next_button3', on_click=next_page)
    else:
        current_sample = st.session_state.test_data[st.session_state.test_sample_index]
        display_single_example(current_sample, next_sample)
