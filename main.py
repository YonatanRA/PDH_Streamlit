# libraries
import streamlit as st
from chatbot import DesignHandbookBot
from tools import logger
import os
from dotenv import load_dotenv
load_dotenv()

def pdh_chatbot():

    # memory
    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    if 'chat' not in st.session_state:
            st.session_state.chat = DesignHandbookBot(thread_id=st.session_state.thread_id)

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


    # conversation
    if prompt := st.chat_input('Introduce tu pregunta...'):
    
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('user'):
            st.markdown(prompt)

        logger.info('Start conversation...')
        with st.chat_message('assistant'):
            with st.spinner('Esperando respuesta del DesignHandbookBot...'):
                message_placeholder = st.empty()  
                full_response = ''  
                
                
                for response in st.session_state.chat.invoke(prompt):
                
                    full_response += response
                    message_placeholder.markdown(full_response + '▌')  
                    
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
        logger.info('End conversation.')


# mendesaltaren image
st.sidebar.image('https://cdn.prod.website-files.com/62f3882bdf914586601e212a/6399f2839e8ccf9859da1874_6202b1a3f47605af39b227cf_AxenykX.png')

# user thread ids 
THREAD_IDS = os.getenv('THREAD_IDS').split(',')


# if not login...
if 'thread_id' not in st.session_state:

    # empty container
    placeholder = st.empty()

    # login form
    with placeholder.form('login'):
        st.markdown('Introduce tus credenciales')
        user = st.text_input('User')
        password = st.text_input('Password', type='password')
        submit = st.form_submit_button('Login')
        thread_id = user+password

    # if correct login..
    if submit and thread_id in THREAD_IDS:

        placeholder.empty()
        st.session_state.thread_id = thread_id
        pdh_chatbot()

    # login error
    elif submit and thread_id not in THREAD_IDS:
        st.error('Fallo del Login')

    else:
        pass

# continue app
else:
    pdh_chatbot()