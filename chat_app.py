import streamlit as st
from chat_model import ChatModel
from dotenv import load_dotenv
load_dotenv()

st.title("WhatsAI")

chat_model = ChatModel()

# Initialize the message history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Hola"):
    # Append the new user message to the history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve only the last 10 messages from the history
    history = st.session_state.messages[-10:]

    # Generate assistant response using the chat model
    assistant_response = chat_model.generate_response(
        prompt=prompt,
        history=history,  # Pass only the last 10 messages for context
    )

    with st.chat_message("assistant"):
        # Display all generated responses
        for response in assistant_response:
            st.markdown(response)

    # Append the last assistant response to the message history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response[-1]})
