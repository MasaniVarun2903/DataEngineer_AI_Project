"""
STEP 5: The chat website - what a client (and your company laptop, once
deployed) will actually open in a browser.
Run locally: streamlit run app.py
"""

import streamlit as st
from agent_groq import ask_agent

st.set_page_config(page_title="Ask Your Sales Data", page_icon="📊")
st.title("📊 Ask Your Sales Data — AI Agent")
st.write("Ask questions about retail sales in plain English. Example: "
         "*How did Detergent sales do in South region?*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the data..."):
            answer = ask_agent(question)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})