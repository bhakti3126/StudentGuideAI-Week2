import streamlit as st
from rag import ask_question

st.set_page_config(
    page_title="Student Guide AI",
    page_icon="📚"
)

st.title("📚 Student Guide AI")

st.write("Ask any question from your PDF notes.")

question = st.text_input("Enter your question:")

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Thinking..."):
            answer = ask_question(question)

        st.success("Answer")

        st.write(answer)