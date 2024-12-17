import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(
    page_title="Streamlit",
    page_icon="🤖",
)


# chat = ChatOpenAI(
#     model_name="gpt-3.5-turbo",
#     streaming=True,
#     api_key=api_key,
#     callbacks=[
#         StreamingStdOutCallbackHandler(),
#     ],
# )
with st.sidebar:
    st.text_input("Write down your OpenAI key", placeholder="k-proj-NDE*********")