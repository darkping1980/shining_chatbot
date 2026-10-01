import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="챗봇",
    page_icon="😂",
    layout="wide"
)

# 모델을 로드 -> 사용자마다 다를 필요가 없다 -> 캐시에 저장
# cache_data : 데이터(표, 숫자, 리스트) -> 미리 가져올 사용
# cache_resource : 모델, DB연결, 에이전트 와 같은 한번 만들어서 
# 계속 재사용하는 객체 -> resource 에 저장

st.title("나의 첫번째 챗봇")


@st.cache_resource
def get_model():
    return init_chat_model("openai:gpt-6-luna", 
                           reasoning_effort="none",
                           api_key=st.secrets["OPENAI_API_KEY"])

model = get_model()

# 대화 내용이 누적
if "messages" not in st.session_state:
    st.session_state['messages'] = []   # 초기화 

# 기존 대화가 있다면 ->
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input("무엇이든 물어보세요")

# 채팅을 입력하면 -> 모델에 요청이 들어가야 -> 답변을 받으면 출력
if question:    # 채팅을 입력하면
    with st.chat_message("user"):
        st.markdown(question)

    # 질문한 내역을 저장 -> messages 에 저장
    st.session_state['messages'].append({'role': 'user', 'content': question})

    # LLM 요청 및 응답
    with st.chat_message('assistant'):
        with st.spinner("답변하는 중...."):
            answer = model.invoke(st.session_state['messages'])
        st.markdown(answer.content)

        # LLM 응답을 messages 에 넣기
        st.session_state['messages'].append({'role': 'assistant', 'content': answer.content})
