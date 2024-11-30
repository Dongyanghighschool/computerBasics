import streamlit as st
import pandas as pd
import random
from openai import OpenAI

@st.cache_data
def load_data():
    df = pd.read_csv('./final_test.csv')
    return df

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def initialize_quiz_state(df):
    # 새로운 문제를 로드하고 상태를 초기화
    row = df.sample(n=1).iloc[0]
    st.session_state['question'] = row['문제']
    st.session_state['correct_answer'] = row['답']

    # 선택지 생성
    options = [st.session_state['correct_answer']]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "당신은 오답생성 전문가입니다."
            },
            {
                "role": "user",
                "content": f"""다음 문제에 대한 정답과 유사하지만 잘못된 선택지 3개를 생성해 주세요.
                               문제: {st.session_state['question']}
                               정답: {st.session_state['correct_answer']}
                               오답 선택지: """
            }
        ]
    )
    # API로부터 응답을 받아 오답 선택지를 생성
    wrong_answers = response.choices[0].message.content
    import re
    wrong_answers_list = [re.sub(r'^\d+\.\s*', '', ans).strip() for ans in wrong_answers.strip().split('\n')]
    options.extend(wrong_answers_list)
    random.shuffle(options)

    # 상태 초기화
    st.session_state['options'] = options
    st.session_state['answered'] = False
    st.session_state['show_result'] = False
    st.session_state['result_handled'] = False  # 해설이 한 번만 출력되도록 관리

def handle_result():
    if not st.session_state['result_handled']:
        # 해설 한 번만 출력
        if st.session_state['user_choice'] == st.session_state['correct_answer']:
            st.success("정답입니다!")
        else:
            st.error("오답입니다!")

        # GPT를 통해 해설 생성
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o"

        prompt = f"문제: {st.session_state['question']}\n정답: {st.session_state['correct_answer']}\n"
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            if 'messages' not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # 해설 출력 완료 표시
        st.session_state['result_handled'] = True

def main():
    st.title("컴퓨터 개론 만점 가즈아!!")
    df = load_data()

    if 'quiz_started' not in st.session_state:
        st.session_state['quiz_started'] = False
        st.session_state['answered'] = False
        st.session_state['show_result'] = False
        st.session_state['result_handled'] = False

    if not st.session_state['quiz_started']:
        if st.button("시작"):
            st.session_state['quiz_started'] = True
            initialize_quiz_state(df)
            st.experimental_rerun()
    else:
        # 문제와 보기 항상 표시
        st.subheader("문제:")
        st.write(st.session_state['question'])
        st.radio("보기 중에서 선택하세요:", options=st.session_state['options'], key="user_choice", index=0)

        # 결과 표시
        if st.session_state['show_result']:
            # 해설과 문제 화면이 사라지지 않도록 구성
            st.subheader("결과:")
            handle_result()

        # 제출 버튼
        if not st.session_state['show_result']:
            if st.button("제출", key="submit"):
                st.session_state['answered'] = True
                st.session_state['show_result'] = True
                st.experimental_rerun()

        # "다른 문제 풀기" 버튼
        if st.session_state['show_result']:
            st.divider()
            if st.button("다른 문제 풀기", key="next_question"):
                initialize_quiz_state(df)
                st.experimental_rerun()

if __name__ == "__main__":
    main()