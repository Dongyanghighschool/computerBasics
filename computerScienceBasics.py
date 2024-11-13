import streamlit as st  # Streamlit 라이브러리를 st로 임포트하여 웹 애플리케이션을 구축할 수 있게 합니다.
import pandas as pd     # Pandas 라이브러리를 pd로 임포트하여 데이터 처리를 수행합니다.
import random           # random 모듈을 임포트하여 무작위 선택 기능을 제공합니다.
from openai import OpenAI  # OpenAI 라이브러리에서 OpenAI 클래스를 임포트하여 GPT 모델을 사용할 수 있게 합니다.

# CSV 파일을 로드하고 데이터 캐싱을 처리하는 함수 정의
@st.cache_data
def load_data():
    # 지정된 경로에서 CSV 파일을 읽어들입니다. 실제 파일 경로로 변경해야 합니다.
    df = pd.read_csv('./final_test.csv')
    return df  # 데이터프레임을 반환합니다.

# OpenAI API 클라이언트를 설정합니다.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Streamlit 시크릿을 사용하여 API 키를 안전하게 불러옵니다.

def main():
    st.title("컴퓨터 개론 만점 가즈아!!")  # 웹 앱의 제목을 설정합니다.

    df = load_data()  # CSV 파일에서 데이터를 로드합니다.

    # 세션 상태 변수를 초기화합니다.
    if 'quiz_started' not in st.session_state:
        st.session_state['quiz_started'] = False  # 퀴즈 시작 여부를 저장합니다.
    if 'question' not in st.session_state:
        st.session_state['question'] = None  # 현재 문제를 저장합니다.
    if 'options' not in st.session_state:
        st.session_state['options'] = None  # 선택지를 저장합니다.
    if 'correct_answer' not in st.session_state:
        st.session_state['correct_answer'] = None  # 정답을 저장합니다.
    if 'user_choice' not in st.session_state:
        st.session_state['user_choice'] = None  # 사용자의 선택을 저장합니다.
    if 'explanation' not in st.session_state:
        st.session_state['explanation'] = None  # 해설을 저장합니다.
    if 'answered' not in st.session_state:
        st.session_state['answered'] = False  # 답변 제출 여부를 저장합니다.

    # 시작 버튼이 눌리지 않은 경우
    if not st.session_state['quiz_started']:
        if st.button("시작", key='start_button'):  # '시작' 버튼을 생성하고 클릭을 감지합니다.
            st.session_state['quiz_started'] = True  # 퀴즈 시작 상태를 True로 변경합니다.
            st.session_state['answered'] = False  # 답변 제출 상태를 초기화합니다.

            # 랜덤으로 문제를 선택합니다.
            row = df.sample(n=1).iloc[0]  # 데이터프레임에서 무작위로 한 행을 선택합니다.
            question = row['문제']  # '문제' 열에서 질문을 가져옵니다.
            correct_answer = row['답']  # '답' 열에서 정답을 가져옵니다.

            # 선택지 생성 (정답 포함)
            options = [correct_answer]  # 선택지 리스트에 정답을 추가합니다.
            # 정답을 제외한 다른 답변 중 3개를 무작위로 선택합니다.

            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role" : "system",
                                        "content" : "당신은 오답생성 전문가입니다."
                                    },{
                                        "role" : "user",
                                        "content" : f"""다음 문제에 대한 정답과 유사하지만 잘못된 선택지 3개를 생성해 주세요.
                                                       문제: {question}
                                                       정답: {correct_answer}
                                                       오답 선택지: """
                                    }]
                            )
            # API로부터 응답 받기
            wrong_answers = response.choices[0].message.content

            # 문자열을 오답 리스트로 분할
            wrong_answers_list = wrong_answers.strip().split('\n')

            # 각 오답 정제 (번호 제거, 공백 제거 등)
            import re
            wrong_answers_list = [re.sub(r'^\d+\.\s*', '', ans).strip() for ans in wrong_answers_list]

            # 선택지 리스트에 오답 추가
            options.extend(wrong_answers_list)
            
            random.shuffle(options)  # 선택지의 순서를 무작위로 섞습니다.
            st.write(options)



            # 세션 상태에 문제와 관련 정보를 저장합니다.
            st.session_state['question'] = question
            st.session_state['options'] = options
            st.session_state['correct_answer'] = correct_answer
            st.session_state['user_choice'] = None
            st.session_state['explanation'] = None

            # 화면을 갱신하여 변경 사항을 반영합니다.
            st.experimental_rerun()
    else:
        # 문제와 선택지를 화면에 표시합니다.
        st.subheader("문제:")
        st.write(st.session_state['question'])

        # 답변을 제출하기 전 상태
        if not st.session_state['answered']:
            # 라디오 버튼을 사용하여 선택지를 제공합니다.
            if st.session_state['options'] is not None:
                st.session_state['user_choice'] = st.radio("보기 중에서 선택하세요:", options = st.session_state['options'])
            else:
                st.error("선택지를 불러오는 데 문제가 발생했습니다. 다시 시도해주세요.")

            # '제출' 버튼이 눌렸을 때
            if st.button("제출", key='submit_button'):
                st.session_state['answered'] = True  # 답변 제출 상태를 True로 변경합니다.

                if st.session_state['user_choice'] == st.session_state['correct_answer']:
                    st.success("정답입니다!")  # 정답인 경우 메시지를 표시합니다.
                    if "openai_model" not in st.session_state:
                        st.session_state["openai_model"] = "gpt-4o"

                    # React to user input

                    with st.chat_message("user"):
                            st.markdown(f"문제: {st.session_state['question']}\n")
                            st.markdown(f"정답: {st.session_state['correct_answer']}\n")
                    

                    # React to user input
                    prompt = f"문제: {st.session_state['question']}\n정답: {st.session_state['correct_answer']}\n"  # GPT에게 전달할 프롬프트를 생성합니다.

                    if prompt:
                        # ======== USER CONTAINER ========
                        # display user message in chat message container

                        # add message to chat history
                        if 'messages' not in st.session_state:
                            st.session_state.messages = []
                        st.session_state.messages.append({'role': 'user', 'content': prompt})

                        # ======== ASSISTANT CONTAINER ========
                        # display assistant response in chat message container
                        with st.chat_message("assistant"):
                            stream = client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages  # focus on the loop, helps keep context by simulating a conversation.
                                ], 
                                stream=True,
                            )
                            response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})  

                    st.write(st.session_state['explanation'])  # 해설을 화면에 표시합니다.


                    st.write("다른 문제를 풀고 싶다면 아래 버튼을 클릭하세요.")
                else:
                    st.error("오답입니다.")  # 오답인 경우 메시지를 표시합니다.
                    # GPT를 사용하여 해설을 생성합니다.

                    if "openai_model" not in st.session_state:
                        st.session_state["openai_model"] = "gpt-4o"

                    # React to user input

                    with st.chat_message("user"):
                            st.markdown(f"문제: {st.session_state['question']}\n")
                            st.markdown(f"정답: {st.session_state['correct_answer']}\n")



                    prompt = f"문제: {st.session_state['question']}\n정답: {st.session_state['correct_answer']}\n"  # GPT에게 전달할 프롬프트를 생성합니다.

                    if prompt:
                        # ======== USER CONTAINER ========
                        # display user message in chat message container
                      
                        # add message to chat history
                        if 'messages' not in st.session_state:
                            st.session_state.messages = []
                        st.session_state.messages.append({'role': 'user', 'content': prompt})

                        # ======== ASSISTANT CONTAINER ========
                        # display assistant response in chat message container
                        with st.chat_message("assistant"):
                            stream = client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages  # focus on the loop, helps keep context by simulating a conversation.
                                ], 
                                stream=True,
                            )
                            response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})  

                    st.write(st.session_state['explanation'])  # 해설을 화면에 표시합니다.
                    st.write("다른 문제를 풀고 싶다면 아래 버튼을 클릭하세요.")
            # '다른 문제 풀기' 버튼이 눌렸을 때
            if st.button("다른 문제 풀기", key='next_question_button'):
                # 상태를 초기화하고 새로운 문제를 로드합니다.
                st.session_state['answered'] = False  # 답변 제출 상태를 초기화합니다.
                st.session_state['explanation'] = None  # 해설을 초기화합니다.

                # 새로운 문제를 랜덤으로 선택합니다.
                row = df.sample(n=1).iloc[0]
                question = row['문제']
                correct_answer = row['답']

                # 선택지를 생성합니다.
                options = [correct_answer]
                wrong_answers = df[df['답'] != correct_answer]['답'].sample(n=3).tolist()
                options.extend(wrong_answers)
                random.shuffle(options)

                # 세션 상태를 업데이트합니다.
                st.session_state['question'] = question
                st.session_state['options'] = options
                st.session_state['correct_answer'] = correct_answer
                st.session_state['user_choice'] = None
                st.session_state['explanation'] = None

                # 화면을 갱신하여 변경 사항을 반영합니다.
                st.experimental_rerun()

                # 화면을 갱신하여 상태 변화를 반영합니다.
                #st.experimental_rerun()
        else:
            # 답변 제출 후 결과와 해설을 표시합니다.
            if st.session_state['user_choice'] == st.session_state['correct_answer']:
                st.success("정답입니다!")  # 정답 메시지를 표시합니다.
            else:
                st.error("오답입니다.")  # 오답 메시지를 표시합니다.
                st.info("해설:")
                st.write(st.session_state['explanation'])  # 해설을 표시합니다.

            st.write("다른 문제를 풀고 싶다면 아래 버튼을 클릭하세요.")

            # '다른 문제 풀기' 버튼이 눌렸을 때
            if st.button("다른 문제 풀기", key='next_question_button'):
                # 상태를 초기화하고 새로운 문제를 로드합니다.
                st.session_state['answered'] = False  # 답변 제출 상태를 초기화합니다.
                st.session_state['explanation'] = None  # 해설을 초기화합니다.

                # 새로운 문제를 랜덤으로 선택합니다.
                row = df.sample(n=1).iloc[0]
                question = row['문제']
                correct_answer = row['답']

                # 선택지 생성 (정답 포함)
                options = [correct_answer]  # 선택지 리스트에 정답을 추가합니다.
                # 정답을 제외한 다른 답변 중 3개를 무작위로 선택합니다.

                response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role" : "system",
                                        "content" : "당신은 오답생성 전문가입니다."
                                    },{
                                        "role" : "user",
                                        "content" : f"""다음 문제에 대한 정답과 유사하지만 잘못된 선택지 3개를 생성해 주세요.
                                                       문제: {question}
                                                       정답: {correct_answer}
                                                       오답 선택지: """
                                    }]
                            )
                # API로부터 응답 받기
                wrong_answers = response.choices[0].message.content

                # 문자열을 오답 리스트로 분할
                wrong_answers_list = wrong_answers.strip().split('\n')

                # 각 오답 정제 (번호 제거, 공백 제거 등)
                import re
                wrong_answers_list = [re.sub(r'^\d+\.\s*', '', ans).strip() for ans in wrong_answers_list]

                # 선택지 리스트에 오답 추가
                options.extend(wrong_answers_list)
            
                random.shuffle(options)  # 선택지의 순서를 무작위로 섞습니다.
                st.write(options)

                # 세션 상태를 업데이트합니다.
                st.session_state['question'] = question
                st.session_state['options'] = options
                st.session_state['correct_answer'] = correct_answer
                st.session_state['user_choice'] = None
                st.session_state['explanation'] = None

                # 화면을 갱신하여 변경 사항을 반영합니다.
                st.experimental_rerun()

if __name__ == "__main__":
    main()  # 메인 함수를 호출하여 웹 앱을 실행합니다.
