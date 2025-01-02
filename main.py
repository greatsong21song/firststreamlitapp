import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(
    page_title="나의 첫 Streamlit 프로젝트",
    page_icon="📊",
    layout="wide"
)

# 타이틀과 소개
st.title('나의 첫 Streamlit 프로젝트')
st.write('다양한 Streamlit 요소들을 실험해보세요!')

# 사이드바 만들기
st.sidebar.header('사이드바 메뉴')
selected_menu = st.sidebar.selectbox(
    '원하는 데모를 선택하세요',
    ['기본 요소', '데이터 시각화', '인터랙티브 위젯']
)

# 기본 요소 섹션
if selected_menu == '기본 요소':
    st.header('기본 텍스트 요소')
    
    # 다양한 텍스트 스타일
    st.text('이것은 기본 텍스트입니다.')
    st.markdown('**이것은 마크다운 텍스트**입니다.')
    st.success('이것은 성공 메시지입니다!')
    st.error('이것은 에러 메시지입니다!')
    st.warning('이것은 경고 메시지입니다!')
    st.info('이것은 정보 메시지입니다!')
    
    # 코드 블록 표시
    st.code('''
    def hello_streamlit():
        print("Hello, Streamlit!")
    ''')

# 데이터 시각화 섹션
elif selected_menu == '데이터 시각화':
    st.header('간단한 데이터 시각화')
    
    # 샘플 데이터 생성
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    # 라인 차트
    st.subheader('라인 차트')
    st.line_chart(chart_data)
    
    # 바 차트
    st.subheader('바 차트')
    st.bar_chart(chart_data)

# 인터랙티브 위젯 섹션
else:
    st.header('인터랙티브 위젯')
    
    # 텍스트 입력
    user_input = st.text_input('이름을 입력하세요:', '')
    if user_input:
        st.write(f'안녕하세요, {user_input}님!')
    
    # 슬라이더
    age = st.slider('나이를 선택하세요:', 0, 100, 25)
    st.write(f'선택한 나이: {age}세')
    
    # 날짜 선택
    date = st.date_input('날짜를 선택하세요:', datetime.now())
    st.write(f'선택한 날짜: {date}')
    
    # 체크박스
    if st.checkbox('추가 정보 보기'):
        st.write('여기에 추가 정보가 표시됩니다!')

# 푸터
st.markdown('---')
st.markdown('Made with ❤️ by Streamlit')
