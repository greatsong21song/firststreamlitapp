import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="주식 예측 시스템", layout="wide")
st.title("📈 주식 예측 및 투자 추천 시스템")

# 한국 주식 종목 정보 가져오기
@st.cache_data
def get_korean_stocks():
    # KOSPI 종목 가져오기
    kospi_url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=stockMkt"
    kospi_df = pd.read_html(kospi_url)[0]
    kospi_df['시장구분'] = 'KOSPI'
    
    # KOSDAQ 종목 가져오기
    kosdaq_url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=kosdaqMkt"
    kosdaq_df = pd.read_html(kosdaq_url)[0]
    kosdaq_df['시장구분'] = 'KOSDAQ'
    
    # 데이터 합치기
    stocks_df = pd.concat([kospi_df, kosdaq_df])
    
    # 필요한 컬럼만 선택하고 이름 변경
    stocks_df = stocks_df[['회사명', '종목코드', '시장구분']]
    stocks_df['종목코드'] = stocks_df['종목코드'].astype(str).str.zfill(6)
    
    # Yahoo Finance 심볼 형식으로 변환 (.KS for KOSPI, .KQ for KOSDAQ)
    stocks_df['야후코드'] = stocks_df.apply(
        lambda x: f"{x['종목코드']}.KS" if x['시장구분'] == 'KOSPI' else f"{x['종목코드']}.KQ", 
        axis=1
    )
    
    return stocks_df

# 주식 데이터 검색 함수 (자동완성 강화 버전)
def search_stocks(search_text, stocks_df):
    if not search_text:
        return stocks_df
    
    # 초성 변환 함수
    def get_chosung(text):
        chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        code_list = [ord(ch) - ord('가') for ch in text]
        cho_list = []
        for code in code_list:
            if 0 <= code <= 11171:  # 한글 범위
                cho_list.append(chosung_list[code // 588])
            else:
                cho_list.append(ch)
        return ''.join(cho_list)
    
    # 검색어 처리
    search_text = search_text.upper()
    search_chosung = get_chosung(search_text)
    
    # 검색 조건
    conditions = (
        # 회사명에 검색어가 포함된 경우 (대소문자 구분 없음)
        stocks_df['회사명'].str.contains(search_text, case=False) |
        # 종목코드에 검색어가 포함된 경우
        stocks_df['종목코드'].str.contains(search_text) |
        # 회사명의 초성이 검색어와 일치하는 경우
        stocks_df['회사명'].apply(lambda x: get_chosung(x)).str.contains(search_chosung) |
        # 영문 회사명에 검색어가 포함된 경우
        stocks_df['영문명'].str.contains(search_text, case=False)
    )
    
    # 정확도에 따른 가중치 부여
    matches = stocks_df[conditions].copy()
    if not matches.empty:
        matches['정확도'] = 0
        # 정확한 종목코드 일치
        matches.loc[matches['종목코드'] == search_text, '정확도'] += 100
        # 회사명 시작 부분 일치
        matches.loc[matches['회사명'].str.startswith(search_text, na=False), '정확도'] += 50
        # 부분 일치
        matches.loc[matches['회사명'].str.contains(search_text, case=False, na=False), '정확도'] += 30
        # 초성 일치
        matches.loc[matches['회사명'].apply(lambda x: get_chosung(x)).str.startswith(search_chosung, na=False), '정확도'] += 20
        
        return matches.sort_values('정확도', ascending=False)
    
    return matches

try:
    stocks_df = get_korean_stocks()
    
    # 사이드바 설정
    st.sidebar.header("종목 선택")
    
    search_text = st.sidebar.text_input("회사명 또는 종목코드 검색")
    filtered_stocks = search_stocks(search_text, stocks_df)
    
    if not filtered_stocks.empty:
        selected_stock = st.sidebar.selectbox(
            "종목 선택",
            filtered_stocks.apply(lambda x: f"{x['회사명']} ({x['종목코드']} - {x['시장구분']})", axis=1),
            index=0
        )
        
        # 선택된 종목의 야후 코드 찾기
        selected_stock_code = selected_stock.split('(')[1].split(')')[0].split(' - ')[0]
        stock_symbol = stocks_df[stocks_df['종목코드'] == selected_stock_code]['야후코드'].iloc[0]
    else:
        st.sidebar.warning("검색 결과가 없습니다.")
        stock_symbol = None

    st.sidebar.header("설정")
    # 날짜 범위 설정
    today = datetime.now()
    default_start = today - timedelta(days=365)
    max_date = today + timedelta(days=365)

    start_date = st.sidebar.date_input("시작일", default_start)
    end_date = st.sidebar.date_input("종료일", today)
    forecast_days = st.sidebar.slider("예측 기간 (일)", 1, 365, 30)

    # 알고리즘 선택
    algorithm = st.sidebar.selectbox(
        "예측 알고리즘 선택",
        ["Linear Regression", "Prophet"],
        help="""
        Linear Regression: 단순하지만 안정적인 선형 예측 모델
        Prophet: 계절성을 고려한 페이스북의 시계열 예측 모델
        """
    )

    @st.cache_data
    def load_data(symbol, start, end):
        try:
            df = yf.download(symbol, start=start, end=end)
            return df
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {e}")
            return None

    def prepare_data(df, target_col='Close'):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[[target_col]])
        return scaled_data, scaler

    def linear_regression_predict(data, forecast_days, scaler):
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.reshape(-1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        
        return scaler.inverse_transform(future_predictions.reshape(-1, 1))

    def prophet_predict(df, forecast_days):
        df_prophet = pd.DataFrame({
            'ds': df.index,
            'y': df['Close']
        })
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future_dates = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future_dates)
        
        return forecast.tail(forecast_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def calculate_investment_metrics(df):
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'current_price': df['Close'].iloc[-1],
            'price_change': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        }

    def get_investment_recommendation(metrics, forecast_prices):
        current_price = metrics['current_price']
        last_forecast_price = forecast_prices[-1]
        expected_return = (last_forecast_price - current_price) / current_price * 100
        
        if expected_return > 10 and metrics['sharpe_ratio'] > 0.5:
            return "강력 매수", "높은 수익이 예상되며, 위험 대비 수익률이 양호합니다."
        elif expected_return > 5:
            return "매수", "적정한 수익이 예상됩니다."
        elif expected_return > -5:
            return "관망", "큰 가격 변동이 예상되지 않습니다."
        else:
            return "매도", "가격 하락이 예상됩니다."

    # 메인 로직
    if stock_symbol:
        df = load_data(stock_symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            st.subheader("주가 데이터")
            st.dataframe(df.tail())
            
            scaled_data, scaler = prepare_data(df)
            
            if algorithm == "Linear Regression":
                predictions = linear_regression_predict(scaled_data, forecast_days, scaler)
                future_dates = pd.date_range(
                    start=df.index[-1] + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
            else:  # Prophet
                forecast = prophet_predict(df, forecast_days)
                predictions = forecast['yhat'].values.reshape(-1, 1)
                future_dates = forecast['ds'].values
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                name="실제 가격",
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions.flatten(),
                name="예측 가격",
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="주가 예측 차트",
                xaxis_title="날짜",
                yaxis_title="가격",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            metrics = calculate_investment_metrics(df)
            recommendation, reason = get_investment_recommendation(
                metrics, 
                predictions if algorithm == "Linear Regression" else forecast['yhat'].values
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("현재 가격", f"{metrics['current_price']:,.0f}")
            with col2:
                st.metric("가격 변동률", f"{metrics['price_change']:.1f}%")
            with col3:
                st.metric("변동성", f"{metrics['volatility']:.2f}")
            with col4:
                st.metric("샤프 비율", f"{metrics['sharpe_ratio']:.2f}")
            
            st.subheader("투자 추천")
            st.write(f"**추천**: {recommendation}")
            st.write(f"**이유**: {reason}")
            
            st.sidebar.subheader("알고리즘 설명")
            if algorithm == "Linear Regression":
                st.sidebar.info("""
                선형 회귀는 가장 기본적인 예측 모델입니다.
                - 장점: 단순하고 안정적, 해석이 쉬움
                - 단점: 복잡한 패턴 포착 어려움
                """)
            else:
                st.sidebar.info("""
                Prophet은 Facebook이 개발한 시계열 예측 모델입니다.
                - 장점: 계절성 고려, 이상치에 강건
                - 단점: 급격한 변화 대응 어려움
                """)
        else:
            st.error("유효한 주식 데이터를 불러올 수 없습니다.")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
    st.info("잠시 후 다시 시도해주세요.")
