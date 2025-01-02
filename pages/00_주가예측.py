import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="주식 예측 시스템", layout="wide")
st.title("📈 주식 예측 및 투자 추천 시스템")

# 사이드바 설정
st.sidebar.header("설정")
stock_symbol = st.sidebar.text_input("주식 심볼 입력 (예: 005930.KS for Samsung Electronics)", "005930.KS")

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
    ["LSTM", "Linear Regression", "Prophet"],
    help="""
    LSTM: 복잡한 패턴을 학습할 수 있는 딥러닝 모델
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

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def lstm_predict(data, forecast_days, scaler):
    # LSTM 모델 생성
    seq_length = 10
    X, y = create_sequences(data, seq_length)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    # 예측
    last_sequence = data[-seq_length:]
    future_predictions = []
    
    for _ in range(forecast_days):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

def linear_regression_predict(data, forecast_days, scaler):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.reshape(-1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
    future_predictions = model.predict(future_X)
    
    return scaler.inverse_transform(future_predictions.reshape(-1, 1))

def prophet_predict(df, forecast_days):
    # Prophet 데이터 준비
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
    # 기본적인 투자 지표 계산
    returns = df['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # 연간 변동성
    sharpe_ratio = (returns.mean() * 252) / volatility  # Sharpe Ratio
    
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
        
        # 데이터 준비
        scaled_data, scaler = prepare_data(df)
        
        # 예측 수행
        if algorithm == "LSTM":
            predictions = lstm_predict(scaled_data, forecast_days, scaler)
        elif algorithm == "Linear Regression":
            predictions = linear_regression_predict(scaled_data, forecast_days, scaler)
        else:  # Prophet
            forecast = prophet_predict(df, forecast_days)
            predictions = forecast['yhat'].values.reshape(-1, 1)
        
        # 시각화
        fig = go.Figure()
        
        # 실제 데이터
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name="실제 가격",
            line=dict(color='blue')
        ))
        
        # 예측 데이터
        future_dates = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
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
        
        # 투자 지표 및 추천
        metrics = calculate_investment_metrics(df)
        recommendation, reason = get_investment_recommendation(metrics, predictions)
        
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
        
        # 알고리즘 설명
        st.sidebar.subheader("알고리즘 설명")
        if algorithm == "LSTM":
            st.sidebar.info("""
            LSTM(Long Short-Term Memory)은 딥러닝 기반의 시계열 예측 모델입니다.
            - 장점: 복잡한 패턴 학습 가능, 장기 의존성 포착
            - 단점: 많은 데이터 필요, 학습 시간이 김
            """)
        elif algorithm == "Linear Regression":
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
