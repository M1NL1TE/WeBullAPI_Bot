# Import necessary libraries
import numpy as np
import pandas as pd
#from webull import paper_webull # import webull library
from my_webull import my_webull
from my_webull import my_paper_webull
import talib
import logging
import math
from pytz import timezone
import time
from datetime import datetime, timedelta, time as dt_time
import os
from typing import List
from PyInquirer import prompt, Validator, ValidationError
import site
from twilio.rest import Client #SEND SMS
#print(site.getsitepackages())

#SMS Setup Twilio
#CODE FOR SMS - TWILIO
# Your Account SID from twilio.com/console
account_sid = "AC54c3859d3ced571ac797753e65d8bd47"
# Your Auth Token from twilio.com/console
auth_token  = "e1361a19022e1928c91a626a65b0499f"

client = Client(account_sid, auth_token)
# configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('mylog.log'),
        logging.StreamHandler()
    ]
)

class CredentialsValidator(Validator):
    def validate(self, document):
        if not document.text:
            raise ValidationError(message="Please enter a value", cursor_position=0)

def get_credentials() -> dict:
    questions = [
        {
            "type": "input",
            "name": "username",
            "message": "Enter your Webull username:",
            "validate": CredentialsValidator,
        },
        {
            "type": "password",
            "name": "password",
            "message": "Enter your Webull password:",
            "validate": CredentialsValidator,
        },
    ]
    answers = prompt(questions)
    return answers

trade_grades = {'A+': {'stop_loss_pct': 0.50, 'trade_pct': 0.2},
                'A': {'stop_loss_pct': 0.47, 'trade_pct': 0.18},
                'B+': {'stop_loss_pct': 0.44, 'trade_pct': 0.15},
                'B': {'stop_loss_pct': 0.41, 'trade_pct': 0.15},
                'C+': {'stop_loss_pct': 0.38, 'trade_pct': 0.12},
                'C': {'stop_loss_pct': 0.35, 'trade_pct': 0.10},
                'D+': {'stop_loss_pct': 0.30, 'trade_pct': 0.05},
                'D': {'stop_loss_pct': 0.25, 'trade_pct': 0.05}}


def get_rsi_grade(df):
    rsi = talib.RSI(df['close'])
    df_last = df.tail(1)
    if rsi.iloc[-1] <= 30 and df_last['close'].iloc[-1] > df_last['open'].iloc[-1]:
        return 'A+'
    if rsi.iloc[-1] >= 70 and df_last['close'].iloc[-1] < df_last['open'].iloc[-1]:
        return 'A+'
    if (rsi.iloc[-2] > rsi.iloc[-1] and rsi.iloc[-1] < 30) or (rsi.iloc[-2] < rsi.iloc[-1] and rsi.iloc[-1] > 70):
        return 'A'
    return 'D'


def get_macd_grade(df):
    macd, macdsignal, macdhist = talib.MACD(df['close'])
    if macd.iloc[-1] > macdsignal.iloc[-1] and macd.iloc[-2] < macdsignal.iloc[-2]:
        return 'A+'
    return 'D'


def get_candle_grade(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    candlesticks = {}
    for candle in candle_names:
        candlesticks[candle] = getattr(talib, candle)(df['open'], df['high'], df['low'], df['close'])
    if 'CDL3LINESTRIKE' in candlesticks and candlesticks['CDL3LINESTRIKE'][-1] != 0 or 'CDLKICKINGBYLENGTH' in candlesticks and candlesticks['CDLKICKINGBYLENGTH'][-1] != 0:
        return 'A+'
    if 'CDLEVENINGDOJISTAR' in candlesticks and candlesticks['CDLEVENINGDOJISTAR'][-1] != 0 or 'CDLEVENINGSTAR' in candlesticks and candlesticks['CDLEVENINGSTAR'][-1] != 0 or 'CDLGRAVESTONEDOJI' in candlesticks and candlesticks['CDLGRAVESTONEDOJI'][-1] != 0 or 'CDLHAMMER' in candlesticks and candlesticks['CDLHAMMER'][-1] != 0 or 'CDLHANGINGMAN' in candlesticks and candlesticks['CDLHANGINGMAN'][-1] != 0:
        return 'A'
    if 'CDLMORNINGDOJISTAR' in candlesticks and 'CDLBULLISHHARAMI' in candlesticks:
        return 'B+'
    if 'CDLMORNINGDOJISTAR' in candlesticks or 'CDLBULLISHHARAMI' in candlesticks:
        return 'B'
    if 'CDLBELTHOLD' in candlesticks and 'CDLRISEFALL3METHODS' in candlesticks:
        return 'C+'
    if 'CDLBELTHOLD' in candlesticks or 'CDLRISEFALL3METHODS' in candlesticks:
        return 'C'
    return 'D'

'''
def get_volume_profile_grade(df):
    volume_profile = talib.volume_profile(df['close'], df['volume'], nbins=20)
    if volume_profile[15] > volume_profile[10] > volume_profile[5] and volume_profile[15] > volume_profile[14] > volume_profile[13] and volume_profile[15] > volume_profile[16] > volume_profile[17]:
        return 'A+'
    elif volume_profile[15] > volume_profile[10] > volume_profile[5] and volume_profile[15] > volume_profile[14] > volume_profile[13]:
        return 'A'
    elif volume_profile[15] > volume_profile[10] > volume_profile[5] and volume_profile[15] > volume_profile[16] > volume_profile[17]:
        return 'B+'
    elif volume_profile[15] > volume_profile[10] > volume_profile[5]:
        return 'B'
    elif volume_profile[15] > volume_profile[14] > volume_profile[13] and volume_profile[15] > volume_profile[16] > volume_profile[17]:
        return 'C+'
    elif volume_profile[15] > volume_profile[14] > volume_profile[13]:
        return 'C'
    else:
        return 'D'
'''

def get_volume_candle_grade(df):
    df_last = df.tail(3)
    if df_last['volume'].iloc[-1] > df_last['volume'].iloc[-2] and df_last['close'].iloc[-1] > df_last['open'].iloc[-1]:
        return 'A+'
    elif df_last['volume'].iloc[-1] > df_last['volume'].iloc[-2]:
        return 'A'
    elif df_last['close'].iloc[-1] > df_last['open'].iloc[-1]:
        return 'B+'
    elif df_last['volume'].iloc[-1] > df_last['volume'].iloc[-3] and df_last['close'].iloc[-1] > df_last['open'].iloc[-1]:
        return 'B'
    elif df_last['volume'].iloc[-1] > df_last['volume'].iloc[-3]:
        return 'C+'
    elif df_last['close'].iloc[-1] > df_last['open'].iloc[-1]:
        return 'C'
    else:
        return 'D'


def rsi_agreeance(df, timeframe):
    rsi = talib.RSI(df['close'], timeperiod=14)
    timeframe_count = int(timeframe[:-3])
    timeframe_unit = timeframe[-3:]
    if timeframe_unit == 'min':
        resampled = rsi.resample(f'{timeframe_count}T').last().dropna()
    elif timeframe_unit == 'h':
        resampled = rsi.resample(f'{timeframe_count}H').last().dropna()
    elif timeframe_unit == 'd':
        resampled = rsi.resample(f'{timeframe_count}D').last().dropna()
    else:
        raise ValueError('Invalid timeframe')

    if len(resampled) < 2:
        return False

    prev_rsi = resampled.iloc[-2]
    curr_rsi = resampled.iloc[-1]

    if prev_rsi < 30 and curr_rsi >= 30:
        return True
    if prev_rsi > 70 and curr_rsi <= 70:
        return True
    return False

def macd_agreeance(df, timeframe):
    macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    timeframe_count = int(timeframe[:-3])
    timeframe_unit = timeframe[-3:]
    if timeframe_unit == 'min':
        resampled_hist = hist.resample(f'{timeframe_count}T').last().dropna()
    elif timeframe_unit == 'h':
        resampled_hist = hist.resample(f'{timeframe_count}H').last().dropna()
    elif timeframe_unit == 'd':
        resampled_hist = hist.resample(f'{timeframe_count}D').last().dropna()
    else:
        raise ValueError('Invalid timeframe')

    if len(resampled_hist) < 2:
        return False

    prev_hist = resampled_hist.iloc[-2]
    curr_hist = resampled_hist.iloc[-1]

    if prev_hist < 0 and curr_hist >= 0:
        return True
    if prev_hist > 0 and curr_hist <= 0:
        return True
    return False

def get_trade_grade(wb,df, lookback, ztd, zbd, zts, zbs):
    supply_demand_result = supply_demand(wb,df, lookback, ztd, zbd, zts, zbs)
    rsi_grade = get_rsi_grade(df)
    macd_grade = get_macd_grade(df)
    candle_grade = get_candle_grade(df)
    #volume_profile_grade = get_volume_profile_grade(df)
    volume_candle_grade = get_volume_candle_grade(df)

    # Calculate the final trade grade
    timeframes = ['5min', '15min', '30min']
    timeframes_in_agreeance = []
    for tf in timeframes:
        rsi_in_agreeance = rsi_agreeance(df, tf)
        macd_in_agreeance = macd_agreeance(df, tf)
        if rsi_in_agreeance and macd_in_agreeance:
            timeframes_in_agreeance.append(tf)

    if supply_demand_result in ['in supply zone', 'in demand zone']:
        if timeframes_in_agreeance:
            if len(timeframes_in_agreeance) >= 2:
                if candle_grade in ['A+', 'A'] and volume_candle_grade in ['A+', 'A']:
                    trade_grade = 'A+'
                elif candle_grade in ['A+', 'A'] or volume_candle_grade in ['A+', 'A']:
                    trade_grade = 'A'
                elif candle_grade in ['B+', 'B'] and volume_candle_grade in ['B+', 'B']:
                    trade_grade = 'B+'
                elif candle_grade in ['B+', 'B'] or volume_candle_grade in ['B+', 'B']:
                    trade_grade = 'B'
                else:
                    if supply_demand_result == 'not in a zone':
                        trade_grade = 'C+'
                    else:
                        trade_grade = 'D'
            else:
                if supply_demand_result == 'not in a zone':
                    trade_grade = 'C'
                else:
                    trade_grade = 'D'
        else:
            if supply_demand_result == 'not in a zone':
                trade_grade = 'C'
            else:
                trade_grade = 'D'
    else:
        if candle_grade in ['C+', 'C'] and volume_candle_grade in ['C+', 'C']:
            trade_grade = 'C+'
        elif candle_grade in ['C+', 'C'] or volume_candle_grade in ['C+', 'C']:
            trade_grade = 'C'
        else:
            trade_grade = 'D'
    
    return trade_grade







# Define function to check for bullish RSI divergence
def bullish_divergence(prices, lookback):
    close_prices = prices['close'].values  # Convert close prices to a numpy array
    rsi_values = talib.RSI(close_prices)
    diff = np.diff(close_prices)
    diff = np.concatenate([[0], diff])
    rsi_values = rsi_values[-lookback:]
    diff = diff[-lookback:]
    if np.any((diff < 0) & (rsi_values > 30)):
        return True
    return False


# Define function to check for bearish RSI divergence
def bearish_divergence(prices, lookback):
    close_prices = prices['close'].values
    rsi_values = talib.RSI(close_prices)
    diff = np.diff(close_prices)
    diff = np.concatenate([[0],diff])
    rsi_values = rsi_values[-lookback:]
    diff = diff[-lookback:]
    if np.any((diff > 0) & (rsi_values < 70)):
        return True
    return False

# Define function to check for supply/demand zones
def supply_demand(wb,prices, lookback, ztd, zbd, zts, zbs):
    # find the highest and lowest candles
    #prices_1d.iloc[:-1][prices_1d.iloc[:-1]['high'] == np.max(prices_1d.iloc[:-1]['high'][-lookback:])]
    highest_candle = prices.iloc[:-4][prices.iloc[:-4]['high'] == np.max(prices.iloc[:-4]['high'][-lookback:])]
    lowest_candle = prices.iloc[:-4][prices.iloc[:-4]['low'] == np.min(prices.iloc[:-4]['low'][-lookback:])]
    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(prices)
    print(f"Supply zone timestamp: {highest_candle.index[0]}")
    print(f"Demand zone timestamp: {lowest_candle.index[0]}")
    # find the candle before the highest and lowest candles
    idx_highest_candle = prices.index.get_loc(highest_candle.index[0])
    idx_lowest_candle = prices.index.get_loc(lowest_candle.index[0])
    
    '''
    if idx_highest_candle < 1 or idx_lowest_candle < 1:
        # not enough data to calculate zones
        return 'not in a zone', ztd, zbd, zts, zbs
        '''
    
    before_highest_candle = prices.iloc[idx_highest_candle-1]
    before_lowest_candle = prices.iloc[idx_lowest_candle-1]

    # calculate supply and demand zones based on the candles
    #SUPPLY
    if highest_candle['close'].iloc[0] <= highest_candle['open'].iloc[0]: #red candle
        #zone_type_supply = 'supply'
        
        if idx_highest_candle < 1:
            zone_bottom_supply = highest_candle['close'].iloc[0]
            zone_top_supply = highest_candle['high'].iloc[0]
        elif before_highest_candle['close'] >= before_highest_candle['open'] and before_highest_candle['high'] > highest_candle['low'].iloc[0] and before_highest_candle['open'] < highest_candle['close'].iloc[0]: #before candle green
            zone_bottom_supply = before_highest_candle['open'] # use open of green candle before red canlde of supply zone
            zone_top_supply = highest_candle['high'].iloc[0]
        elif before_highest_candle['close'] <= before_highest_candle['open'] and before_highest_candle['high'] > highest_candle['low'].iloc[0] and before_highest_candle['close'] < highest_candle['close'].iloc[0]: #before candle red
            zone_bottom_supply = before_highest_candle['close']
            zone_top_supply = highest_candle['high'].iloc[0]
        else:
            zone_bottom_supply = highest_candle['close'].iloc[0]
            zone_top_supply = highest_candle['high'].iloc[0]
    elif highest_candle['close'].iloc[0] >= highest_candle['open'].iloc[0]: #green candle
        #zone_type_supply = 'supply'
        
        if idx_highest_candle < 1:
            zone_bottom_supply = highest_candle['open'].iloc[0]
            zone_top_supply = highest_candle['high'].iloc[0] #use open of green candle as bottom of supply zone
        elif before_highest_candle['close'] >= before_highest_candle['open'] and before_highest_candle['high'] > highest_candle['low'].iloc[0] and before_highest_candle['open'] < highest_candle['open'].iloc[0]: #before candle green
            zone_bottom_supply = before_highest_candle['open'] # use open of green candle before red canlde of supply zone
            zone_top_supply = highest_candle['high'].iloc[0]
        elif before_highest_candle['close'] <= before_highest_candle['open'] and before_highest_candle['high'] > highest_candle['low'].iloc[0] and before_highest_candle['close'] < highest_candle['open'].iloc[0]: #before candle red
            zone_bottom_supply = before_highest_candle['close']
            zone_top_supply = highest_candle['high'].iloc[0]
        else: #before candle is red
            zone_bottom_supply = highest_candle['open'].iloc[0]
            zone_top_supply = highest_candle['high'].iloc[0]
    else:
        #zone_type_supply = ''
        zone_top_supply = 0
        zone_bottom_supply = 0
    #DEMAND 
    if lowest_candle['close'].iloc[0] >= lowest_candle['open'].iloc[0]: #green candle
        #zone_type_demand = 'demand'
        
        if idx_lowest_candle < 1:
            zone_bottom_demand = lowest_candle['low'].iloc[0]
            zone_top_demand = lowest_candle['close'].iloc[0] 
        elif before_lowest_candle['close'] <= before_lowest_candle['open'] and before_lowest_candle['low'] < lowest_candle['high'].iloc[0] and before_lowest_candle['open'] > lowest_candle['close'].iloc[0]: # before candle is red
            zone_top_demand = before_lowest_candle['open']
            zone_bottom_demand = lowest_candle['low'].iloc[0]
        elif before_lowest_candle['close'] >= before_lowest_candle['open'] and before_lowest_candle['low'] < lowest_candle['high'].iloc[0] and before_lowest_candle['close'] > lowest_candle['close'].iloc[0]: #before candle is green
            zone_top_demand = before_lowest_candle['close']
            zone_bottom_demand = lowest_candle['low'].iloc[0]
        else:
            zone_bottom_demand = lowest_candle['low'].iloc[0]
            zone_top_demand = lowest_candle['close'].iloc[0]    
    elif lowest_candle['close'].iloc[0] <= lowest_candle['open'].iloc[0]: #red candle
        #zone_type_demand = 'demand'
        
        if idx_lowest_candle < 1:
            zone_bottom_demand = lowest_candle['low'].iloc[0]
            zone_top_demand = lowest_candle['open'].iloc[0] 
        elif before_lowest_candle['close'] <= before_lowest_candle['open'] and before_lowest_candle['low'] < lowest_candle['high'].iloc[0] and before_lowest_candle['open'] > lowest_candle['open'].iloc[0]: # before candle is red
            zone_top_demand = before_lowest_candle['open']
            zone_bottom_demand = lowest_candle['low'].iloc[0]
        elif before_lowest_candle['close'] >= before_lowest_candle['open'] and before_lowest_candle['low'] < lowest_candle['high'].iloc[0] and before_lowest_candle['close'] > lowest_candle['open'].iloc[0]: #before candle is green
            zone_top_demand = before_lowest_candle['close']
            zone_bottom_demand = lowest_candle['low'].iloc[0]
        else:
            zone_bottom_demand = lowest_candle['low'].iloc[0]
            zone_top_demand = lowest_candle['open'].iloc[0]
    else:
        zone_type_demand = ''
        zone_top_demand = 0
        zone_bottom_demand = 0
    
    ztd = zone_top_demand
    zbd = zone_bottom_demand
    zts = zone_top_supply
    zbs = zone_bottom_supply
    #return zone_top_demand, zone_bottom_demand, zone_top_supply, zone_bottom_supply
    # check if current price is in a zone
    # data = wb.get_bars(symbol, tId, interval, count, extendTrading=1)
    #    current_price = float(wb.get_quote(stock=symbol)['close'])

    prices_5m = wb.get_bars('SPY', wb.get_ticker('SPY'), 'm5', 10, 0)
    prices_1d = wb.get_bars('SPY', wb.get_ticker('SPY'), 'd1', 90, 0)

    #Excludes last row so the data isn't messed up on the daily timeframe
    highest_candle_1d = prices_1d.iloc[:-1][prices_1d.iloc[:-1]['high'] == np.max(prices_1d.iloc[:-1]['high'][-lookback:])]

    lowest_candle_1d = prices_1d.iloc[:-1][prices_1d.iloc[:-1]['low'] == np.min(prices_1d.iloc[:-1]['low'][-lookback:])]

    print(f"Last 5m closing price {prices_5m.iloc[-1]['close']}")

    if prices_5m.iloc[-1]['close'] >= highest_candle_1d['high'].iloc[-1]:
        return "new high, don't enter trade", ztd, zbd, zts, zbs
    elif prices_5m.iloc[-1]['close'] <= lowest_candle_1d['low'].iloc[-1]:
        return "new low, don't enter trade", ztd, zbd, zts, zbs

    elif prices_5m.iloc[-1]['close'] <= zone_top_supply and prices_5m.iloc[-1]['close'] >= zone_bottom_supply:
        return 'in supply zone', ztd, zbd, zts, zbs
    elif prices_5m.iloc[-1]['close'] >= zone_bottom_demand and prices_5m.iloc[-1]['close'] <= zone_top_demand:
        return 'in demand zone', ztd, zbd, zts, zbs
    else:
        return 'neutral', ztd, zbd, zts, zbs

#Edit 6/10/23 7:02PM - took out prices_one_minute because I will try to get the opening range from the 5 minute candles
#Notes for fix
'''
Possible fixes to orb strategy:

EDIT( 6/10/23) 11:11: it may be better to just always use the second last and third last candles for the direction variable.
Also a way to prevent long candles that go below or above the opening range candle (I.e. the 9:50/9:55 candle on 6/5/23)

1. for calls see if the direction is going down to show a bounce up
• a: using the last two candles see if the close of the second last candle [red] is >= the open of the last candle [green] (OMITTED)

OR
• b: using the second and third last candles see if the high close/open of the third last candle [green/red] is higher than the high of the second last candle [red]

2. For puts see if the direction is going down to show a bounce down
• a: using the last two candles see if the close of the second last candle [green] is <= the open of the last candle [red] (OMITTED)
OR
• b: using the second last and third last candles see if the low of the third last candle [green/red] is lower than the low of the second last candle [green]

3. Add condition to orb strategy for a way to prevent long candles that go below or above the opening range candle (I.e. the 9:50/9:55 candle on 6/5/23)
• for calls the last candle should not open lower than the open of the opening candle if it’s green or the close of the opening if it’s red
• For puts the last candle should not open higher than close of the opening candle if it’s green or the open of the opening if it’s red


'''
def orb_strategy(prices, orb_down,orb_up, orb_up_candle_time, orb_down_candle_time):
    
   # Get the current date
    current_date = datetime.now().date()
    #TEST CODE
    #current_date = datetime(2023, 6, 9).date()
    # Specify the start time and end time for the range
    opening_candle_time = datetime(current_date.year, current_date.month, current_date.day, hour=9, minute=35)
    #end_time = datetime(current_date.year, current_date.month, current_date.day, hour=9, minute=35)

    # Specify the timezone of the prices_one_minute index
    prices_timezone = timezone('America/New_York')

    # Localize start_time to the desired timezone
    opening_candle_time = opening_candle_time.astimezone(prices_timezone)

    # Localize end_time to the desired timezone
    #end_time = end_time.astimezone(prices_timezone)

    # Filter the prices_one_minute dataframe for the desired time range
    #Edit 6/10/23 7:02PM - get the first opening 5 minute candle
    prices_opening_candle = prices[prices.index == opening_candle_time]   
    '''Test code
    current_date = pd.Timestamp.now().floor('D')
    previous_date = current_date - pd.DateOffset(days=1)

# Set the market open time on the previous date
    market_open_time = previous_date + pd.to_timedelta('09:31:00') - pd.Timedelta(hours=4)


    # Find the index positions of the market open time and the next four minutes
    # Find the index of the nearest available timestamp to the market open time
    start_time = pd.Timestamp(year=2023, month=5, day=15, hour=9, minute=31)
    end_time = pd.Timestamp(year=2023, month=5, day=15, hour=9, minute=35)

    # Specify the timezone of the prices_one_minute index
    #prices_one_minute_timezone = timezone('America/New_York')

    # Specify the timezone of the prices_one_minute index
    # Specify the timezone of the prices_one_minute index
    prices_one_minute_timezone = timezone('America/New_York')

    # Localize start_time to the desired timezone
    start_time = start_time.tz_localize(prices_one_minute_timezone)

    # Localize end_time to the desired timezone
    end_time = end_time.tz_localize(prices_one_minute_timezone)

    # Filter the prices_one_minute dataframe for the desired time range
    filtered_prices_one_minute = prices_one_minute[(prices_one_minute.index >= start_time) & (prices_one_minute.index <= end_time)]
    '''



    

    # Calculate the opening range high and low
    opening_range_high = prices_opening_candle['high'].iloc[-1]
    opening_range_low = prices_opening_candle['low'].iloc[-1]
    print(f"ORB 5m Opening ranges\nHigh: {opening_range_high}\nLow:{opening_range_low}\n")
    # Get the current price and candle direction
    #
    candle_trend_up = False
    candle_trend_down = False
    current_price = prices['close'].iloc[-1]
    current_candle_direction = 'Green' if current_price > prices['open'].iloc[-1] else 'Red'
    previous_candle_direction = 'Green' if prices['close'].iloc[-2] > prices['open'].iloc[-2] else 'Red'
    third_candle_direction = 'Green' if prices['close'].iloc[-3] > prices['open'].iloc[-3] else 'Red'
    opening_candle_direction = 'Green' if prices_opening_candle['close'].iloc[-1] > prices_opening_candle['open'].iloc[-1] else 'Red'
    
    # Check if the current candle closed above or below the opening range
    if current_price > opening_range_high and not orb_up:
        if current_candle_direction == 'Green':
            orb_up = True
            # Add code here
            orb_up_candle_time = prices.index[-1]  # Get the timestamp of the current candle
            print("Waiting for retest of opening range high (Call)\n")
        else:
            print('No ORB break detected')

    if current_price < opening_range_low and not orb_down:
        if current_candle_direction == 'Red':
            orb_down = True
            # Add code here
            orb_down_candle_time = prices.index[-1]  # Get the timestamp of the current candle
            print("Waiting for retest of opening range low (Put)\n")
        else:
            print('No ORB break detected')
    
    # Check for retest of opening range
    #FIX THIS: There should be a flag variable to show if the range was already broken in a previous function call. Then this section can be entered
    #It should be a time after the first flag to prevent it from grabbing data from first break.
    

    
    
    if orb_up and prices.index[-1] > orb_up_candle_time:

        if third_candle_direction == 'Green' and previous_candle_direction == 'Red' and prices['close'].iloc[-3] >= prices['open'].iloc[-2]:
            candle_trend_down = True

        if third_candle_direction == 'Red' and previous_candle_direction == 'Red' and prices['high'].iloc[-3] > prices['high'].iloc[-2]:
            candle_trend_down = True
        

        
    #Should check for current candle touching the same line that was broken before. (i.e currnet_price['high'] >= opening_range high and current_price['low'] <= opening_range_high)
    
        if (
            current_candle_direction == 'Green' and previous_candle_direction == 'Red' and prices['high'].iloc[-1] > opening_range_high and 
            prices['low'].iloc[-1] < opening_range_high and 
            ((opening_candle_direction == 'Green' and prices['open'].iloc[-1] > prices_opening_candle['open'].iloc[-1]) or (opening_candle_direction == 'Red' and prices['open'].iloc[-1] > prices_opening_candle['close'].iloc[-1])) 
            and candle_trend_down
            ):
            return 'call', orb_down, orb_up, orb_up_candle_time, orb_down_candle_time
    #Edit 6/8/23 - Adding if statement so that it checks the high and low of the second last candle to see if it is within the orb breakout line
    # This is for cases when the previous candle may be within that line but the actual "breakout"candle misses it    
        if (
            current_candle_direction == 'Green' and previous_candle_direction == 'Red' and prices['high'].iloc[-2] > opening_range_high and 
            prices['low'].iloc[-2] < opening_range_high and 
            ((opening_candle_direction == 'Green' and prices['open'].iloc[-1] > prices_opening_candle['open'].iloc[-1]) or (opening_candle_direction == 'Red' and prices['open'].iloc[-1] > prices_opening_candle['close'].iloc[-1])) 
            and candle_trend_down
            ):
            return 'call', orb_down, orb_up, orb_up_candle_time, orb_down_candle_time
    
    if orb_down and prices.index[-1] > orb_down_candle_time:

        if third_candle_direction == 'Red' and previous_candle_direction == 'Green' and prices['close'].iloc[-3] <= prices['open'].iloc[-2]:
            candle_trend_up = True
        
        if third_candle_direction == 'Green' and previous_candle_direction == 'Green' and prices['low'].iloc[-3] < prices['low'].iloc[-2]:
            candle_trend_up = True
        
        if (current_candle_direction == 'Red' and previous_candle_direction == 'Green' and prices['high'].iloc[-1] > opening_range_low and 
            prices['low'].iloc[-1] < opening_range_low and 
            ((opening_candle_direction == 'Green' and prices['open'].iloc[-1] < prices_opening_candle['close'].iloc[-1]) or (opening_candle_direction == 'Red' and prices['open'].iloc[-1] < prices_opening_candle['open'].iloc[-1])) 
            and candle_trend_up):
            return 'put', orb_down, orb_up, orb_up_candle_time, orb_down_candle_time
    #Edit 6/8/23 - Adding if statement so that it checks the high and low of the second last candle to see if it is within the orb breakout line
    # This is for cases when the previous candle may be within that line but the actual "breakout"candle misses it      
        if (current_candle_direction == 'Red' and previous_candle_direction == 'Green' and prices['high'].iloc[-2] > opening_range_low and 
            prices['low'].iloc[-2] < opening_range_low and 
            ((opening_candle_direction == 'Green' and prices['open'].iloc[-1] < prices_opening_candle['close'].iloc[-1]) or (opening_candle_direction == 'Red' and prices['open'].iloc[-1] < prices_opening_candle['open'].iloc[-1])) 
            and candle_trend_up):
            return 'put', orb_down, orb_up, orb_up_candle_time, orb_down_candle_time
    
    # No trade signal
    
    return 'Hold', orb_down, orb_up, orb_up_candle_time, orb_down_candle_time


# Define function to check for bullish candlestick patterns
def bullish_candlestick(df, lookback):
    df = df.iloc[-lookback:]
    if talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])[-1] == 100:
        return True
    if talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])[-1] == 100:
        return True
    if talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])[-1] == 100:
        return True
    return False

# Define function to check for bearish candlestick patterns
def bearish_candlestick(df, lookback):
    df = df.iloc[-lookback:]
    if talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])[-1] == -100:
        return True
    if talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])[-1] == 100:
        return True
    if talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])[-1] == -100:
        return True
    return False

'''
# Define function to get real-time data from WeBull API
def get_data(wb: my_paper_webull, symbol: str, tId: int, interval: str, count: int, extendTrading: int) -> pd.DataFrame:
    # Get historical data
    data = wb.get_bars(symbol, tId, interval, count, extendTrading)


    
        get bars returns a pandas dataframe
        params:
            interval: m1, m5, m15, m30, m60, m120, m240, d1, w1
            count: number of bars to return
            extendTrading: change to 1 for pre-market and afterhours bars
            timeStamp: If epoc timestamp is provided, return bar count up to timestamp. If not set default to current time.
        
    # Filter data by start and end time
   # data = data.loc[start_time:end_time]

    return data
'''


#New Paste
'''Need to add option type parameter
   Need to add if statement for option type to cut operations in half for calls and puts'''
# Define function to filter SPY options based on strike price and volume
def filter_options(symbol, wb, option_type):
    # Get current date and day of the week
    current_date = datetime.now().date()
    current_day = current_date.weekday()

    # Determine minimum expiration date based on current day of the week
    if current_day == 4:  # Friday
        min_days = 3
    else:
        min_days = 1
    min_date = current_date + timedelta(days=min_days)

    # Get expiration dates for the specified stock
    expiration_dates = wb.get_options_expiration_dates(symbol)

    # Filter out expiration dates that are before the minimum date
    expiration_dates = [date for date in expiration_dates if datetime.strptime(date['date'], '%Y-%m-%d').date() >= min_date]

    # Get the first expiration date from the remaining dates
    expiration_date = expiration_dates[0]['date']

    # Get options contracts for the specified stock and expiration date
    #options = wb.get_options(stock=symbol, count=-1, direction='all', expireDate=expiration_date)
    
    current_price = float(wb.get_quote(stock=symbol)['close'])
    at_the_money_price = current_price
    rounded_at_the_money_price = round(at_the_money_price)
    #logging.info(options)
    if option_type == 'call':
        call_options = wb.get_options(stock=symbol, count=-1, direction='call', expireDate=expiration_date)
        call_options = [opt for opt in call_options if rounded_at_the_money_price <= float(opt['strikePrice']) <= rounded_at_the_money_price + 5]
        #logging.info(call_options)
        call_options.sort(key=lambda x: int(wb.get_option_quote(stock=symbol, optionId=x['call']['tickerId'])['data'][0]['volume']), reverse=True)
        #logging.info(call_options)

       # Check if the cheaper option is only 20,000 lower in volume then make it the first option in the list
        for i in range(len(call_options)):
            options = call_options[i:i+2]
            if len(options) == 2:
                first_option_price = float(wb.get_option_quote(stock=symbol, optionId=options[0]['call']['tickerId'])['data'][0]['askList'][0]['price'])
                second_option_price = float(wb.get_option_quote(stock=symbol, optionId=options[1]['call']['tickerId'])['data'][0]['askList'][0]['price'])
                if second_option_price < first_option_price and int(options[0]['call']['volume']) - int(options[1]['call']['volume']) <= 20000:
                    call_options[i], call_options[i+1] = call_options[i+1], call_options[i]
        #add code here to check if the 
        #logging.info(call_options)
        return call_options

                

    elif option_type == 'put':
        put_options = wb.get_options(stock=symbol, count=-1, direction='put', expireDate=expiration_date)
        put_options = [opt for opt in put_options if rounded_at_the_money_price - 5 <= float(opt['strikePrice']) <= rounded_at_the_money_price]
        put_options.sort(key=lambda x: int(wb.get_option_quote(stock=symbol, optionId=x['put']['tickerId'])['data'][0]['volume']), reverse=True)

        # Check if the cheaper option is only 20,000 lower in volume then make it the first option in the list
        for i, options in enumerate(put_options):
            if i > 0:
                first_option_price = float(wb.get_option_quote(stock=symbol, optionId=options['put']['tickerId'])['data'][0]['askList'][0]['price'])
                second_option_price = float(wb.get_option_quote(stock=symbol, optionId=put_options[i - 1]['put']['tickerId'])['data'][0]['askList'][0]['price'])
                if second_option_price < first_option_price and int(options['put']['volume']) - int(put_options[i - 1]['put']['volume']) <= 20000:
                    put_options[i], put_options[i - 1] = put_options[i - 1], put_options[i]

        #logging.info(put_options)
        return put_options

    # Filter out options with strike prices further than 5 strikes away from the current at the money price
    #Enter code here

        # Get current market price
    

    # Use current market price as at the money price
    

    # Filter out options with strike prices further than 5 strikes away from the current at the money price
    
    
   
    
    # Combine the call and put options back into one list
    #options = call_options + put_options

    # Filter options by highest volume first (not working)
    #options = sorted(options, key=lambda opt: opt['call']['volume'] + opt['put']['volume'], reverse=True)
    '''Get quotes of the remaining contracts depending on if they're puts or calls
        current_price_option = wb.get_option_quote(stock=symbol, optionId= 1038392923)

    '''

    
    #logging.info(current_price_option)
   
 # Get quotes of the remaining contracts depending on if they're puts or calls
    
    '''
    # Create dictionaries for call and put options with strike price and volume
    call_dict = {option['call']['strikePrice']: {'strikePrice': option['call']['strikePrice'], 'volume': int(wb.get_option_quote(stock=symbol, optionId=option['call']['tickerId'])['data'][0]['volume'])} for option in call_options}
    put_dict = {option['put']['strikePrice']: {'strikePrice': option['put']['strikePrice'], 'volume': int(wb.get_option_quote(stock=symbol, optionId=option['put']['tickerId'])['data'][0]['volume'])} for option in put_options}

    # Sort the call and put dictionaries by volume in descending order
    sorted_call = sorted(call_dict.values(), key=lambda x: x['volume'], reverse=True)
    sorted_put = sorted(put_dict.values(), key=lambda x: x['volume'], reverse=True)

    '''
    
    

    
    #def place_order_option(self, optionId=None, lmtPrice=None, stpPrice=None, action=None, orderType='LMT', enforce='DAY', quant=0):
    '''WORKING PAPER OPTIONS TRADE

    wb.place_order_option(optionId='1038464919', 
                          lmtPrice=wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price'],
                          stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price']) -0.25 ),
                          action='BUY',
                          orderType='LMT',
                          enforce='GTC',
                          quant=1)
    '''
                          
    '''
    wb.place_order_otoco_option(optionId='1038465080', price=wb.get_option_quote(stock=symbol, optionId=1038465080)['data'][0]['askList'][0]['price'],
                                 stop_loss_price=str(float(wb.get_option_quote(stock=symbol, optionId=1038465080)['data'][0]['askList'][0]['price']) -0.25 ), 
                                 limit_profit_price=str(float(wb.get_option_quote(stock=symbol, optionId=1038465080)['data'][0]['askList'][0]['price']) +0.25 ), time_in_force='DAY', quant=0) 

    '''
    
    
    '''
    wb.place_order_otoco_option(optionId='1038464919',
                                price=wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price'],
                                stop_loss_price=str(float(wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price']) -0.25 ),
                                limit_profit_price=str(float(wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price']) +0.25 ),
                                time_in_force='DAY',
                                 quant=1 )
    '''
    
    

    

'''
To do for buy_call/put:
take away filtered_chain and just use it for the place_order_otoco_option function


'''

# Define function to buy call option
def buy_call(wb, symbol, trade_grade, account_info, action = 'BUY'):
    # Get option chain for specified symbol and expiration date
    filtered_chain = filter_options(symbol, wb, 'call')

     # Calculate order quantity based on trade grade and account value
    account_value = float(account_info['accountMembers'][1]['value'])
    trade_pct = trade_grades[trade_grade]['trade_pct']
    option_price = wb.get_option_quote(stock=symbol, optionId=filtered_chain[0]['call']['tickerId'])['data'][0]['askList'][0]['price']
    order_quantity = float((account_value * trade_pct) / (float(option_price) * 100))
    order_quantity = math.floor(order_quantity)
    # Check if order_quantity is less than 1
    if order_quantity < 1:
        # Find an option with a price that allows order_quantity to be greater than 1
        for option in filtered_chain:
            option_price = wb.get_option_quote(stock=symbol, optionId=option['call']['tickerId'])['data'][0]['askList'][0]['price']
            new_order_quantity = (account_value * trade_pct) / (float(option_price) * 100)
            if new_order_quantity >= 1:
                # Move the current option to the first position in the list
                filtered_chain.remove(option)
                filtered_chain.insert(0, option)
                order_quantity = math.floor(new_order_quantity)
                break
        else:
            return None
    
    wb.place_order_option(optionId=filtered_chain[0]['call']['tickerId'], 
                        lmtPrice=option_price,
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=filtered_chain[0]['call']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action=action,
                        orderType='LMT',
                        enforce='GTC',
                        quant=order_quantity)
    message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body=(f"Took call order for {option_price}"))

    print(f"SID: {message.sid}")

def buy_put(wb,symbol, trade_grade, account_info):
    # Get option chain for specified symbol and expiration date
    filtered_chain = filter_options(symbol, wb, 'put')

     # Calculate order quantity based on trade grade and account value
    account_value = float(account_info['accountMembers'][1]['value'])
    trade_pct = trade_grades[trade_grade]['trade_pct']
    option_price = wb.get_option_quote(stock=symbol, optionId=filtered_chain[0]['put']['tickerId'])['data'][0]['askList'][0]['price']
    order_quantity = float((account_value * trade_pct) / (float(option_price) * 100))
    order_quantity = math.floor(order_quantity)
    # Check if order_quantity is less than 1
    if order_quantity < 1:
        # Find an option with a price that allows order_quantity to be greater than 1
        for option in filtered_chain:
            option_price = wb.get_option_quote(stock=symbol, optionId=option['put']['tickerId'])['data'][0]['askList'][0]['price']
            new_order_quantity = (account_value * trade_pct) / (float(option_price) * 100)
            if new_order_quantity >= 1:
                # Move the current option to the first position in the list
                filtered_chain.remove(option)
                filtered_chain.insert(0, option)
                order_quantity = math.floor(new_order_quantity)
                break
        else:
            return None

    # Use the floor function to ensure order_quantity is a whole number without decimal places
    
    #INSTRUCTIONS
    #add code here to check if calculation for order_quantity is greater than 1 float((account_value * trade_pct) / (float(option_price) * 100))
    #If it is less than 1 then it should find an option in the filtered_chain and find an option that has a price that will allow order_quantity to be greater than 1
    #The floor function should be used for order_quantity so that it's always a whole number without a decimal place or a function to just truncate the decimals and leave it as a whole number
    
    wb.place_order_option(optionId=filtered_chain[0]['put']['tickerId'], 
                        lmtPrice=option_price,
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=filtered_chain[0]['put']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='BUY',
                        orderType='LMT',
                        enforce='GTC',
                        quant=order_quantity)
    message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body=(f"Took put order for {option_price}"))

    print(f"SID: {message.sid}")

# Define function to execute trades based on signals
def execute_trades(data):
    # Check for bullish RSI divergence
    if bullish_divergence(data['close'], data['rsi'], 14):
        # Check for demand zone
        if supply_demand(data['close'], 10) == 'demand':
            # Check for bullish candlestick pattern
            if bullish_candlestick(data, 3):
                # Buy call option
                buy_call(symbol, '2023-06-16', 1.50, 434)
    
    # Check for bearish RSI divergence
    if bearish_divergence(data['close'], data['rsi'], 14):
        # Check for supply zone
        if supply_demand(data['close'], 10) == 'supply':
            # Check for bearish candlestick pattern
            if bearish_candlestick(data, 3):
                # Buy put option
                buy_put(symbol, '2023-06-16', 1.50, 434)

# Define main function

def main():
    # Define variables

    
    symbol = 'SPY'
    interval = 'm30'
    interval2 = '30m'
    expiration_date = ''
    option_type = 'call'
    ztd, zbd, zts, zbs = 0,0,0,0
    orb_up_time, orb_down_time = '', ''
    orb_break_upside, orb_break_downside = False,False
    confirm_past_orb = 0
    confirm_zone_order = 0
    zone_order_taken, orb_order_taken = False,False
    zone_broken = False
    zone_break_candle_demand_price, zone_break_candle_supply_price = 0,0
    # Create Webull instance and log in 
    
    #wb = my_webull()
    wb = my_webull()
    #wb.logout()
    wb._set_did = "97468314e9674207be4aa99b13f749c9"

    data = wb.login("<Email>","<pw>")
    ACCESS_TOKEN = data['access_token']
    REFRESH_TOKEN = data['refreshToken']
    TOKEN_EXPIRE = data['tokenExpireTime']
    UUID = data['uuid']
    login_data = wb.api_login(access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN, token_expire= TOKEN_EXPIRE, uuid=UUID)
    #old did - XXXX
    
   

    

    #wb._did = "XXXX"

 



    #wb.login_prompt()
    
    account_info = wb.get_account()
    logging.info(account_info)
    stockId = wb.get_ticker(symbol)
    
    trade_grade = ''
    
    
    #
    #
    #To do: extract supply and demand zones and use the 5 minute timeframe to see if it is in the range of a demand or supply zone
    #
    #
    #
    
    #bullish_divergence(df,100)
   # bearish_divergence(df,100)
    
    #
    #options = filter_options(symbol, wb, option_type)
    #logging.info(options)
    #df2 = wb.get_options_bars(options[0]['call']['tickerId'], interval2, 100, 0)
        
    #logging.info(df2)
    # Wait until market open
    
    
    
    #NEED TO UPDATE - GET_CALENDAR DOESNT GET DATA BEFORE MARKET OPEN
#While loop to print output before market open time of 9:30AM EST


#TEST FOR ORDERS (Update: test works now. Had to fix order quantity)
    '''
    df = wb.get_bars(symbol, stockId, interval, 90, 1)
    trade_grade = get_trade_grade(wb,df, 90, ztd, zbd, zts, zbs)
    buy_put(wb, symbol, trade_grade, account_info)
    trade_grade = get_trade_grade(wb,df, 90, ztd, zbd, zts, zbs)
    buy_call(wb, symbol, trade_grade, account_info)
    '''

    '''
    TEST WORKS

    test = "test message"
    message = client.messages.create(
                to="+<phone number>", 
                from_="+<Twilio Number>",
                body=(f"This is a {test}"))

    print(f"SID: {message.sid}")
    '''

    #df_five_minute = wb.get_bars(symbol, stockId, 'm5', 80)
                    #Edit 6/10/23 7:02PM - took out df_one_minute because I will try to get the opening range from the 5 minute candles
                    #df_one_minute = wb.get_bars(symbol, stockId, 'm1', 400,1)
   # orb_result, orb_break_downside, orb_break_upside, orb_up_time, orb_down_time = orb_strategy(df_five_minute, orb_break_downside, orb_break_upside, orb_up_time, orb_down_time)
    df = wb.get_bars(symbol, stockId, interval, 140, 1) # 1 is for extended trading hours
    zone_check, ztd, zbd, zts, zbs = supply_demand(wb,df, 140,ztd,zbd,zts,zbs)
    print(f"Demand zone: {zbd} - {ztd}\nSupply zone: {zbs} - {zts}\n{zone_check}\n")
    while (
    datetime.today().weekday() < 5  # Check if it's a weekday (Monday to Friday)
    and datetime.now().time() < datetime.strptime('9:30', '%H:%M').time()):  # Check if it's before 9:30 AM
        print('Waiting for market open...')
        time.sleep(60)
        
        
    # Check if the market is open
   
    while (datetime.today().weekday() < 5  # Check if it's a weekday (Monday to Friday)
    and datetime.now().time() > datetime.strptime('9:30', '%H:%M').time()
    and datetime.now().time() < datetime.strptime('16:00', '%H:%M').time()):
        #current_time = datetime.now().time()
        #target_time
        # Check if there are any current positions
        logging.info("Current Time\n")
        #only ask about orb if it's after the first possible breakout of the day on the 5m
        if datetime.now().time() < datetime.strptime('9:41', '%H:%M').time():
                confirm_past_orb = 2

        if confirm_zone_order <= 1 and account_info['positions']:
            confirm_zone_order = int(input("Is the current open order a zone order? (1 = Yes | 2 = No): "))

            if confirm_zone_order == 1:
                zone_order_taken = True
                confirm_zone_order = 2

            else:
                confirm_zone_order = 2
            

        if confirm_past_orb <= 1:
            confirm_past_orb = int(input("Do you need to confirm past ORB breakouts? (1 = Yes | 2 = No): "))

            if confirm_past_orb == 1:
                past_orb_up = int(input("Did an ORB already breakout on the upside? (1 = Yes | 2 = No): "))
                past_orb_down = int(input("Did an ORB already breakout on the downside? (1 = Yes | 2 = No): "))
                df_five_minute = wb.get_bars(symbol, stockId, 'm5', 2)
                if past_orb_up == 1:
                    orb_break_upside = True
                    orb_up_time = df_five_minute.index[-2]
                if past_orb_down == 1:
                    orb_break_downside = True
                    orb_down_time = df_five_minute.index[-2]

                confirm_past_orb = 2

        if not account_info['positions'] and not account_info['openOrders']:
            print("No positions. Looking for potential trade")
            df = wb.get_bars(symbol, stockId, interval, 140, 1) # 1 is for extended trading hours
            zone_check, ztd, zbd, zts, zbs = supply_demand(wb,df, 140,ztd,zbd,zts,zbs)
            print(f"Demand zone: {zbd} - {ztd}\nSupply zone: {zbs} - {zts}\n{zone_check}\n")

            
            

            #TEST CODE
            #trade_grade = get_trade_grade(wb,df, 110, ztd, zbd, zts, zbs)
            #buy_call(wb, symbol, trade_grade, account_info)
            #TEST CODE
            if zone_check == 'in supply zone':
             
                trade_grade = get_trade_grade(wb,df, 140, ztd, zbd, zts, zbs)
                buy_put(wb, symbol, trade_grade, account_info)
                confirm_zone_order = 2
                logging.info("Taking put supply zone trade:\n")
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Order reason: Supply zone")

                print(f"SID: {message.sid}")
                zone_order_taken = True
                
            elif zone_check == 'in demand zone':
            
                trade_grade = get_trade_grade(wb,df, 140, ztd, zbd, zts, zbs)
                logging.info("Taking call demand zone trade:\n")
                buy_call(wb, symbol, trade_grade, account_info)
                confirm_zone_order = 2
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Order reason: Demand Zone ")

                print(f"SID: {message.sid}")
                zone_order_taken = True
                
            else:
                # Check if the current time is after 9:36 AM and is a multiple of 5 minutes
                # Get the current time
                current_time_orb = datetime.now().time()
                    
                    # Define the target time (9:36 AM)
                target_time_orb = dt_time(9, 35)
                    
                if current_time_orb >= target_time_orb and current_time_orb.minute % 5 == 0:
            # Add your code here
                    df_five_minute = wb.get_bars(symbol, stockId, 'm5', 80)
                        #Edit 6/10/23 7:02PM - took out df_one_minute because I will try to get the opening range from the 5 minute candles
                        #df_one_minute = wb.get_bars(symbol, stockId, 'm1', 400,1)
                    orb_result, orb_break_downside, orb_break_upside, orb_up_time, orb_down_time = orb_strategy(df_five_minute, orb_break_downside, orb_break_upside, orb_up_time, orb_down_time)
                    if orb_result == 'call':
                        trade_grade = get_trade_grade(wb, df, 140, ztd, zbd, zts, zbs)
                        buy_call(wb, symbol, trade_grade, account_info)
                        logging.info("Taking call ORB trade:\n")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Took ORB call trade")

                        print(f"SID: {message.sid}")
                        orb_order_taken = True
                    elif orb_result == 'put':
                        trade_grade = get_trade_grade(wb, df, 140, ztd, zbd, zts, zbs)
                        buy_put(wb, symbol, trade_grade, account_info)
                        logging.info("Taking put ORB trade:\n")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Took ORB put trade")

                        print(f"SID: {message.sid}")
                        orb_order_taken = True
        
        account_info = wb.get_account()
        
        
        if account_info['openOrders']:
            #Check if new price is too hgih (2% higher)
            if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                print("Modify the open order please")
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Modify the open order please")

                print(f"SID: {message.sid}")
            else:
                wb.cancel_all_orders()
            
        
        #Position selling
        if account_info['positions'] and not account_info['openOrders'] and orb_order_taken:
            #Check percent gain or loss of position to sell it
            #gain_loss_pct (create new function) = gain_loss_pct(wb)
            df = wb.get_bars(symbol, stockId, interval, 140, 1)
            zone_check, ztd, zbd, zts, zbs = supply_demand(wb,df, 140,ztd,zbd,zts,zbs)
            trade_grade = get_trade_grade(wb, df, 140, ztd, zbd, zts, zbs)
            
            logging.info(wb.get_positions())
            print("ORB order in progress")
                        
            target_price = float(account_info['positions'][0]['costPrice']) * ( 1 + trade_grades[trade_grade]['stop_loss_pct'])
            stop_loss = float(account_info['positions'][0]['costPrice']) - (float(account_info['positions'][0]['costPrice']) * trade_grades[trade_grade]['stop_loss_pct'])
            
            account_info = wb.get_account()
            if float(account_info['positions'][0]['lastPrice']) >= target_price:
                wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                        lmtPrice=account_info['positions'][0]['lastPrice'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='SELL',
                        orderType='LMT',
                        enforce='GTC',
                        quant=int(account_info['positions'][0]['position']))
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Sold position for profit")
                print(f"SID: {message.sid}")

                time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()

                
                orb_order_taken = False
                
            #account_info = wb.get_account()
            elif float(account_info['positions'][0]['lastPrice']) <= stop_loss:
                wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                        lmtPrice=account_info['positions'][0]['lastPrice'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='SELL',
                        orderType='LMT',
                        enforce='GTC',
                        quant=int(account_info['positions'][0]['position']))
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Sold position for loss")

                print(f"SID: {message.sid}")
                time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()
                orb_order_taken = False

        account_info = wb.get_account()
        
        if account_info['positions'] and not account_info['openOrders'] and zone_order_taken:
            df = wb.get_bars(symbol, stockId, interval, 140, 1)
            prices_5m = wb.get_bars('SPY', wb.get_ticker('SPY'), 'm5', 10, 0)
            zone_check, ztd, zbd, zts, zbs = supply_demand(wb,df, 140,ztd,zbd,zts,zbs)
            trade_grade = get_trade_grade(wb, df, 140, ztd, zbd, zts, zbs)
            
            logging.info(wb.get_positions())
            print("Zone order in progress")
            print(f"Demand zone: {zbd} - {ztd}\nSupply zone: {zbs} - {zts}\n{zone_check}")
            if prices_5m.iloc[-1]['close'] < zbd and not zone_broken:
                zone_broken = True
                # wait until price action is above a supply zone or below a demand zone and then start the 25% stop loss from the price the option is at
                # Might need a variable defined in main to hold the value of the option when it breaks the negative side of the zone
                zone_break_candle_demand_price = float(account_info['positions'][0]['lastPrice'])
                logging.info(f"Broke out of demand zone. The current option price is {zone_break_candle_demand_price} and the stop loss will be 25% less than that")

            if prices_5m.iloc[-1]['close'] > zts and not zone_broken:
                zone_broken = True
                zone_break_candle_supply_price = float(account_info['positions'][0]['lastPrice'])
                logging.info(f"Broke out of supply zone. The current option price is {zone_break_candle_supply_price} and the stop loss will be 25% less than that")

            target_price = float(account_info['positions'][0]['costPrice']) * 1.42
            account_info = wb.get_account()
            if float(account_info['positions'][0]['lastPrice']) >= target_price:
                wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                        lmtPrice=account_info['positions'][0]['lastPrice'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='SELL',
                        orderType='LMT',
                        enforce='GTC',
                        quant=int(account_info['positions'][0]['position']))
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Sold position for profit")

                print(f"SID: {message.sid}")

                time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()
                zone_order_taken = False
            
            if zone_break_candle_supply_price != 0:
                stop_loss = zone_break_candle_supply_price - (zone_break_candle_supply_price * trade_grades[trade_grade]['stop_loss_pct'])
                
                account_info = wb.get_account()
                if float(account_info['positions'][0]['lastPrice']) <= stop_loss:
                    wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                            lmtPrice=account_info['positions'][0]['lastPrice'],
                            stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                            action='SELL',
                            orderType='LMT',
                            enforce='GTC',
                            quant=int(account_info['positions'][0]['position']))
                    message = client.messages.create(
                    to="+1<Phone Number>", 
                    from_="+<Twilio Phone Number>",
                    body="Sold position for loss")

                    print(f"SID: {message.sid}")
                    zone_order_taken = False
                    zone_broken = False
                    time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()

                
            
            elif zone_break_candle_demand_price != 0:
                stop_loss = zone_break_candle_demand_price - (zone_break_candle_demand_price * trade_grades[trade_grade]['stop_loss_pct'])
                
                account_info = wb.get_account()
                if float(account_info['positions'][0]['lastPrice']) <= stop_loss:
                    wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                            lmtPrice=account_info['positions'][0]['lastPrice'],
                            stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                            action='SELL',
                            orderType='LMT',
                            enforce='GTC',
                            quant=int(account_info['positions'][0]['position']))
                    message = client.messages.create(
                    to="+1<Phone Number>", 
                    from_="+<Twilio Phone Number>",
                    body="Sold position for loss")

                    print(f"SID: {message.sid}")
                    zone_order_taken = False
                    zone_broken = False
                    time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()

                
            
        account_info = wb.get_account()    
        if account_info['positions'] and not account_info['openOrders'] and not (orb_order_taken or zone_order_taken):
            #Check percent gain or loss of position to sell it
            #gain_loss_pct (create new function) = gain_loss_pct(wb)
            df = wb.get_bars(symbol, stockId, interval, 140, 1)
            zone_check, ztd, zbd, zts, zbs = supply_demand(wb,df, 140,ztd,zbd,zts,zbs)
            trade_grade = get_trade_grade(wb, df, 140, ztd, zbd, zts, zbs)
            
            logging.info(wb.get_positions())
            print("Default order in progress")
                        
            target_price = float(account_info['positions'][0]['costPrice']) * ( 1 + trade_grades[trade_grade]['stop_loss_pct'])
            stop_loss = float(account_info['positions'][0]['costPrice']) - (float(account_info['positions'][0]['costPrice']) * trade_grades[trade_grade]['stop_loss_pct'])
            
            account_info = wb.get_account()
            if float(account_info['positions'][0]['lastPrice']) >= target_price:
                wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                        lmtPrice=account_info['positions'][0]['lastPrice'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='SELL',
                        orderType='LMT',
                        enforce='GTC',
                        quant=int(account_info['positions'][0]['position']))
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Sold position for profit")

                print(f"SID: {message.sid}")

                time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()
                

                
                
            elif float(account_info['positions'][0]['lastPrice']) <= stop_loss:
                wb.place_order_option(optionId=account_info['positions'][0]['ticker']['tickerId'], 
                        lmtPrice=account_info['positions'][0]['lastPrice'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId=account_info['positions'][0]['ticker']['tickerId'])['data'][0]['askList'][0]['price']) -0.25 ),
                        action='SELL',
                        orderType='LMT',
                        enforce='GTC',
                        quant=int(account_info['positions'][0]['position']))
                message = client.messages.create(
                to="+1<Phone Number>", 
                from_="+<Twilio Phone Number>",
                body="Sold position for loss")

                print(f"SID: {message.sid}")

                time.sleep(10)
                account_info = wb.get_account()
                while account_info['openOrders']:
                    #Check if new price is too hgih (2% higher)
                    if float(account_info['openOrders'][0]['ask']) <= (1.05 * float(account_info['openOrders'][0]['lmtPrice'])):
                        #wb.modify_order_option(wb.get_current_orders(), account_info['openOrders'][0]['ask'], None, 'GTC', int(account_info['openOrders'][0]['totalQuantity']) - int(account_info['openOrders'][0]['filledQuantity']))
                        print("Modify the open order please")
                        message = client.messages.create(
                        to="+1<Phone Number>", 
                        from_="+<Twilio Phone Number>",
                        body="Modify the open order please")

                        print(f"SID: {message.sid}")
                    else:
                        wb.cancel_all_orders()
                    time.sleep(15)
                    account_info = wb.get_account()
                
        # Check if the market is still open
        
        account_info = wb.get_account()
        time.sleep(15)  # Wait 30 seconds before checking again

    
    if datetime.now().time() > datetime.strptime('16:00', '%H:%M').time():
            logging.info('Market is closed. Logging out...')
            wb.logout()
    
    
    
    '''
    # Set trade grade based on strategy
    trade_grade = 'A+'
    # Buy call or put option based on trade grade
    if trade_grade in ['A+', 'A', 'B+', 'B']:
        option_type = 'call'
    else:
        option_type = 'put'
    # Buy the option using OTOCO option order with stop loss and limit take profit orders
    if option_type == 'call':
        buy_call(wb, symbol, expiration_date, strike_price, option_to_buy['adjusted_mark_price'], stop_loss_price, limit_profit_price, quantity)
    else:
        buy_put(wb, symbol, expiration_date, strike_price, option_to_buy['adjusted_mark_price'], stop_loss_price, limit_profit_price, quantity)
    # Check if the market is still open
    market_hours = wb.get_calendar(stock=symbol)
    if market_hours and market_hours['isClosed']:
        logging.info('Market is closed. Logging out...')
        wb.logout()
    else:
        time.sleep(60)  # Wait 1 minute before checking again
    
        WORKING PAPER OPTIONS TRADE
    wb.place_order_option(optionId='1038464919', 
                        lmtPrice=wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price'],
                        stpPrice=str(float(wb.get_option_quote(stock=symbol, optionId='1038464919')['data'][0]['askList'][0]['price']) -0.25 ),
                        action='BUY',
                        orderType='LMT',
                        enforce='GTC',
                        quant=1)
    '''
    
if __name__ == '__main__':
    main()

