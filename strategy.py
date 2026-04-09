#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票/ETF 轮动策略回测（日线，无WR）
- 买入：每月倒数第2个交易日
- 卖出：次月第8个交易日
- 过滤：沪深300 >= 20日均线
- 轮动：创业板ETF(159915) vs 标普500ETF(513500)，20日动量强者
- 防御：条件不满足时持有交通银行(601328)
- 满仓进出，包含交易成本、分红税
"""

import backtrader as bt
import akshare as ak
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数配置 ====================
START_DATE = "2013-12-10"   # 513500 上市日期
END_DATE   = "2026-04-09"
INIT_CASH  = 100000

# 标的代码
ETF_CHINEXT = "159915"      # 创业板
ETF_SP500   = "513500"      # 标普500
DEFENSE_STOCK = "601328"    # 交通银行
INDEX_HS300 = "000300"      # 沪深300

# 交易成本
ETF_COST = 0.00042          # ETF 买卖总成本 0.042%
STOCK_COST = 0.0005         # 股票买卖成本 0.05%（印花税+佣金）

# 分红参数（交通银行）
DIVIDEND_TAX = 0.20         # 股息税 20%
DIVIDEND_RATE = 0.0416      # 年化股息率 4.16%（税前）

# ==================== 数据获取（含多种接口备选） ====================
def fetch_daily_data(symbol, stock_type='etf', start=None, end=None):
    """获取日线数据（未复权），支持多种接口，返回 DataFrame"""
    cache_file = f"data_{symbol}.csv"
    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"从缓存加载 {symbol}")
        return df
    except FileNotFoundError:
        pass

    # 统一日期格式（用于接口）
    start_str = start.replace('-', '') if start else None
    end_str = end.replace('-', '') if end else None

    df = None
    if stock_type == 'etf':
        # 方法1：尝试 stock_zh_a_hist（ETF 可当作股票）
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_str, end_date=end_str,
                                    adjust="")
            if df is not None and not df.empty:
                print(f"使用 stock_zh_a_hist 获取 {symbol} 成功")
        except Exception as e:
            print(f"stock_zh_a_hist 失败: {e}")

        # 方法2：尝试 fund_etf_hist_em
        if df is None or df.empty:
            try:
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                         start_date=start_str, end_date=end_str,
                                         adjust="")
                if df is not None and not df.empty:
                    print(f"使用 fund_etf_hist_em 获取 {symbol} 成功")
            except Exception as e:
                print(f"fund_etf_hist_em 失败: {e}")

        # 方法3：尝试 fund_etf_hist_sina
        if df is None or df.empty:
            try:
                df = ak.fund_etf_hist_sina(symbol=symbol)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df["date"])
                    if start:
                        df = df[start:end]
                    print(f"使用 fund_etf_hist_sina 获取 {symbol} 成功")
            except Exception as e:
                print(f"fund_etf_hist_sina 失败: {e}")

    elif stock_type == 'stock':
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_str, end_date=end_str,
                                    adjust="")
            print(f"使用 stock_zh_a_hist 获取股票 {symbol} 成功")
        except Exception as e:
            print(f"股票 {symbol} 获取失败: {e}")

    elif stock_type == 'index':
        # 沪深300指数代码 "000300"
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
            df.index = pd.to_datetime(df["date"])
            if start:
                df = df[start:end]
            print(f"使用 stock_zh_index_daily 获取指数 {symbol} 成功")
        except Exception as e:
            print(f"指数 {symbol} 获取失败: {e}")

    if df is None or df.empty:
        raise ValueError(f"无法获取 {symbol} 数据，请检查网络或 AKShare 接口")

    # 统一列名
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })
    df = df[["open", "close", "high", "low", "volume"]]
    df.columns = ["open", "close", "high", "low", "volume"]
    df.to_csv(cache_file)
    print(f"下载并保存 {symbol} 数据，共 {len(df)} 行")
    return df

def fetch_dividend_data(stock_code):
    """获取股票历史分红记录（除息日，每股税前红利）"""
    cache_file = f"dividend_{stock_code}.csv"
    try:
        div_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"从缓存加载分红数据 {stock_code}")
        return div_df
    except FileNotFoundError:
        pass

    # 获取分红数据
    try:
        df = ak.stock_history_dividend_detail(symbol=stock_code)
        # 筛选已实施的分红
        df = df[df["实施进度"] == "实施"]
        df["除权除息日"] = pd.to_datetime(df["除权除息日"])
        df["每股股利"] = df["每股股利"].astype(float)
        div_df = df[["除权除息日", "每股股利"]].set_index("除权除息日")
        div_df.columns = ["dividend_per_share"]
        div_df.to_csv(cache_file)
        print(f"获取 {stock_code} 分红数据，共 {len(div_df)} 次")
        return div_df
    except Exception as e:
        print(f"获取分红数据失败: {e}，将使用固定股息率模拟")
        # 如果无法获取实际分红，则模拟每年6月30日派息
        years = range(2007, 2027)
        dates = [datetime.date(y, 6, 30) for y in years]
        div_df = pd.DataFrame(index=pd.DatetimeIndex(dates))
        div_df["dividend_per_share"] = 0.0
        return div_df

# ==================== 策略类 ====================
class RotationStrategy(bt.Strategy):
    """
    日线轮动策略（无WR）
    """
    params = (
        ('momentum_period', 20),
        ('ma_period', 20),
        ('etf_cost', ETF_COST),
        ('stock_cost', STOCK_COST),
        ('dividend_tax', DIVIDEND_TAX),
        ('defense_dividend_rate', DIVIDEND_RATE),
    )

    def __init__(self):
        # 数据线别名
        self.hs300 = self.datas[0]
        self.chn = self.datas[1]
        self.sp = self.datas[2]
        self.defense = self.datas[3]

        # 指标：沪深300 MA20
        self.ma20 = bt.ind.SimpleMovingAverage(self.hs300.close, period=self.p.ma_period)

        # 20日动量 (当前价 / 20日前价 - 1)
        self.mom_chn = (self.chn.close / self.chn.close(-self.p.momentum_period)) - 1
        self.mom_sp  = (self.sp.close / self.sp.close(-self.p.momentum_period)) - 1

        # 交易状态
        self.order = None
        self.current_target = None
        self.buy_flags = []
        self.sell_flags = []
        self.dividend_records = fetch_dividend_data(DEFENSE_STOCK)
        self.last_div_year = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入 {order.data._name}  {order.executed.price:.4f} x {order.executed.size}")
            else:
                self.log(f"卖出 {order.data._name}  {order.executed.price:.4f} x {order.executed.size}")
            self.order = None

    def start(self):
        # 预计算买卖日期
        dates = [self.datas[0].datetime.date(i) for i in range(len(self.datas[0]))]
        trading_days = sorted(set(dates))
        month_days = defaultdict(list)
        for d in trading_days:
            month_days[(d.year, d.month)].append(d)

        # 买入标志：每月倒数第二个交易日
        buy_set = set()
        for days in month_days.values():
            if len(days) >= 2:
                buy_set.add(days[-2])

        # 卖出标志：次月第8个交易日
        sell_set = set()
        for i, d in enumerate(trading_days):
            next_month = d + relativedelta(months=1)
            next_days = [dd for dd in trading_days if dd.year == next_month.year and dd.month == next_month.month]
            if len(next_days) >= 8:
                sell_day = next_days[7]
                sell_set.add(sell_day)

        self.buy_flags = [date in buy_set for date in dates]
        self.sell_flags = [date in sell_set for date in dates]

    def next(self):
        if self.order:
            return

        idx = len(self.datas[0]) - 1
        current_date = self.datas[0].datetime.date(0)

        # 1. 卖出日：平仓所有
        if self.sell_flags[idx]:
            for data in [self.chn, self.sp, self.defense]:
                if self.getposition(data).size > 0:
                    self.order = self.close(data)
                    self.current_target = None
                    return

        # 2. 买入日
        if self.buy_flags[idx]:
            # 先清仓
            for data in [self.chn, self.sp, self.defense]:
                if self.getposition(data).size > 0:
                    self.close(data)

            # 过滤条件
            hs300_close = self.hs300.close[0]
            ma20_val = self.ma20[0]
            if hs300_close >= ma20_val:
                mom_c = self.mom_chn[0]
                mom_s = self.mom_sp[0]
                if mom_c >= mom_s:
                    target = self.chn
                    tname = "创业板"
                else:
                    target = self.sp
                    tname = "标普500"
                self.log(f"条件满足，动量({mom_c:.4f},{mom_s:.4f}) 买入 {tname}")
            else:
                target = self.defense
                tname = "交通银行(防御)"
                self.log(f"沪深300({hs300_close:.2f}) < MA20({ma20_val:.2f}) 买入防御")

            cash = self.broker.get_cash()
            size = int(cash / target.close[0])
            if size > 0:
                self.order = self.buy(target, size=size)
                self.current_target = tname

        # 3. 处理分红（仅在持有防御品种时）
        pos = self.getposition(self.defense)
        if pos.size > 0:
            # 实际分红记录
            if current_date in self.dividend_records.index:
                div_ps = self.dividend_records.loc[current_date, "dividend_per_share"]
                if div_ps > 0:
                    gross = div_ps * pos.size
                    net = gross * (1 - self.p.dividend_tax)
                    self.broker.add_cash(net)
                    self.log(f"实际分红 {DEFENSE_STOCK}: 税前 {gross:.2f} 税后 {net:.2f}")
            # 模拟固定股息率（每年6月30日）
            elif current_date.month == 6 and current_date.day == 30:
                if self.last_div_year != current_date.year:
                    value = pos.size * self.defense.close[0]
                    gross = value * self.p.defense_dividend_rate
                    net = gross * (1 - self.p.dividend_tax)
                    self.broker.add_cash(net)
                    self.log(f"模拟分红: 市值 {value:.2f} 税后 {net:.2f}")
                    self.last_div_year = current_date.year

# ==================== 主程序 ====================
def run_backtest():
    # 1. 获取所有数据
    print("正在获取数据...")
    hs300_df = fetch_daily_data(INDEX_HS300, stock_type='index', start=START_DATE, end=END_DATE)
    chn_df = fetch_daily_data(ETF_CHINEXT, stock_type='etf', start=START_DATE, end=END_DATE)
    sp500_df = fetch_daily_data(ETF_SP500, stock_type='etf', start=START_DATE, end=END_DATE)
    defense_df = fetch_daily_data(DEFENSE_STOCK, stock_type='stock', start=START_DATE, end=END_DATE)

    # 2. 对齐日期索引（取所有交易日期的交集）
    common_dates = hs300_df.index.intersection(chn_df.index).intersection(sp500_df.index).intersection(defense_df.index)
    if len(common_dates) == 0:
        raise ValueError("无共同交易日，请检查数据范围")
    hs300_df = hs300_df.loc[common_dates]
    chn_df = chn_df.loc[common_dates]
    sp500_df = sp500_df.loc[common_dates]
    defense_df = defense_df.loc[common_dates]

    # 3. 转换为Backtrader数据源
    class PandasData(bt.feeds.PandasData):
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
        )

    data_hs300 = PandasData(dataname=hs300_df, name='HS300')
    data_chn = PandasData(dataname=chn_df, name='159915')
    data_sp = PandasData(dataname=sp500_df, name='513500')
    data_defense = PandasData(dataname=defense_df, name='601328')

    # 4. 初始化回测引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data_hs300)
    cerebro.adddata(data_chn)
    cerebro.adddata(data_sp)
    cerebro.adddata(data_defense)

    cerebro.addstrategy(RotationStrategy)

    # 设置初始资金
    cerebro.broker.setcash(INIT_CASH)

    # 分别设置不同品种的佣金
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=ETF_COST, name='159915'))
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=ETF_COST, name='513500'))
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=STOCK_COST, name='601328'))

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # 打印初始资金
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 输出结果
    print(f"\n最终资金: {cerebro.broker.getvalue():.2f}")
    print(f"总收益率: {(cerebro.broker.getvalue()/INIT_CASH - 1)*100:.2f}%")

    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.4f}")

    # 最大回撤
    dd = strat.analyzers.drawdown.get_analysis()
    print(f"最大回撤: {dd.max.drawdown:.2f}%")

    # 年化收益率
    rets = strat.analyzers.returns.get_analysis()
    print(f"年化收益率: {rets.get('rnorm100', 0):.2f}%")

    # 交易统计
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.total.total if hasattr(trades, 'total') else 0
    print(f"总交易次数: {total_trades}")

    # 可选：保存净值曲线图（适用于非交互环境）
    try:
        import matplotlib.pyplot as plt
        fig = cerebro.plot(style='candlestick')[0][0]
        fig.savefig('backtest_result.png')
        print("净值曲线已保存为 backtest_result.png")
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    run_backtest()
