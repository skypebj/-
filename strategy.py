#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票/ETF 轮动策略回测
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

# ==================== 数据获取（含分红记录） ====================
def fetch_daily_data(symbol, stock_type='etf', start=None, end=None):
    """获取日线数据（未复权），返回 DataFrame"""
    cache_file = f"data_{symbol}.csv"
    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"从缓存加载 {symbol}")
        return df
    except FileNotFoundError:
        pass

    if stock_type == 'etf':
        # ETF 历史数据（前复权，但为了分红处理我们使用未复权，akshare默认未复权）
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                 start_date=start, end_date=end,
                                 adjust="")   # 空字符串表示不复权
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 数据")
        df.index = pd.to_datetime(df["日期"])
        df = df[["开盘", "收盘", "最高", "最低", "成交量"]]
        df.columns = ["open", "close", "high", "low", "volume"]
    elif stock_type == 'stock':
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end,
                                adjust="")   # 不复权
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 数据")
        df.index = pd.to_datetime(df["日期"])
        df = df[["开盘", "收盘", "最高", "最低", "成交量"]]
        df.columns = ["open", "close", "high", "low", "volume"]
    elif stock_type == 'index':
        # 沪深300指数
        df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        df.index = pd.to_datetime(df["date"])
        df = df[["open", "close", "high", "low", "volume"]]
        df.columns = ["open", "close", "high", "low", "volume"]
        df = df[start:end]
    else:
        raise ValueError("stock_type must be etf/stock/index")

    # 保存缓存
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
        # 根据净股息率反推每股股利（需要当前价格，这里填0，实际在回测中动态计算）
        div_df["dividend_per_share"] = 0.0
        return div_df

# ==================== 策略类 ====================
class RotationStrategy(bt.Strategy):
    """
    日线轮动策略（无WR）
    """
    params = (
        ('momentum_period', 20),      # 动量周期
        ('ma_period', 20),            # 均线周期
        ('etf_cost', ETF_COST),
        ('stock_cost', STOCK_COST),
        ('dividend_tax', DIVIDEND_TAX),
        ('defense_dividend_rate', DIVIDEND_RATE),
    )

    def __init__(self):
        # 数据线别名
        self.hs300 = self.datas[0]      # 沪深300
        self.chn = self.datas[1]        # 创业板ETF
        self.sp = self.datas[2]         # 标普500ETF
        self.defense = self.datas[3]    # 交通银行

        # 指标：沪深300 MA20
        self.ma20 = bt.ind.SimpleMovingAverage(self.hs300.close, period=self.p.ma_period)

        # 20日动量 (当前价 / 20日前价 - 1)
        self.mom_chn = (self.chn.close / self.chn.close(-self.p.momentum_period)) - 1
        self.mom_sp  = (self.sp.close / self.sp.close(-self.p.momentum_period)) - 1

        # 交易状态
        self.order = None
        self.current_target = None   # 当前持有标的代号: 'chn', 'sp', 'defense'
        self.next_buy_date = None    # 下一个买入日
        self.next_sell_date = None   # 下一个卖出日

        # 分红数据（仅对防御股）
        self.dividend_records = fetch_dividend_data(DEFENSE_STOCK)
        self.last_dividend_date = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入 {order.data._name}  {order.executed.price:.3f} 数量 {order.executed.size}")
            else:
                self.log(f"卖出 {order.data._name}  {order.executed.price:.3f} 数量 {order.executed.size}")
            self.order = None

    def notify_cashvalue(self, cash, value):
        self.cash = cash
        self.value = value

    def _is_last_second_trading_day(self, date):
        """判断是否为当月倒数第二个交易日（基于数据中的交易日历）"""
        # 获取当前月份的所有交易日（从数据中获取，但这里简化：使用当前数据的日期列表）
        # 由于backtrader中无法直接获取未来交易日，我们通过检查下一天是否为下个月来判断
        # 方法：获取当前日期在当月的序号，并检查本月剩余交易日数量
        # 简便方法：从当前日期开始，往后找2个交易日，如果跨越月份，则当前是倒数第二或第一？
        # 更可靠：利用数据中的日期索引，但我们无法直接访问。换一种方式：
        # 使用pandas日期功能：获取当前月份最后一天，然后往前推2个交易日
        # 由于策略只在next中调用，我们可以在外部预处理一个布尔列表传入，但为了简洁，我们在这里实现：
        # 获取当前数据的时间序列（通过self.datas[0].datetime.array 但不可直接获取）
        # 替代方案：在策略初始化时，预先计算每个日期的买卖标志，存入lines。
        # 为了让代码简单且可读，我们在next中动态判断：检查下一个交易日是否属于下个月，且下下个交易日也属于下个月？
        # 但不够准确。推荐在初始化时预计算所有日期的标志，然后通过一个line来访问。
        # 这里给出预计算方法：
        pass

    def _precompute_trade_dates(self):
        """预计算每个数据点的买入日和卖出日标志（在策略start中调用）"""
        # 获取所有数据的时间索引（取第一个数据的日期）
        dates = [self.datas[0].datetime.date(i) for i in range(len(self.datas[0]))]
        self.buy_flag = [False] * len(dates)
        self.sell_flag = [False] * len(dates)

        # 生成交易日历（去重）
        trading_days = sorted(set(dates))
        # 按月分组
        from collections import defaultdict
        month_days = defaultdict(list)
        for d in trading_days:
            month_days[(d.year, d.month)].append(d)

        # 计算每月倒数第二个交易日
        for (year, month), days in month_days.items():
            if len(days) >= 2:
                last_second = days[-2]
                # 找到该日期在原始dates中的索引
                idx = dates.index(last_second)
                self.buy_flag[idx] = True

        # 计算次月第8个交易日
        for i, d in enumerate(trading_days):
            # 计算下个月的第8个交易日
            next_month = d + relativedelta(months=1)
            next_month_days = [dd for dd in trading_days if dd.year == next_month.year and dd.month == next_month.month]
            if len(next_month_days) >= 8:
                sell_day = next_month_days[7]  # 索引7即第8个
                if d == sell_day:
                    self.sell_flag[i] = True
        # 注意：上面的循环会标记卖出日，但每个卖出日只对应一个买入日关系。实际上我们需要在买入日后的次月第8个交易日卖出，
        # 所以更准确的做法是：对每个买入日，计算对应的卖出日。但上面的方法会标记所有满足“当天是某个月的第8个交易日”的日期，
        # 如果某个月的第8个交易日恰好也是另一个月的倒数第二天？不影响，因为卖出日总是先于下一个买入日。
        # 我们将这些标志存储为策略的lines，以便在next中快速访问。
        # 由于backtrader lines动态性，我们简单存储为列表，并在next中用索引访问。
        self.buy_flags = self.buy_flag
        self.sell_flags = self.sell_flag

    def start(self):
        # 预计算买卖日期
        self._precompute_trade_dates()

    def next(self):
        if self.order:
            return  # 等待订单完成

        # 获取当前数据索引
        idx = len(self.datas[0]) - 1  # 当前bar的索引
        current_date = self.datas[0].datetime.date(0)

        # 1. 检查卖出日
        if self.sell_flags[idx]:
            # 卖出所有持仓
            for data in [self.chn, self.sp, self.defense]:
                if self.getposition(data).size > 0:
                    self.order = self.close(data)
                    self.current_target = None
                    return

        # 2. 检查买入日
        if self.buy_flags[idx]:
            # 先平仓（防止上月卖出日未执行的情况）
            for data in [self.chn, self.sp, self.defense]:
                if self.getposition(data).size > 0:
                    self.close(data)

            # 获取过滤条件：沪深300 >= MA20
            hs300_close = self.hs300.close[0]
            ma20_val = self.ma20[0]
            filter_ok = hs300_close >= ma20_val

            if filter_ok:
                # 计算20日动量，选择强者
                mom_c = self.mom_chn[0]
                mom_s = self.mom_sp[0]
                if mom_c >= mom_s:
                    target = self.chn
                    target_name = "创业板ETF"
                else:
                    target = self.sp
                    target_name = "标普500ETF"
                self.log(f"买入条件满足，动量({mom_c:.4f},{mom_s:.4f}) 选择 {target_name}")
            else:
                target = self.defense
                target_name = "交通银行(防御)"
                self.log(f"沪深300({hs300_close:.2f}) < MA20({ma20_val:.2f})，转入防御品种")

            # 全仓买入
            cash = self.broker.get_cash()
            size = int(cash / target.close[0])
            if size > 0:
                self.order = self.buy(target, size=size)
                self.current_target = target_name

    def notify_trade(self, trade):
        """交易完成后的手续费记录"""
        if trade.isclosed:
            self.log(f"交易毛利 {trade.pnl:.2f} 净利 {trade.pnlcomm:.2f}")

    def next_month_dividend(self):
        """处理防御品种的分红（在每年的除息日）"""
        # 仅在持有交通银行时处理分红
        pos = self.getposition(self.defense)
        if pos.size == 0:
            return

        current_date = self.datas[0].datetime.date(0)
        # 检查分红记录中是否有今天的除息日
        if current_date in self.dividend_records.index:
            div_per_share = self.dividend_records.loc[current_date, "dividend_per_share"]
            if div_per_share > 0:
                # 税前红利总额
                gross_div = div_per_share * pos.size
                net_div = gross_div * (1 - self.p.dividend_tax)
                self.broker.add_cash(net_div)
                self.log(f"收到 {DEFENSE_STOCK} 分红: 税前 {gross_div:.2f} 税后 {net_div:.2f}")
        else:
            # 如果没有实际分红数据，使用固定股息率模拟（每年一次，在6月30日）
            if current_date.month == 6 and current_date.day == 30:
                if self.last_dividend_date != current_date.year:
                    # 按当日市值估算税前红利
                    value = pos.size * self.defense.close[0]
                    gross_div = value * self.p.defense_dividend_rate
                    net_div = gross_div * (1 - self.p.dividend_tax)
                    self.broker.add_cash(net_div)
                    self.log(f"模拟分红: 市值 {value:.2f} 税后 {net_div:.2f}")
                    self.last_dividend_date = current_date.year

    def next(self):
        # 原有的next逻辑保留，上面已经定义了一个next，这里重写会覆盖，所以需要合并。
        # 由于上面已经写了next，这里不能再写一个。我将上面的next内容复制过来，并加入分红调用。
        # 实际代码中只有一个next方法，我将上面的next删除，用这个合并后的。
        pass

# 修复：上面定义了两个next，需要整合。将下面代码替换上面的next方法。
# 由于在代码编写中已经写了第一个next，但后面又写了一个占位，会导致错误。我们重新组织一下策略类，确保只有一个next。
# 下面是完整的策略类（修正版）。

# ==================== 修正版策略类（整合所有逻辑） ====================
class RotationStrategyFinal(bt.Strategy):
    params = (
        ('momentum_period', 20),
        ('ma_period', 20),
        ('etf_cost', ETF_COST),
        ('stock_cost', STOCK_COST),
        ('dividend_tax', DIVIDEND_TAX),
        ('defense_dividend_rate', DIVIDEND_RATE),
    )

    def __init__(self):
        self.hs300 = self.datas[0]
        self.chn = self.datas[1]
        self.sp = self.datas[2]
        self.defense = self.datas[3]

        self.ma20 = bt.ind.SimpleMovingAverage(self.hs300.close, period=self.p.ma_period)
        self.mom_chn = (self.chn.close / self.chn.close(-self.p.momentum_period)) - 1
        self.mom_sp  = (self.sp.close / self.sp.close(-self.p.momentum_period)) - 1

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
        from collections import defaultdict
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

    cerebro.addstrategy(RotationStrategyFinal)

    # 设置初始资金
    cerebro.broker.setcash(INIT_CASH)

    # 设置佣金（分别针对ETF和股票，但统一用setcommission会覆盖，我们通过策略中动态设置？Backtrader不支持按不同证券设置不同佣金，可以在策略中每次交易前设置？不完美）
    # 简便方法：在cerebro中设置一个默认佣金，然后在策略的next中根据交易品种动态修改broker的commission？太复杂。
    # 另一种：使用两个不同的数据对应的佣金，可以通过重写getcommission。这里我们简单设置一个平均成本0.045%？不对，用户要求ETF0.042%，股票0.05%。
    # 我们可以在策略中，在买入卖出时手动扣除？但backtrader自动计算。可以使用setcommission参数中，通过判断数据名称来设置不同佣金。
    # 通过创建一个自定义佣金方案：
    class FixedCommInfo(bt.CommissionInfo):
        def getsize(self, data, cash):
            return super().getsize(data, cash)

    # 更简单：分别对每个数据设置佣金
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=ETF_COST, name='159915'))
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=ETF_COST, name='513500'))
    cerebro.broker.addcommissioninfo(bt.CommissionInfo(commission=STOCK_COST, name='601328'))
    # 沪深300指数不交易，佣金无关

    # 设置滑点（可选）
    cerebro.broker.set_slippage_perc(0.0001)

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

    # 可选：绘制结果
    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    run_backtest()
