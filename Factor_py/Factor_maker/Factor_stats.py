import numpy as np
from utils import reshape_to_2d,shift_up


def annual_vol(position_array,return_array,periods_per_year=365*90):
    portfolio_returns = np.nansum(position_array * return_array, axis=1)
    annualized_volatility = np.nanstd(portfolio_returns) * np.sqrt(periods_per_year)
    return annualized_volatility

def annual_return(position_array,return_array,periods_per_year=365*96):
    portfolio_returns = np.nansum(position_array * return_array, axis=1)
    total_return = np.prod(1 + portfolio_returns) - 1
    T = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (periods_per_year / T) - 1
    return annualized_return


def sharpe_ratio(position_array, return_array, risk_free_rate=0.0, periods_per_year=365*96):
    """
    Compute the Sharpe Ratio for a given portfolio.

    Parameters:
    - position_array (numpy array): Portfolio positions at each time step.
    - return_array (numpy array): Returns for each asset at each time step.
    - risk_free_rate (float): Annualized risk-free rate. Default is 0.0.
    - periods_per_year (int): Number of periods per year. Default is 365*90.

    Returns:
    - float: The Sharpe Ratio.
    """
    annual_ret = annual_return(position_array, return_array, periods_per_year)
    annual_volatility = annual_vol(position_array, return_array, periods_per_year)
    sharpe = (annual_ret-risk_free_rate)/annual_volatility
    return sharpe





def max_drawdown(position_array,return_array):
    portfolio_returns = np.sum(position_array * return_array, axis=1)
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    max_return = np.maximum.accumulate(cumulative_returns)
    drawdown = (max_return - cumulative_returns) / (1 + max_return)
    max_drawdown = np.max(drawdown)
    return max_drawdown

def compute_win_rate(weights, returns):
    # 找出交易开始和结束的索引
    trade_indices = np.where(np.diff(weights, axis=0).any(axis=1))[0] + 1
    trade_starts = [0] + list(trade_indices)
    trade_ends = list(trade_indices) + [len(weights)]

    winning_trades = 0
    total_trades = len(trade_starts)

    for start, end in zip(trade_starts, trade_ends):
        total_return = np.prod(1 + np.sum(weights[start:end-1] * returns[start+1:end], axis=1)) - 1
        if total_return > 0:
            winning_trades += 1

    return winning_trades / total_trades if total_trades != 0 else 0

# def average_profit_trade(position_array,return_array,fee):



def turnover(position_array,periods_per_day=96):
    turnover = np.mean(np.sum(np.abs(position_array[1:] - position_array[:-1]), axis=1))*periods_per_day
    return turnover


if __name__ == "__main__":
    return_array = np.memmap("./Cache/Return",dtype="float32",mode="r",shape=(2800,1,96))
    position_array = np.memmap(f"./Factor/Mom_decay_10_5_position",dtype="float32",mode="r",shape=(2800,1,96))
    return_2d = reshape_to_2d(return_array[731:1828])
    return_lag = shift_up(return_2d)
    position_2d = reshape_to_2d(position_array[731:1828])
    print("Sharpe")
    print(sharpe_ratio(position_2d,return_lag))
    print("Return")
    print(annual_return(position_2d,return_lag))
    print("turnover")
    print(turnover(position_2d))