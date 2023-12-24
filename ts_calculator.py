#!/usr/bin/env python
# coding: utf-8

# In[8]:

import numba as nb
import numpy as np
@nb.jit()
def threshold(data,threshold_high,threshold_low):
    result = np.where(data>threshold_high,1,np.where(data<threshold_low,-1,0))
    return result

@nb.jit()
def find_quantile_positions(matrix_2d, array_1d):
    if np.isscalar(array_1d):
        array_1d = [array_1d]
    
    results = []

    for value, col in zip(array_1d, matrix_2d.T):
        quantile_position = np.searchsorted(np.sort(col), value) / len(col)
        results.append(quantile_position)
        
    return np.array(results)


def ema_decay(matrix, alpha=0.1):
    """
    Compute the Exponential Moving Average (EMA) for a 2D array row-wise with the last row as the newest data.
    
    :param matrix: 2D array-like data
    :param alpha: Decay factor for EMA. It determines the weight of the current observation. 
                  A value closer to 1 gives more weight to recent observations, 
                  and a value closer to 0 gives more weight to older observations.
    :return: 2D array with the same shape as input with EMA values.
    """
    matrix = np.nan_to_num(matrix)
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return []
    
    n_rows, n_cols = len(matrix), len(matrix[0])
    result = [[0] * n_cols for _ in range(n_rows)]

    # Initialize the last row of result
    for j in range(n_cols):
        result[-1][j] = matrix[-1][j]

    # Compute EMA for each row based on the next row (since we are starting from the end)
    for i in range(n_rows - 2, -1, -1):  # We start from the second last row and move upwards
        for j in range(n_cols):
            result[i][j] = alpha * matrix[i][j] + (1 - alpha) * result[i+1][j]
    
    return result

@nb.jit()
def linear_regression(y, x):
    """
    Perform linear regression on two series, y and x, where both y and x are 2D arrays.

    :param y: 2D array where each column is a dependent variable series
    :param x: 2D array where each column is an independent variable series
    :return: A tuple of three 1D arrays (alphas, betas, errors) where alphas are the intercepts,
             betas are the slopes, and errors are the residuals of the regression for each series.
    """
    y = np.nan_to_num(y)
    x = np.nan_to_num(x)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")

    num_series = y.shape[1]
    alphas = np.zeros(num_series)
    betas = np.zeros(num_series)
    errors = np.zeros_like(y)

    for i in range(num_series):
        # Calculating beta (slope) for each series
        x_mean = np.mean(x[:, i])
        y_mean = np.mean(y[:, i])
        beta = np.sum((x[:, i] - x_mean) * (y[:, i] - y_mean)) / np.sum((x[:, i] - x_mean)**2)
        betas[i] = beta

        # Calculating alpha (intercept) for each series
        alpha = y_mean - beta * x_mean
        alphas[i] = alpha

        # Calculating errors (residuals) for each series
        errors[:, i] = y[:, i] - (alpha + beta * x[:, i])

    return alphas, betas, errors


def niomean(data):
    return np.nanmean(data,axis=0)


def niostddev(data):
    return np.nanstd(data,axis=0)


def niovar(data):
    return np.nanvar(data,axis=0)


def kurtosis(arr):
    """
    Calculate the kurtosis for each column in a 2D numpy array, replacing NaNs with zeros.

    :param arr: A 2D numpy array.
    :return: A 1D numpy array containing the kurtosis of each column.
    """
    # Replace NaNs with 0
    arr = np.nan_to_num(arr)

    n = arr.shape[0]
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1)
    adjusted_sum = np.sum(((arr - mean) / std)**4, axis=0)
    kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * adjusted_sum
    kurtosis -= 3 * ((n - 1)**2) / ((n - 2) * (n - 3))
    return kurtosis


def rank(arr):
    """
    Return the rank of each element in its row within a 2D numpy array.

    :param arr: A 2D numpy array.
    :return: A 2D numpy array where each element is replaced by its rank within its row.
    """
    # Convert NaNs to a number (optional based on how you want to handle NaNs)
    arr = np.nan_to_num(arr)

    # Get the rank of each element in its row
    ranked_arr = np.array([np.argsort(np.argsort(row)) + 1 for row in arr])
    return ranked_arr


def skew(arr):
    """
    Calculate the skewness for each column in a 2D numpy array.

    :param arr: A 2D numpy array.
    :return: A 1D numpy array containing the skewness of each column.
    """
    arr = np.nan_to_num(arr)
    n, num_columns = arr.shape
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1)

    skewness = np.sum(((arr - mean) / std) ** 3, axis=0) * (n / ((n - 1) * (n - 2)))
    return skewness

@nb.jit()
def linear_decay(matrix, decay_factor=0.1):
    """
    Compute the Linear Decay for a 2D array row-wise.

    :param matrix: 2D array-like data
    :param decay_factor: Linear decay factor.
    :return: 2D array with the same shape as input with Linear Decay applied.
    """
    matrix = np.nan_to_num(matrix)
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return np.array([])

    n_rows, n_cols = len(matrix), len(matrix[0])
    result = np.zeros_like(matrix)

    # Apply linear decay to each row
    for i in range(n_rows):
        for j in range(n_cols):
            decay_amount = decay_factor * j
            result[i][j] = max(matrix[i][j] - decay_amount, 0)  # Ensure the result is not negative

    return result


def correlation(arr1, arr2):
    """
    Calculate the correlation for each pair of columns in two 2D numpy arrays.

    :param arr1: First 2D numpy array.
    :param arr2: Second 2D numpy array.
    :return: A 1D numpy array containing the correlation of each pair of columns.
    """
    arr1 = np.nan_to_num(arr1)
    arr2 = np.nan_to_num(arr2)
    if arr1.shape[0] != arr2.shape[0] or arr1.shape[1] != arr2.shape[1]:
        raise ValueError("arr1 and arr2 must have the same shape")

    num_columns = arr1.shape[1]
    correlations = np.zeros(num_columns)

    for i in range(num_columns):
        # Compute correlation for each pair of columns
        corr_matrix = np.corrcoef(arr1[:, i], arr2[:, i])
        # The correlation of two series is located at [0, 1] and [1, 0] in the matrix
        correlations[i] = corr_matrix[0, 1]

    return correlations


def nioewa(arr, alpha=0.1):
    """
    Calculate the exponential weighted average for each column in a 2D array.
    The last row is considered the most recent data.

    :param arr: 2D array (list of lists or NumPy array).
    :param alpha: Smoothing factor (0 < alpha <= 1).
    :return: NumPy array containing the exponential weighted averages for each column.
    """
    arr = np.nan_to_num(arr)
    arr = np.array(arr)  # Ensure the input is converted to a NumPy array
    reversed_arr = np.flipud(arr)  # Reverse the array to start with the most recent data

    ewa = np.zeros(reversed_arr.shape[1])  # Initialize the EWA array
    for row in reversed_arr:
        ewa = alpha * row + (1 - alpha) * ewa

    return ewa

def ts_zscore(arr):
    arr = np.nan_to_num(arr)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1)

    z_scores = (arr - mean) / std
    return z_scores


def cross_zscore(arr):
    """
    Convert each element in a 2D numpy array to its row-wise Z-Score.

    :param arr: A 2D numpy array.
    :return: A 2D numpy array with each element converted to its row-wise Z-Score.
    """
    arr = np.nan_to_num(arr)
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, ddof=1, keepdims=True)
    z_scores = (arr - mean) / std

    return z_scores

@nb.jit()
def co_skewness(arr1, arr2):
    """
    Calculate the co-skewness for each pair of columns in two 2D numpy arrays.

    :param arr1: First 2D numpy array.
    :param arr2: Second 2D numpy array.
    :return: A 1D numpy array containing the co-skewness of each pair of columns.
    """
    arr1 = np.nan_to_num(arr1)
    arr2 = np.nan_to_num(arr2)
    if arr1.shape[0] != arr2.shape[0] or arr1.shape[1] != arr2.shape[1]:
        raise ValueError("arr1 and arr2 must have the same shape")

    num_columns = arr1.shape[1]
    co_skewness = np.zeros(num_columns)

    for i in range(num_columns):
        x = arr1[:, i]
        y = arr2[:, i]
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        co_skewness[i] = np.mean((x - mu_x)**2 * (y - mu_y))

    return co_skewness


@nb.jit()
def co_kurtosis(arr1, arr2):
    """
    Calculate the co-kurtosis for each pair of columns in two 2D numpy arrays.

    :param arr1: First 2D numpy array.
    :param arr2: Second 2D numpy array.
    :return: A 1D numpy array containing the co-kurtosis of each pair of columns.
    """
    arr1 = np.nan_to_num(arr1)
    arr2 = np.nan_to_num(arr2)
    if arr1.shape[0] != arr2.shape[0] or arr1.shape[1] != arr2.shape[1]:
        raise ValueError("arr1 and arr2 must have the same shape")

    num_columns = arr1.shape[1]
    co_kurtosis = np.zeros(num_columns)

    for i in range(num_columns):
        x = arr1[:, i]
        y = arr2[:, i]
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        co_kurtosis[i] = np.mean((x - mu_x)**2 * (y - mu_y)**2)

    return co_kurtosis


@nb.jit()
def ts_percent(arr, percent):
    """
    Calculate the specified percentile for each column in a 2D numpy array.

    :param arr: A 2D numpy array.
    :param percent: The percentile to calculate (between 0 and 100).
    :return: A 1D numpy array containing the specified percentile of each column.
    """
    arr = np.nan_to_num(arr)
    if percent < 0 or percent > 100:
        raise ValueError("Percent must be between 0 and 100")

    # Calculate the specified percentile for each column
    percent_values = np.percentile(arr, percent, axis=0)

    return percent_values


@nb.jit()
def ts_arg_max(arr):
    """
    Return the position of the maximum value in each column of a 2D numpy array.
    If the maximum value is in the last row, return 1; second to last, return 2, etc.

    :param arr: A 2D numpy array.
    :return: A 1D numpy array with the position of the max value in each column.
    """
    # Get the index of the max value in each column
    max_indices = np.argmax(arr, axis=0)

    # Convert indices to the specified format
    positions = arr.shape[0] - max_indices

    return positions


@nb.jit()
def ts_arg_min(arr):
    """
    Return the position of the minimum value in each column of a 2D numpy array.
    If the minimum value is in the last row, return 1; second to last, return 2, etc.

    :param arr: A 2D numpy array.
    :return: A 1D numpy array with the position of the min value in each column.
    """
    # Get the index of the min value in each column
    min_indices = np.argmin(arr, axis=0)

    # Convert indices to the specified format
    positions = arr.shape[0] - min_indices

    return positions