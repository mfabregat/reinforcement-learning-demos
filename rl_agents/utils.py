import numpy as np

def moving_average(arr, window):
    arr = np.array(arr, dtype=float).flatten()
    if arr.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    window = max(1, min(window, arr.size))
    if window == 1:
        return np.arange(arr.size), arr
    ma = np.convolve(arr, np.ones(window) / window, mode="valid")
    x = np.arange(window - 1, arr.size)
    return x, ma

def almost_factors(number):
    '''
    https://stackoverflow.com/a/77243426
    find a pair of factors that are close enough for a number that is close enough
    '''
    def close_factors(number):
        ''' 
        find the closest pair of factors for a given number
        '''
        factor1 = 0
        factor2 = number
        while factor1 +1 <= factor2:
            factor1 += 1
            if number % factor1 == 0:
                factor2 = number // factor1
            
        return factor1, factor2
    while True:
        factor1, factor2 = close_factors(number)
        if 1/2 * factor1 <= factor2: # the fraction in this line can be adjusted to change the threshold aspect ratio
            break
        number += 1
    return factor1, factor2
