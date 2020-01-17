import traceback
from time import time


def separate(info='', sep='=', num=5):
    print()
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)
    print()


def time_filter_with_dict_param(func, **kwargs):
    try:
        t1 = time()
        func(**kwargs)
        t2 = time()
        print(func.__name__, " spends ", t2 - t1, 'ms')
    except:
        traceback.print_exc()


def time_filter_with_tuple_param(func, *args):
    t1 = time()
    func(*args)
    t2 = time()
    print(func.__name__, " spends ", t2 - t1, 'ms')


def test_sum(n):
    result = 1
    for i in range(n):
        result *= n
    return result


if __name__ == "__main__":
    # time_filter_with_dict_param(test_sum, n=500)
    # time_filter_with_dict_param(test_sum, n=5000)
    separate()
