import psutil
import torch
from time import perf_counter
from contextlib import contextmanager

"""
Measure the time and memory usage of a block of code.
"""

@contextmanager
def profiler(description: str, length: int = 80, pad_char: str = ':') -> None:
    def get_memory_usage():
        usage = {
            'ram used ': psutil.virtual_memory().used / 1e9,
            'ram avail': psutil.virtual_memory().available / 1e9
        }
        if torch.cuda.is_available():
            usage.update({
                'gpu alloc': torch.cuda.memory_allocated() / 1e9,
                'gpu reser': torch.cuda.memory_reserved() / 1e9
            })
        return usage

    print('\n' + description.center(length, pad_char))
    before = get_memory_usage()
    # print all the memory usage on the same line with consistent width
    [print(f'{k}: {v:6.1f}', end=' | ') for k, v in before.items()]
    print()
    start = perf_counter()
    yield

    seconds = perf_counter() - start
    after = get_memory_usage()
    # print all the memory usage DIFFERENCES on the same line with a '+' or '-' sign
    diff = {k: after[k] - v for k, v in before.items()}
    [print(f'{k}: {v - before[k]:+6.1f}', end=' | ') for k, v in after.items()]
    print()
    print(f'{seconds:.2f} s for {description}'.center(length, pad_char))

# Example usage:
if __name__ == '__main__':
    with profiler('executing some code block to test the profiler'):
        print('Calculating...')
        