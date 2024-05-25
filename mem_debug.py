#memory_debug.py
import psutil
import torch
import time
from datetime import timedelta
from functools import wraps

def mem_debug(func):
    def print_memory_info():
        BAR = 50
        ram = psutil.virtual_memory()
        used_bar = int(ram.used / ram.total * BAR)
        avail_bar = int(ram.available / ram.total * BAR)
        ram_bar = '[' + '█' * used_bar + '░' * avail_bar + ' ' * (BAR - avail_bar - used_bar) + ']'

        print(f"{ram_bar}  RAM: {ram.used / (1024 ** 3):4.1f} used      {ram.available / (1024 ** 3):4.1f} available {ram.total / (1024 ** 3):4.1f} GB ({ram.percent:.0f}%)")

        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_allocated = torch.cuda.memory_allocated()
            vram_reserved = torch.cuda.memory_reserved()
            alloc_bar = int(vram_allocated / vram_total * BAR)
            reserv_bar = int(vram_reserved / vram_total * BAR)

            vram_bar = '[' + '█' * alloc_bar + '#' * reserv_bar + ' ' * (BAR - reserv_bar - alloc_bar) + ']'
            print(f"{vram_bar} VRAM: {vram_allocated / (1024 ** 3):4.1f} allocated {vram_reserved / (1024 ** 3):4.1f} reserved  {vram_total / (1024 ** 3):4.1f} GB")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"'{func.__name__}' called")
        print_memory_info()

        start_time = time.perf_counter()
        print(f"Executing '{func.__name__}'")
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        s = timedelta(seconds=end_time - start_time).total_seconds()
        time = (f"{s/3600:.2g}h" if s >= 3600 else f"{s/60:.2g}m" if s >= 60 else f"{s:.2g}s")

        print_memory_info()
        print(f"Finished '{func.__name__}' in {time}\n")

        return result
    return wrapper

