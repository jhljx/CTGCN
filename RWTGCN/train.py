import os
import torch

def set_thread(self, thread_num=None):
    if thread_num is None:
        thread_num = os.cpu_count() - 4
    torch.set_num_threads(thread_num)

