from tkinter import W
from log_tqdm import my_log
from tqdm import tqdm
import time

log = my_log('./../results/test2.log')

i_tqdm = tqdm(range(30))

for i in i_tqdm:
    if i % 10 == 0:
        log.info('test')
        log.debug('debug')
    time.sleep(0.1)
    i_tqdm.set_description(f'test of {i}')
