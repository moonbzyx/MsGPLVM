# from signal import pause
# from tqdm import tqdm
# import time

# iter_tqdm = tqdm(range(1000))

# for i in iter_tqdm:
#     time.sleep(0.1)
#     iter_tqdm.set_description(f'This is {i} in tqdm')

import logging
import colorlog
from loguru import logger
from tqdm import tqdm
import time


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class TestHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# log = logging.getLogger(__name__)
log = logging.getLogger('msgplvm_log')
log.setLevel(logging.DEBUG)
# handler = TqdmHandler()
handler = TqdmLoggingHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%d-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'SUCCESS:': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        },
    ))

# log.addHandler(TqdmLoggingHandler())
log.addHandler(handler)
#    colorize=True,
#    format="<green>{time}</green>|<red>{level}</red>|{message}")
# logger.add(handler)
# logger.add(handler)

for i in tqdm(range(30)):
    if i % 10 == 0:
        log.info("Half-way there!")
        log.debug("test the debug")
        log.error("This is error")
        log.critical('Test critical')
        # logger.debug("test")
    time.sleep(0.1)

# my_format = "<green>{time:YY-MM-DD HH:mm:ss.SSS}</green> | " \
#             "<level>{level: <8}</level> | " \
#             "<blue>{name}</blue>:<cyan>{function}</cyan>:<y>{line}</y> - <level>{message}</level>"

# logger.remove()
# handler2 = TestHandler()
# logger.add(handler2, colorize=True, format=my_format, level='DEBUG')
# # logger.add(handler2)
# logger.add('./../results/test_tqdm.log', level='ERROR', mode='w')

# def my_log_test():
#     for i in tqdm(range(30)):
#         if i % 10 == 0:
#             # log.info("Half-way there!")
#             # log.debug("test the debug")
#             # log.error("This is error")
#             # log.critical('Test critical')
#             logger.debug("This is debug")
#             logger.info("This is info")
#             logger.error("This is error")
#         time.sleep(0.1)

# if __name__ == '__main__':
#     my_log_test()
