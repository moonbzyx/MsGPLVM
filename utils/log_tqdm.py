# from signal import pause
# from tqdm import tqdm
# import time

# iter_tqdm = tqdm(range(1000))

# for i in iter_tqdm:
#     time.sleep(0.1)
#     iter_tqdm.set_description(f'This is {i} in tqdm')

import logging
from loguru import logger
from tqdm import tqdm
import time


class TqdmHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def my_log(path='./../results/test_tqdm.log'):

    format_1 = "<green>{time:[MM-DD] HH:mm:ss}</green> | " \
                "<level>{level: <8}</level> | " \
                "<blue>{name}</blue>:_<cyan>{function}</cyan>:_<y>{line}</y> - <level>{message}</level>"

    format_2 = "<green>{time:[_DD_] HH:mm:ss.SSS}</green> | " \
                "<level>{level: <8}</level> | " \
                "<blue>{name}</blue>:_<cyan>{function}</cyan>:_<y>{line}</y> - <level>{message}</level>"
    # remove the default handler: stderr
    logger.remove()
    # replacing stderr with tqdmHandler
    handler = TqdmHandler()
    # the first handler, showing messages on screen
    logger.add(handler, colorize=True, format=format_2, level='DEBUG')
    # log the messages into file
    logger.add(path, format=format_1, level='DEBUG', mode='w')
    return logger


if __name__ == '__main__':

    def my_log_test():
        log = my_log()
        iter_tqdm = tqdm(range(30))
        for i in iter_tqdm:
            if i % 10 == 0:
                # logger.debug("This is debug")
                # logger.info("This is info")
                # logger.error("This is error")
                log.info(f"This is the {i} info")
                log.debug(f"This is the {i} debug")
            time.sleep(0.2)
            iter_tqdm.set_description(f'This is the {i}th sample')

    my_log_test()
