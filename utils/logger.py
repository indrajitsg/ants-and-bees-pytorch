import logging
import os
import sys

def setup_logger(name, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        if not os.path.exists(os.path.join(save_dir, 'log.txt')):
            open(os.path.join(save_dir, 'log.txt'), 'w').close()
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger