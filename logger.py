import logging
import sys
import time

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f"log/{time.time()}.log")
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
logger.info("Começando a logar")