from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def create_log(filename, folder):
    #p = Path(folder)
    loggfile = Path(folder)
    loggfile= loggfile / filename
    logger = logging.getLogger('Energianalyselog')
    logger.setLevel('DEBUG')
    r_filh = RotatingFileHandler(loggfile, mode = 'a', maxBytes = 10000000, backupCount = 50)
    r_filh.setLevel('DEBUG')  # initially set to debug
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
    r_filh.setFormatter(formatter)
    logger.addHandler(r_filh)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.info("-----------------------STARTER NY ANALYSE-------------------------")
    logger.info("-----------------------STARTER NY ANALYSE-------------------------")
    logger.info("-----------------------STARTER NY ANALYSE-------------------------")
    logger.info("Filnavn: " + loggfile.name)