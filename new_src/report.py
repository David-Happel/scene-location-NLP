import logging
import time
from datetime import datetime
import os

t = time.time()
now = datetime.now()

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
report_dir = os.path.join(directory_path, 'reports', 'report-'+now.strftime("%Y-%m-%d-%H-%M-%S"))
os.mkdir(report_dir)

logging.basicConfig(filename=os.path.join(report_dir, 'log.log'), level=logging.DEBUG)

def log(text):
    print_string = "{:0.4f} - {}"
    logging.info(print_string.format(time.time() - t, text))

def report_path(filename):
    return  os.path.join(report_dir, filename)