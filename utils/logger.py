import logging
import sys


class Logger(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if config.log.log_level == "debug":
            logging_level = logging.DEBUG
        elif config.log.log_level == "info":
            logging_level = logging.INFO
        elif config.log.log_level == "warn":
            logging_level = logging.WARN
        elif config.log.log_level == "error":
            logging_level = logging.ERROR
        else:
            raise TypeError(
                "No logging type named %s, candidate is: info, debug, error")
        logging.basicConfig(filename=config.log.logger_file,
                            level=logging_level,
                            format='%(asctime)s : %(levelname)s  %(message)s',
                            filemode="a", datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def debug(msg):
        """Log debug message
            msg: Message to log
        """
        logging.debug(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def info(msg):
        """"Log info message
            msg: Message to log
        """
        logging.info(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def warn(msg):
        """Log warn message
            msg: Message to log
        """
        logging.warning(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def error(msg):
        """Log error message
            msg: Message to log
        """
        logging.error(msg)
        sys.stderr.write(msg + "\n")


def init_logger(file_name='main.log'):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=file_name)
