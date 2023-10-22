import logging
import os
import datetime

logging_format = "%(asctime)s:%(filename)s:%(name)s:%(funcName)s:%(levelname)s:%(message)s"


class CustomLogger(logging.Logger):
    """
    Custom logger class that support logging to local file.
    ----------
    Args:
        source_name: str
            Name of the flow file that you want to log, suggest to leave it as __name__ . This helps naming the log file
        level: Optional[str]
            Log level, default to INFO
        write_local: Optional[bool]
            Whether to write log to local file, default to False
        
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton. Return the existed client if one's been already initiated
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        logger_name: str = __name__,
        level: str = "INFO",
        write_local: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(logger_name, level, *args, **kwargs)
        self._logger_name = logger_name
        self._log_file_name = self.log_file_name
        self._add_stream_handler()

        if write_local:
            self._add_file_handler()

    @property
    def log_file_name(self):
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._logger_name}_{tstamp}.log"

    def _add_stream_handler(self):
        """Add stream handler to stream logging info to the console"""
        stream_handler = logging.StreamHandler()
        stream_format = logging.Formatter(logging_format)
        stream_handler.setFormatter(stream_format)
        self.addHandler(stream_handler)

    def _add_file_handler(self):
        """Add file handler to write log to local file """
        log_path = "./logs"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{now}.log"

        file_handler = logging.FileHandler(
            filename=os.path.join(log_path, log_filename)
        )
        file_format = logging.Formatter(logging_format)
        file_handler.setFormatter(file_format)

        self.addHandler(file_handler)