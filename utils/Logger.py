import datetime
import os
from enum import Enum

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

class Logger:
    def __init__(self, log_file="neural_network.log"):
        self.log_file = log_file
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure the logs directory exists"""
        os.makedirs("logs", exist_ok=True)
        self.log_file = os.path.join("logs", self.log_file)

    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format the log message with timestamp and level"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{level.value}] {message}"

    def log(self, message: str, level: LogLevel = LogLevel.INFO, print_to_console: bool = True):
        formatted_message = self._format_message(level, message)
        
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted_message + "\n")
        
        # Print to console if requested
        if print_to_console:
            print(formatted_message)

    def info(self, message: str):
        self.log(message, LogLevel.INFO)

    def warning(self, message: str):
        self.log(message, LogLevel.WARNING)

    def error(self, message: str):
        self.log(message, LogLevel.ERROR)

    def debug(self, message: str):
        self.log(message, LogLevel.DEBUG)