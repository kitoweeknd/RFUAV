import logging
import os
from colorama import Fore, Style, init


class colorful_logger:

    def __init__(self, name, logfile=None):
        self.name = name
        init(autoreset=True)

        logger_name = f"{name}:{os.getpid()}"
        self.logger = logging.getLogger(logger_name)

        # Prevent messages from propagating to root logger to avoid duplicate output
        self.logger.propagate = False

        # If logger already has handlers, deduplicate and return directly
        if self.logger.handlers:
            self._deduplicate_handlers()
            return

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

        # Add file handler for Train type
        if self.name == 'Train' and logfile:
            filehandler = logging.FileHandler(logfile)
            filehandler.setLevel(logging.INFO)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)

        # Add console handler for all types (including Train)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self._deduplicate_handlers()
        self.logger.setLevel(logging.INFO)

    def _deduplicate_handlers(self):
        unique = {}
        for handler in list(self.logger.handlers):
            key = (type(handler), getattr(handler, 'baseFilename', None))
            if key in unique:
                self.logger.removeHandler(handler)
            else:
                unique[key] = handler

    def log_with_color(self, message=None, color=Fore.WHITE):

        if self.name == 'Evaluate':
            color = Fore.CYAN
        elif self.name == 'Inference':
            color = Fore.MAGENTA

        if self.name == 'Train':
            colored_message = message
        else:
            colored_message = f"{color}{message}{Style.RESET_ALL}"

        self.logger.info(colored_message)


# Usage-------------------------------------------------------------
def main():
    test = colorful_logger('Inference')

    test.log_with_color('This is a debug message')
    test.log_with_color('This is an info message')
    test.log_with_color('This is a critical message')


if __name__ == '__main__':
    main()