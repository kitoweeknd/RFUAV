import logging
from colorama import Fore, Style, init


class colorful_logger:

    def __init__(self, name, logfile=None):
        self.name = name
        init(autoreset=True)

        self.logger = logging.getLogger(name)
        
        # 防止消息向上传播到root logger，避免重复输出
        self.logger.propagate = False
        
        # 如果logger已经有handler，不再重复添加
        if self.logger.handlers:
            return

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

        # 为Train类型添加文件handler
        if self.name == 'Train' and logfile:
            filehandler = logging.FileHandler(logfile)
            filehandler.setLevel(logging.INFO)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)
        
        # 为所有类型添加控制台handler（包括Train）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.INFO)

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