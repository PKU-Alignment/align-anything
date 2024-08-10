import logging
from typing import Any, List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from datetime import datetime
import os
import csv

class EvalLogger:
    def __init__(self, name, log_dir='.', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.set_rich_console_handler()
        log_file = self.create_log_file(log_dir, name)
        self.set_file_handler(log_file)
        self.console = Console()
        self.log_dir = log_dir

    def set_rich_console_handler(self):
        console_handler = RichHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def set_file_handler(self, log_file):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def create_log_file(self, log_dir, name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"{name}.log")
        return log_file

    def log(self, level, message):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)

    def print_table(
        self,
        title: str,
        columns: List[str] = None,
        rows: List[List[Any]] = None,
        data: Dict[str, List[Any]] = None,
        max_num_rows: int = None,
        to_csv: bool = False,
        csv_file: Optional[str] = None
    ):
        table = Table(title=title)
        
        if data and columns is None:
            columns = list(data.keys())

        if columns:
            for col in columns:
                table.add_column(col)
        
        if rows:
            if max_num_rows:
                rows = rows[:max_num_rows]
            for row in rows:
                table.add_row(*map(str, row))
        
        if data:
            if max_num_rows:
                data = {k: v[:max_num_rows] for k, v in data.items()}
            num_rows = len(next(iter(data.values())))
            for i in range(num_rows):
                row = [data[col][i] for col in columns]
                table.add_row(*map(str, row))
        
        self.console.print(table)

        if to_csv:
            self.save_to_csv(columns, rows, data, csv_file)

    def save_to_csv(self, columns: List[str], rows: List[List[Any]], data: Dict[str, List[Any]], csv_file: Optional[str]):
        if csv_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            csv_file = os.path.join(self.log_dir, f"table_{timestamp}.csv")
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            
            if rows:
                for row in rows:
                    writer.writerow(row)
            
            if data:
                num_rows = len(next(iter(data.values())))
                for i in range(num_rows):
                    row = [data[col][i] for col in columns]
                    writer.writerow(row)


if __name__ == "__main__":

    log_directory = 'logs'
    custom_logger = EvalLogger('MyApp', log_dir=log_directory)
    
    custom_logger.log('debug', 'This is a debug message')
    custom_logger.log('info', 'This is an info message')
    custom_logger.log('warning', 'This is a warning message')
    custom_logger.log('error', 'This is an error message')
    custom_logger.log('critical', 'This is a critical message')
   
    title = "User Information"
    
    columns = ["Name", "Age", "City"]
    rows = [
        ["Alice", 30, "New York"],
        ["Bob", 25, "San Francisco"],
        ["Charlie", 35, "Los Angeles"]
    ]
    custom_logger.print_table(title=title, columns=columns, rows=rows, to_csv=True)

    data = {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [30, 25, 35],
        "City": ["New York", "San Francisco", "Los Angeles"]
    }
    custom_logger.print_table(title=title, data=data, to_csv=True)