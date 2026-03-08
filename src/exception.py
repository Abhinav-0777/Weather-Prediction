import sys 
import os

def error_message_detail(error, error_detail:sys) :
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    function_name = exc_tb.tb_frame.f_code.co_name

    return f'ERROR :{str(error)}, FILE :{file_name}, LINE :{line_number}, FUNCTION :{function_name}'

class CustomException(Exception) :

    def __init__(self, error_occurred, error_detail:sys) :
        super().__init__(error_occurred)
        self.error_occurred = error_message_detail(error_occurred,error_detail=error_detail)

    def __str__(self) :
        return self.error_occurred