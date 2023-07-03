import sys
import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    #sys.exc_info() -> returns a tuple with info about curr exception being handled
    #exc_tb = traceback information
    file_name= exc_tb.tb_frame.f_code.co_filename
    #retrieves filename where exception occurred
    error_message = "Error occured in python script name [{0}] line number [{1}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        #constructor of superclass called to initialize the custome exception
        #custom exception inherits the basic properties and behavior of regular exceptions, while still allowing us you to customize it as needed. 
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message