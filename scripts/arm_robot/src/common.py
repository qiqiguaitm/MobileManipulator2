import inspect

def debug_print(*args, **kwargs):
    # 获取调用者的框架信息
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    line_number = frame.f_lineno
    function_name = frame.f_code.co_name
    
    # 只显示文件名，不显示完整路径
    import os
    short_filename = os.path.basename(filename)
    
    print(f"[{short_filename}:{line_number}:{function_name}]", *args, **kwargs)


