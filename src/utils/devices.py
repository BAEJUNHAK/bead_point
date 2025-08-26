# parse_devices 공용
import ast

def parse_devices(devices_str: str):
    if devices_str == "auto":
        return "auto"
    try:
        return int(devices_str)
    except ValueError:
        try:
            val = ast.literal_eval(devices_str)
            if isinstance(val, list):
                return val
        except (ValueError, SyntaxError):
            return devices_str
