

def mustbe(variable, _type: type):
    if not isinstance(variable, _type):
        raise TypeError(f"Expected type {_type}, got {type(variable)}")