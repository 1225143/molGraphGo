# calculation.py

def addition(num1: int, num2: int) -> int:
    return num1 + num2

def subtraction(num1: int, num2: int) -> int:
    return num1 - num2

def division(num1: int, num2: int) -> float:
    if num2 == 0:
        raise ZeroDivisionError()
    return num1 / num2


