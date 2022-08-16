from typing import Tuple


class Coefficient():

    def __init__(self, value: float):
        self.value = value

    def __next__(self):
        return self.value

    def jsonize(self):
        return {
            "class_name": self.__class__.__name__,
            "value": self.value
        }


class LinearAnnealing(Coefficient):

    def __init__(self, init_value: float, final_value: float, n_iterations: int) -> None:
        if final_value > init_value:
            raise ValueError(
                f"Final_value: {final_value} must be smaller than init_value: {init_value}")
        self.init_value = init_value
        self.final_value = final_value
        self.n_iterations = n_iterations
        self.delta = (init_value - final_value) / n_iterations
        self.value = init_value

    def __next__(self):
        value_ = self.value
        self.value = max(self.final_value, self.value - self.delta)
        return value_

    def jsonize(self):
        return {
            "class_name": self.__class__.__name__,
            "value": self.value,
            "init_value": self.init_value,
            "final_value": self.final_value,
            "n_iterations": self.n_iterations,
        }


class ExponentialAnnealing(Coefficient):

    def __init__(self, init_value: float, final_value: float, exponent: float) -> None:
        if final_value > init_value:
            raise ValueError(
                f"Final_value: {final_value} must be smaller than init_value: {init_value}")
        if not (0 <= exponent <= 1.0):
            raise ValueError(f"Exponent: {exponent} must be in range [0, 1]")
        self.init_value = init_value
        self.final_value = final_value
        self.exponent = exponent
        self.value = init_value

    def __next__(self):
        value_ = self.value
        self.value = max(self.final_value, self.value * self.exponent)
        return value_

    def jsonize(self):
        return {
            "class_name": self.__class__.__name__,
            "value": self.value,
            "init_value": self.init_value,
            "final_value": self.final_value,
            "exponent": self.exponent,
        }


def load_from_dict(args_dict):
    class_name = args_dict.pop("class_name")
    return getattr(globals(), class_name)(**args_dict)
