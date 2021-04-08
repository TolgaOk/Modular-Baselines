""" Torch optimizer that works as eligibility trace. These traces can be seen as momentum term
but one per environment. Hence, the optimizer requires the environment size at initialization.
"""

