from typing import Callable, Optional

class OwnTool:
    def __init__(self, func: Callable, description: str, params: Optional[str] = None):
        """
        Initialize the OwnTool with the given parameters.

        Args:
            func (Callable): The tool function to use.
            description (str): The description of the tool.
            params (Optional[str]): The parameters for the tool.
        """
        self.func = func
        self.description = description
        self.params = params

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
