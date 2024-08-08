# Agent Class

The `Agent` class is designed to create an AI agent that can intelligently interact with various tools and an underlying LLM (Large Language Model). This class can dynamically manage tools, execute tasks, and generate responses based on provided inputs.

## Features

- **Dynamic Tool Management:** Add or remove tools as needed without restarting the agent.
- **Tool Selection:** Automatically selects the appropriate tool based on the task description.
- **JSON Response Handling:** Generates and processes JSON-based outputs for interaction with tools.
- **Verbose Mode:** Detailed logging for debugging and understanding the decision-making process.
- **LLM Integration:** Uses the GroqLLM for generating AI-driven responses.

<h1>How to use:</h1>

<h3>Example of a tool function and agent interaction</h3>

```python

# tool with One Parameter

from arche.llms import GroqLLM
from arche.tools import OwnTool
from arche.agents import Agent

def greet(name: str) -> str:
    return f"Hello, {name}!"

greet_tool = OwnTool(
    func=greet,
    description="A tool that greets a user.",
    params={"name": {"type": "string", "description": "The name of the user."}}
)

agent = Agent(
    llm=GroqLLM(),
    tools=[greet_tool],
    name="GreetingAgent",
    description="An agent that greets users.",
    task="Greet John.",
    verbose=True
)

result = agent.run()
print(result)

```

<h3>Multi tool use with multi parameters</h3>

```python
from arche.llms import Gemini
from arche.agents import Agent
from arche.tools import OwnTool, get_current_time, web_search

def gcd(a:int, b:int):
    """
    Calculate the Greatest Common Divisor (GCD) of two numbers using the Euclidean algorithm.

    Parameters:
    a (int): The first number.
    b (int): The second number.

    Returns:
    int: The GCD of the two numbers.
    """
    while b:
        a, b = b, a % b
    return a
    # return a+b

# Define the tools using the OwnTool class
gcd_tool = OwnTool(
    func=gcd,
    description="Provides the gcd of two provided numbers",
    params={"a": {"type": "int", "description": "The first number includes only number such as 1 ,2"}, "b": {"type": "int", "description": "The second number such as 1,2 ,3"}}
)

web_tool = OwnTool(
    func=web_search,
    description="Provides the current web results from Google for the given query, best for getting real-time data.",
    params={"hello": {"type": "string", "description": "The query to do search for"}}
)

time_tool = OwnTool(
    func=get_current_time,
    description="Provides the current time.",
)#don't provide the param section for no param tools

# Initialize the language model instance

llm_instance = Gemini()

# Create the agent with multiple tools
agent = Agent(
  llm=llm_instance,
  tools=[time_tool, gcd_tool, web_tool],
  name="Chatbot",
  description="A powerful Chatbot",
  sample_output="",
  task=input(">>> "),
  verbose=False,
  memory=True
)

# Run the agent and print the result
result = agent.run()
print(result)

```
