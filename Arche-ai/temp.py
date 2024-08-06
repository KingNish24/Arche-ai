from llms import GroqLLM
from agents import Agent
from tools import OwnTool, get_weather, get_current_time, web_search

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
    params={"query": {"type": "string", "description": "The query to do search for"}}
)

time_tool = OwnTool(
    func=get_current_time,
    description="Provides the current time."
)

# Initialize the language model instance

llm_instance = GroqLLM()


while True:
    # Create the agent with multiple tools
    agent = Agent(
        llm=llm_instance,
        tools=[gcd_tool, web_tool, time_tool],
        name="Chatbot",
        description="A powerful Chatbot",
        sample_output="",
        task=input(">>> "),
        verbose=True
    )

    # Run the agent and print the result
    result = agent.run()
    # print(result)
