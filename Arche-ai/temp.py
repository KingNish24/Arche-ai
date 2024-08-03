# main.py

from llms import GroqLLM
from agents import Agent
from tools import OwnTool
from tools import get_weather, get_current_time, web_search

# Define the tools using the OwnTool class
weather_tool = OwnTool(
    func=get_weather,
    description="Provides the current weather forecast for a given location.",
    params="location:str"
)

web_tool = OwnTool(
    func=web_search,
    description="Provides the current web results from the google for the given query, best for getting real time data.",
    params="query to search the web:str"
)

time_tool = OwnTool(
    func=get_current_time,
    description="Provides the current time.",
    params=None
)

# Initialize the language model instance
llm_instance = GroqLLM()

while True:
# Create the agent with multiple tools
    agent = Agent(
        llm=llm_instance,
        tools=[web_tool, weather_tool, time_tool],
        name="ChatBot",
        description="A powerful AI Assistant",
        sample_output="",
        task=input(">>> "),
        verbose=True
    )

    # Run the agent and print the result
    result = agent.run()
    # print(result)