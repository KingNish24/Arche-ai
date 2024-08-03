# main.py

from llms import GroqLLM
from agents import Agent, AgentNetwork
from tools import get_weather, get_current_time, OwnTool, web_search

# Define the tools using the OwnTool class
weather_tool = OwnTool(
    func=get_weather,
    description="Provides the current weather forecast for a given location.",
    params="location:str"
)

web_search_tool = OwnTool(
    func=web_search,
    description="Provides the current web_search results from the google",
    params="query to search the web:str"
)

time_tool = OwnTool(
    func=get_current_time,
    description="Provides the current time.",
    params=None
)

# Initialize the language model instance
llm_instance = GroqLLM()

# Create the agents
web_surfer_agent = Agent(
    llm=llm_instance,
    tools=[web_search_tool],
    name="Web_Surfer",
    description="Surfs the web to gather information.",
    sample_output="The web results are: {web_results}",
    task="",
    verbose=True
)

writer_agent = Agent(
    llm=llm_instance,
    tools=[],
    name="Writer",
    description="Writes reports based on the information provided.",
    sample_output="The weather in {location} is {weather}. You should {your_suggestion}.",
    task="Write a report about the weather in Kolkata.",
    verbose=True
)

analyst_agent = Agent(
    llm=llm_instance,
    tools=[],
    name="Analyst",
    description="Analyzes the writer's response to determine if it is good or not.",
    sample_output="The report is {good_or_bad}.",
    task="Analyze the report about the weather in Kolkata.",
    verbose=True
)

# Create the agent network
agent_network = AgentNetwork(
    llm=llm_instance,
    agents=[web_surfer_agent, writer_agent, analyst_agent],
    name="Company Agent Network",
    description="A network of agents working together to perform tasks.",
    task="Get the weather for Kolkata using agents, write a report about it, and analyze the report.",
    verbose=True
)

# Run the agent network and print the result
result = agent_network.run()
# print(result)
