from llms import GroqLLM
from tools import OwnTool, web_search
from typing import Type, List, Optional
import json
from colorama import Fore, Back, Style

class Agent:
    def __init__(
        self,
        llm: Type[GroqLLM],
        tools: List[OwnTool],
        name: str = "agent's name",
        description: str = "agent's description",
        sample_output: str = "expected output from agent",
        task: str = "question or task for the agent",
        verbose: bool = False,  # New parameter for debugging output
    ) -> None:
        """
        Initialize the Agent with the given parameters.

        Args:
            llm (Type[GroqLLM]): The language model to use.
            tools (List[OwnTool]): The list of tools to use for the task.
            name (str): The name of the agent.
            description (str): The description of the agent.
            sample_output (str): The expected output format.
            task (str): The task to perform.
            verbose (bool): Whether to enable verbose logging.
        """
        self.llm = llm
        self.tools = tools
        self.name = name
        self.description = description
        self.sample_output = sample_output
        self.task_to_do = task
        self.verbose = verbose

        self.llm.__init__(
            system_prompt=f"""
You are {self.name}, {self.description}.
"""
        )

    def _run_no_tool(self) -> str:
        """Handles tasks without any tools."""
        self.llm.__init__(
            system_prompt=f"""
You are {self.name}, a {self.description}.

### OUTPUT STYLE:
{self.sample_output}

***If output style not mentioned clearly generate in markdown format.***
"""
        )
        return self.llm.run(self.task_to_do)

    def _run_with_tools(self) -> str:
        """Handles tasks that require using multiple tools."""
        tools_info = "\n".join([
            f"Tool Name: {tool.func.__name__} - {tool.description}\nTool Parameters: {tool.params}"
            for tool in self.tools
        ])

        self.llm.__init__(
            system_prompt=f"""
You are an AI assistant designed to generate JSON responses based on provided tools.

***Your task is to understand the given tools, their parameters, and use them appropriately in your responses.***

***Available Tools:***
{tools_info}

***Instructions:***
1. Read the task description carefully.
2. Identify the required parameters for the tools.
3. Replace the placeholders in the JSON structure with the actual values provided in the task.
4. Always respond in the exact same JSON structure format with the tool_name and tool parameter, nothing else.
5. Only give the JSON response and nothing else, not text or something else.

***JSON Structure:***
{{
    "func_calling": [
        {{
            "tool_name": "name of the tool; <tool_name>",
            "parameter": "understand the tools parameter and then give the suitable word or sentence accordingly; <tool_params>"
        }}
    ]
}}

***Example Task:***
- note it is just an example for reference no weather_tool or time_tool is provided until mentioned above.
Task: Get the weather for New York
Expected JSON Response:
{{
    "func_calling": [
        {{
            "tool_name": "weather_tool",
            "parameter": "New York"
        }}
    ]
}}

***For no parameter tools use this:***
{{
    "func_calling": [
        {{
            "tool_name": "time_tool",
            "parameter": ""
        }}
    ]
}}
"""
        )

        response = self.llm.run(self.task_to_do).strip()
        if self.verbose:
            print(f"{Fore.YELLOW}Raw LLM Response:{Style.RESET_ALL} {response}")

        try:
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            action = json.loads(response)

            results = {}
            for call in action.get("func_calling", []):
                tool_name = call["tool_name"]
                print(type(tool_name))
                query = call["parameter"]

                if self.verbose:
                    print(f"{Fore.BLUE}Parsed JSON:{Style.RESET_ALL} {call}")
                    print(f"{Fore.CYAN}Extracted Tool Name:{Style.RESET_ALL} {tool_name}")
                    print(f"{Fore.CYAN}Extracted Parameter:{Style.RESET_ALL} {query}")

                # Find the tool by name
                tool = next((tool for tool in self.tools if tool.func.__name__ == tool_name), None)
                if tool is None:
                    raise ValueError(f"Tool '{tool_name}' not found.")

                if tool.params is None:
                    tool_response = tool()
                else:
                    tool_response = tool(query)

                results[tool_name] = tool_response
                if self.verbose:
                    print(f"{Fore.GREEN}Tool Response ({tool_name}):{Style.RESET_ALL} {tool_response}")

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"{Fore.RED}JSON Decode Error:{Style.RESET_ALL} {str(e)}")
            return f"Failed to decode JSON: {str(e)}."
        except Exception as e:
            if self.verbose:
                print(f"{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")
            return f"Failed to get info: {str(e)}."

        if self.verbose:
            print()
            print(f"{Fore.GREEN}Tool_RESULTS:\n{results}{Style.RESET_ALL}")
            print()
        return results

    def run(self) -> str:
        """Main execution logic of the agent."""
        if not self.tools:
            response = self._run_no_tool()
            if self.verbose:
                print()
                print(response)
                print()
            return response
        else:
            response = self._run_with_tools()

        self.llm.__init__(
            system_prompt=f"""
You are {self.name} a AI agent. You are provided with Output from the tools in JSON format, SO, You task is to use information
from them and give the best possible answer to query. Reply in ChatGPT style and only in text and to the point and uses simple words, Don not Reply in JSON.

## You will be provided with the response of the following tools:
{self.tools}

### OUTPUT STYLE:
{self.sample_output}

##Instructions:
- If output style is not mentioned clearly just reply in the best possible way.

***Remember: Your responses should be in text form only and not JSON or any other format.***
"""
        )

        try:
            summary = self.llm.run(f"[QUERY]\n{self.task_to_do}\n\n[TOOLS]\n{response}")
            if self.verbose:
                print()
                print(summary)
                print()
            return summary
        except Exception as e:
            return f"Failed to get summary: {str(e)}."


web_tool = OwnTool(
    func=web_search,
    description="Provides the current web_search results from the google for the given query",
    params="query to search the web:str"
)

llm_instance = GroqLLM()

while True:
# Create the agent with multiple tools
    agent = Agent(
        llm=llm_instance,
        tools=[web_tool],
        name="WebSurfer",
        description="a ai agent having access to internet",
        sample_output="",
        task=input(">>> "),
        verbose=True
    )

    # Run the agent and print the result
    result = agent.run()