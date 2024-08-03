from llms import GroqLLM
from typing import Type, List, Optional
import json
from agents import Agent
from colorama import Fore, Back, Style

class AgentNetwork:
    def __init__(
        self,
        llm: Type[GroqLLM],
        agents: List[Agent],
        name: str = "Agent Network",
        description: str = "A network of agents working together to perform tasks.",
        task: str = "question or task for the agent network",
        verbose: bool = False,  # New parameter for debugging output
    ) -> None:
        """
        Initialize the AgentNetwork with the given parameters.

        Args:
            llm (Type[GroqLLM]): The language model to use.
            agents (List[Agent]): The list of agents to use for the task.
            name (str): The name of the agent network.
            description (str): The description of the agent network.
            task (str): The task to perform.
            verbose (bool): Whether to enable verbose logging.
        """
        self.llm = llm
        self.agents = agents
        self.name = name
        self.description = description
        self.task_to_do = task
        self.verbose = verbose

        self.llm.__init__(
            system_prompt=f"""
You are {self.name}, {self.description}.
"""
        )

    def _run_agents(self, task) -> str:
        """Handles tasks that require using multiple agents."""
        self.agents_info = "\n".join([
            f"Agent Name: {agent.name} - {agent.description}"
            for agent in self.agents
        ])

        response = self.llm.run(task).strip()
        if self.verbose:
            print(f"{Fore.YELLOW}Raw Network LLM Response:{Style.RESET_ALL} {response}")

        try:
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            action = json.loads(response)

            # results = {}
            for call in action.get("agent_calling", []):
                self.agent_name = call["agent_name"]
                self.task_description = call["task_description"]

                if self.verbose:
                    print(f"{Fore.BLUE}Parsed JSON:{Style.RESET_ALL} {call}")
                    print(f"{Fore.CYAN}Extracted Agent Name:{Style.RESET_ALL} {self.agent_name}")
                    print(f"{Fore.CYAN}Extracted Task Description:{Style.RESET_ALL} {self.task_description}")

                # Find the agent by name
                agent = next((agent for agent in self.agents if agent.name.lower() == self.agent_name.lower()), None)
                if agent is None:
                    raise ValueError(f"Agent '{self.agent_name}' not found.")

                # Update the agent's task and run it
                agent.task_to_do = self.task_description
                agent_response = agent.run()
                # results[agent_name] = agent_response
                if self.verbose:
                    print(f"{Fore.GREEN}Agent Response ({self.agent_name}):{Style.RESET_ALL} {agent_response}")

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
            print(f"{Fore.GREEN}Agent_RESULTS:\n{agent_response}{Style.RESET_ALL}")
            print()
        return agent_response

    def run(self) -> str:
        """Main execution logic of the agent network."""
        self.llm.__init__(
            system_prompt=f"""
You are an AI assistant designed to coordinate multiple agents to perform tasks.

***Your task is to understand the given agents, their descriptions, and use them appropriately in your responses.***

***Available Agents:***
{self.agents_info}

***Instructions:***
1. Read the task description carefully.
2. Identify the required agents for the task.
3. Replace the placeholders in the JSON structure with the actual values provided in the task.
4. Always respond in the exact same JSON structure format with the agent_name and task description, nothing else.

***JSON Structure:***
{{
    "agent_calling": [
        {{
            "agent_name": "name of the agent, give the exact name as provided because it is case sensitive; <agent_name>",
            "task_description": "understand the agent's task and then give the suitable word or sentence accordingly; <task_description>"
        }}
    ]
}}

***Example Task:***
- Note this is just an example it is not necessary that you are having the same agents provided. Provided Agents are mentioned above.
Task: Get the weather for New York and write a report about it.
You:
{{
    "agent_calling": [
        {{
            "agent_name": "web_surfer",
            "task_description": "Get the weather for New York"
        }}
    ]
}}

Compiler: gives info to you like `Agent(web_surfer): <info>`

You again:
{{
    "agent_calling": [
        {{
            "agent_name": "writer",
            "task_description": "write a report on the weather of new york here is the data:\n <info>"
        }}
    ]
}}


""")
        response = self._run_agents()

        try:
            summary = self.llm.run(f"[QUERY]\n{self.task_to_do}\n\n[Agent({self.agent_name})]\n{response}")
            if self.verbose:
                print()
                print(summary)
                print()
            elif self.verbose != True:
                return summary
        except Exception as e:
            return f"Failed to get summary: {str(e)}."
