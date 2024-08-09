import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

class Gemini:
    USER = "user"
    MODEL = "model"

    def __init__(self,
                 messages: list[dict[str, str]] = [],
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.0,
                 system_prompt: str|None = None,
                 max_tokens: int = 2048,
                 connectors: list[str] = [],
                 verbose: bool = False,
                 api_key: str|None = None
                 ):
        safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        self.api_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.connectors = connectors
        self.verbose = verbose
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "text/plain",
            }
        )
        if self.system_prompt:
            self.add_message(self.MODEL, self.system_prompt)

    def run(self, prompt: str) -> str:
        self.add_message(self.USER, prompt)
        chat_session = self.client.start_chat(history=self.messages)
        response = chat_session.send_message(prompt)
        self.messages.pop()  # Remove the user prompt to avoid duplication
        r = response.text
        if self.verbose:
            print(r)
        return r

    def add_message(self, role: str, content: str) -> None:
        # Adjusting message structure for Gemini
        self.messages.append({"role": role, "parts": [content]})

    def __getitem__(self, index) -> dict[str, str]|list[dict[str, str]]:
        if isinstance(index, slice):
            return self.messages[index]
        elif isinstance(index, int):
            return self.messages[index]
        else:
            raise TypeError("Invalid argument type")

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            self.messages[index] = value
        elif isinstance(index, int):
            self.messages[index] = value
        else:
            raise TypeError("Invalid argument type")
    
    def reset(self) -> None:
        """
        Reset the system prompts and messages

        Returns
        -------
        None
        """
        self.__init__(system_prompt=None,
                      messages=[])
        self.messages = []
        self.system_prompt = None

if __name__ == "__main__":
    q = input(">>> ")
    llm = Gemini(system_prompt="You are a coding expert. Answer questions asked by the USER. Optimize the user's code. Make this code fast, better, and advanced according to the user's requirements. Use concise and crisp comments and warnings. Try your best to make the code the best and advanced, and easy for users.", verbose=True, messages=[{'role': 'user', 'parts': 'open facebook'}, {'role': 'model', 'parts': '```python\nimport webbrowser\ntry:\n    webbrowser.open("https://www.facebook.com/")\n    print("facebook opened")\nexcept Exception as e:\n    print(f"Error: {str(e)}")\nprint("CONTINUE")\n```\n'}, {'role': 'user', 'parts': 'LAST SCRIPT OUTPUT:\nfacebook opened\nCONTINUE'}, {'role': 'model', 'parts': "```python\nprint('I have opened Facebook in your web browser.')\n```"}, {'role': 'user', 'parts': 'LAST SCRIPT OUTPUT:\nI have opened Facebook in your web browser.'}, {'role': 'user', 'parts': 'generate random number'}, {'role': 'model', 'parts': '```python\nimport random\n\nrandom_number = random.randint(1, 100)\nprint(f\'Generated random number: {random_number}\')\nprint("CONTINUE")\n```'}, {'role': 'user', 'parts': 'LAST SCRIPT OUTPUT:\nGenerated random number: 42\nCONTINUE'}, {'role': 'model', 'parts': "```python\nprint('Generated a random number between 1 and 100. The number is 42.')\n```"}, {'role': 'user', 'parts': 'LAST SCRIPT OUTPUT:\nGenerated a random number between 1 and 100. The number is 42.'}])
    llm.add_message(Gemini.USER, q)
    print(llm.run(q))
    print(llm.messages)
