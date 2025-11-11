# DEPRACATED: Use Gemini-flash-lite-latest directly to parse HTML to JSON instead of coding agent(too hard to maintain).


from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from utils import init_chat_model_from_modelscope,init_kimi_k2
import requests
import pprint
import os

@tool
def get_raw_html_content(url: str) -> str:
    '''
    Get the HTML content of a webpage.

    Args:
        url: the URL of the webpage.
    Return:
        the HTML content of the webpage.
    '''
    print("!!!!!Getting HTML content for URL:", url)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"
    
@tool
def write_code_to_file(code: str, filename: str) -> str:
    '''
    Write the given code to a file.

    Args:
        code: the code to write.
        filename: the name of the file.
    Return:
        success message.
    '''
    print(f"!!!!!Writing code to file: {filename}")
    try:
        os.makedirs('parsers', exist_ok=True)
        with open(f'parsers/{filename}', 'w') as f:
            f.write(code)
        return f"Code successfully written to {filename}"
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def execute_code_from_file(filename: str, args):
    '''
    Execute the code from the given file to parse HTML content with specify args.

    Args:
        filename: the name of the file containing the code.
        args: the arguments to pass to the code.
    Return:
        the output of the code execution.
    '''
    cmd = ['python', f'parsers/{filename}'] + args
    print(f"!!!!!Executing code from file: {filename} with args: {args}")
    return os.system(' '.join(cmd))  # Simplified for demonstration

class HTMLParserAgent:
    prompt_template = ''
    
    def __init__(self):
        self.agent = create_agent(
            #model="google_genai:gemini-2.5-pro",
            #model=init_chat_model_from_modelscope("Qwen/Qwen3-VL-235B-A22B-Instruct"),
            model=init_kimi_k2(),
            system_prompt='''
                    You are an expert HTML parser writer agent.
                    Your task is to wirte an Python file to extract paper list from the given HTML content and conference name.
                    The HTML content is from academic conference accepted paper list page.
                    You should write a Python function named "parse_html" that takes a single argument "html_content" (a string containing the HTML content) and returns a list of paper titles.
                    the parser should be a py file and accepte an --url=url args to specify which URL to parse.
                    The return of the parser should be an list of dict: [{"title":title, "abstract":abstract}].
                    
                    You should use `python parsers/conference.py --url=url` to run the parser and to get the parsed content.
                    
                    If you need get the HTML content of a webpage, use the tool "get_raw_html_content" to fetch it.
                    If you need to write code to file, use the tool "write_code_to_file" to do it, the file name should be the Conference name.
                    You should use tool `execute_code_from_file` to run the code you wrote.
                    If the code is not working correctly, you need rewrite the code totally to the file and try again since you can only write code to file but can't modify the file.
                    Make sure to handle different HTML structures and edge cases.
            ''',
            tools=[get_raw_html_content, write_code_to_file, execute_code_from_file],
        )

    def invoke(self, url, conference_name):
        return self.agent.invoke(input={
            "messages": [
                HumanMessage(
                    content=f'''
                            Here is the HTML content and conference name:
                            HTML Url:
                            {url}

                            Conference Name:
                            {conference_name}
                             
                            The script you write should named as {conference_name}.py inside parsers/ folder.
                            '''
                )
            ]
        })
    
    