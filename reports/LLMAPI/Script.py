#pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
from dotenv import load_dotenv
from openai import OpenAI
from task_label import TaskLabelPrompt
import json

load_dotenv()
openai_client = OpenAI()

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def get_response_from_openai(user_message: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are technical expert, you determine whether something is a technical problem or not. You answear only with one of two allowed responses: ""problem"" or ""not problem"" You can only repond with either ""problem"" or ""not problem"", without any additional characters."},
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content


def read_json_file(file_path):
    #Reads a JSON file and returns its content as a dictionary.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    input_json = "C:\\Users\\menmi\\OneDrive\\Dokumenty\\AIAJ\\ai_inventiveness\\reports\\LLMAPI\\classification_task.json"
    data = read_json_file(input_json)
    for x in data:
        message = x["description"]
        x["Ollama_Response"]=get_response_from_ollama(message)
        print(x["Ollama_Response"])
    with open(input_json.replace(".json","") + "_appended.json", 'x') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    main()