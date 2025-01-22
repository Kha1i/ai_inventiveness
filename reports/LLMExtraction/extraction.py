"""
Example script to extract technical contradictions from a problem description   
"""
import os
import json
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from service.openai_service import OpenAIService
from service.task_label import TaskLabelPrompt

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))
openai_service = OpenAIService(provider="groq")
task_label_prompt = TaskLabelPrompt()
# ----------------------------------------------
# Pydantic Model
# ----------------------------------------------

class TechnicalContradiction(BaseModel):
    """
    Technical Contradiction extraction model
    """
    action: str = Field(description="The action that causes the contradiction")
    positive_effect: str = Field(description="The positive effect of the action")
    negative_effect: str = Field(description="The negative effect of the action")

def read_json_file(file_path):
    #Reads a JSON file and returns its content as a dictionary.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
# ----------------------------------------------
# Main function
# ----------------------------------------------

def main():
    """
    Main function to extract technical contradictions
    """
    # Read the input file
    input_json = "C:\\Users\\menmi\\OneDrive\\Dokumenty\\AIAJ\\ai_inventiveness\\reports\\LLMExtraction\\extraction_task.json"
    data = read_json_file(input_json)

    model = "llama-3.3-70b-versatile"

 
    for x in data:
        print("Problem Description:")
        print(x["description"])
        messages = x["description"]
        contradiction_model = openai_service.create_structured_output(
            model=model,
            messages=task_label_prompt.compile_messages(messages),
            response_model=TechnicalContradiction
        )

        x["Action"] = contradiction_model.action
        x["Positive Effect"] = contradiction_model.positive_effect
        x["Negative Effect"] = contradiction_model.negative_effect
        
        print("\nTechnical Contradiction:")
        print(f"Action: {contradiction_model.action}")
        print(f"Positive Effect: {contradiction_model.positive_effect}")
        print(f"Negative Effect: {contradiction_model.negative_effect}")
        
        
        
    with open(input_json.replace(".json","") + "_appended.json", 'x') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    main()