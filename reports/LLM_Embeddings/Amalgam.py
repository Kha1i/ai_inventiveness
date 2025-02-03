"""
Example script to extract technical contradictions from a problem description   
"""

import os
import json
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field

from service.openai_service import OpenAIService
from service.task_label import TaskLabelPrompt
from src.service.embedding_service import EmbeddingService

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))
openai_service = OpenAIService(provider="groq")
task_label_prompt = TaskLabelPrompt()

script_path = os.path.dirname(os.path.abspath(__file__))
parameters_txt_path = os.path.join(script_path, "parameters.txt")
parameters_json_path = os.path.join(script_path, "parameters.json")
embedding_service = EmbeddingService(model="mxbai-embed-large:335m")


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


def embed_parameters(parameters: List[str]) -> dict:
    """
    Embed the TRIZ standard parameters.
    """
    output_list = []
    progress_bar = tqdm(enumerate(parameters), total=len(parameters), desc="Embedding parameters")
    for ind, parameter in progress_bar:
        output_list.append(
            {
                "uuid": str(uuid4()),
                "index": ind,
                "parameter": parameter,
                "embedding": embedding_service.create_embedding(parameter),
            }
        )
    return output_list

def search_parameters(query: str, parameters: List[dict], n: int = 1) -> List[dict]:
    """
    Search for the closest parameters.
    """
    query_embedding = embedding_service.create_embedding(query)
    distances = embedding_service.find_n_closest(
        query_vector=query_embedding,
        embeddings=[param["embedding"] for param in parameters],
        n=n,
    )
    return [parameters[dist["index"]] for dist in distances]

# ----------------------------------------------
# Main function
# ----------------------------------------------

def main():
    """
    Main function to extract technical contradictions
    """
    # Read the input file
    input_json = "C:\\Users\\menmi\\OneDrive\\Dokumenty\\AIAJ\\ai_inventiveness\\reports\\LLM_Embeddings\\extraction_task.json"
    data = read_json_file(input_json)
    
    model = "llama-3.3-70b-versatile"
    
    
    if os.path.exists(parameters_json_path):
        print("Loading existing embeddings from JSON file...")
        try:
            with open(parameters_json_path, "r", encoding='utf-8') as file:
                parameters = json.load(file)
            print(f"Successfully loaded {len(parameters)} embeddings")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load existing embeddings: {e}")
            return
    else:
        print("Embeddings file not found. Generating new embeddings...")
        # Load the TRIZ standard parameters
        try:
            with open(parameters_txt_path, "r", encoding='utf-8') as file:
                parameters_txt = [line.strip() for line in file.readlines()]
            print(f"Successfully loaded {len(parameters_txt)} parameters from text file")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load parameters from text file: {e}")
            return

        # Generate and save embeddings
        try:
            parameters = embed_parameters(parameters_txt)
            with open(parameters_json_path, "w", encoding='utf-8') as file:
                json.dump(parameters, file, indent=4)
            print(f"Successfully generated and saved {len(parameters)} embeddings to {parameters_json_path}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to generate and save embeddings: {e}")
            return
        
        
        
 
    for x in data:
        print("\n\nProblem Description:")
        print(x["description"])
        messages = x["description"]
        contradiction_model = openai_service.create_structured_output(
            model=model,
            messages=task_label_prompt.compile_messages(messages),
            response_model=TechnicalContradiction
        )

        x["Action"] = contradiction_model.action
        x["Positive Effect"] = contradiction_model.positive_effect
        x["Positive Effect Embed"] =  ([param["parameter"] for param in search_parameters(x["Positive Effect"], parameters)])
        x["Negative Effect"] = contradiction_model.negative_effect
        x["Negative Effect Embed"] = ([param["parameter"] for param in search_parameters(x["Negative Effect"], parameters)])
        print("\nTechnical Contradiction:")
        print(f"Action: {x["Action"]}")
        print(f"Positive Effect: {x["Positive Effect"]}")
        print(f"Positive Effect Embed:" .join([param["parameter"] for param in search_parameters(x["Positive Effect"], parameters)]))
        print(f"Negative Effect: {x["Negative Effect"]}")
        print(f"Negative Effect Embed:" .join([param["parameter"] for param in search_parameters(x["Negative Effect"], parameters)]))
        
        
        
    with open(input_json.replace(".json","") + "_appended.json", 'x') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    main()