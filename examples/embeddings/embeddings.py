"""
This script shows how to use Word Embeddings to find relevant sentences.
"""
import os
import json
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm

from src.service.embedding_service import EmbeddingService

script_path = os.path.dirname(os.path.abspath(__file__))
parameters_txt_path = os.path.join(script_path, "parameters.txt")
parameters_json_path = os.path.join(script_path, "parameters.json")

# Load the environment variables
load_dotenv()

# Define embedding service
embedding_service = EmbeddingService(model="mxbai-embed-large:335m")

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

def search_parameters(query: str, parameters: List[dict], n: int = 3) -> List[dict]:
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

def main():
    """
    Main function.
    """
    query = input("Enter a query: ")

    # Check if embeddings file exists
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

    # Search for the closest parameters
    closest_parameters = search_parameters(query, parameters)
    print("Closest parameters:\n" + 60 * "-" + "\n" + "\n".join([param["parameter"] for param in closest_parameters]))

if __name__ == '__main__':
    main()
