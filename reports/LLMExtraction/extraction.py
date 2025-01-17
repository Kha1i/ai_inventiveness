"""
Example script to extract technical contradictions from a problem description   
"""
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from service.openai_service import OpenAIService
from service.task_label import TaskLabelPrompt

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))
openai_service = OpenAIService(provider="groq")

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

# ----------------------------------------------
# Main function
# ----------------------------------------------

def main():
    """
    Main function to extract technical contradictions
    """
    # Read the input file
    with open(os.path.join(script_dir, "problem_desc.txt"), "r", encoding="utf-8") as file:
        data = file.read()

    print("Problem Description:")
    print(data)

    # Extract technical contradictions
    messages=[
            {
                "role": "system",
                "content": "You are technical expert, spocialized in technical contradiction defining."
            },
            {
                "role": "user",
                "content": data
            }
        ]

    model = "llama-3.3-70b-versatile"

    contradiction_model = openai_service.create_structured_output(
        model=model,
        messages=messages,
        response_model=TechnicalContradiction
    )

    print("\nTechnical Contradiction:")
    print(f"Action: {contradiction_model.action}")
    print(f"Positive Effect: {contradiction_model.positive_effect}")
    print(f"Negative Effect: {contradiction_model.negative_effect}")

if __name__ == "__main__":
    main()
