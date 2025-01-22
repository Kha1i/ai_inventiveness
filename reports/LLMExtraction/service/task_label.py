"""
This is a prompt for the task of converting text into bullet points as a Python list.
"""
from pydantic import BaseModel


class TaskLabelPrompt:
    """A prompt class for classifying tasks into predefined labels."""

    # -----------------------------------------------------------------------------
    # System Prompt
    # -----------------------------------------------------------------------------

    SYSTEM_PROMPT = """
    [Classify Task into Predefined Label]

    <prompt_objective>  
    You are technical expert, spocialized in technical contradiction defining. 
    </prompt_objective>  

    <prompt_rules>  
    - ALWAYS list all effects in the desription
    - ALWAYS list positive and negative effects one for each line with "-" at beggining.
    - NEVER list effect without "-" at beggining
    - NEVER list action with "-" at beggining
    - ALWAYS list action wihtout spliting lines.
    - ONLY use numbers when possible.  
    - NEVER use text description if it can be described numericly.
    - Be as precise and specific as possible when describing problems.
    - Remain consice as much as possible while remaining precision of description.
    </prompt_rules>  

    <prompt_examples>  
    USER: A car trailer manufacturer aims to increase the payload capacity of their flatbed trailers. 
    By using high-strength steel in the frame construction, they have successfully increased the maximum load by 20%.
    The new design, while more capable, has resulted in a 15% increase in overall trailer weight.
    This additional weight reduces fuel efficiency for the towing vehicle and may require drivers
    to upgrade to more powerful tow vehicles for maximum capacity loads.  
    AI: Action: increasing payload capacity of flatbed trailers by using jhigh-strength steel
    Positive Effect:  - 20% increase in maximum load
    Negative Effect:  - 15% increase in overall trailer weight
                      - Reduced fuel efficiency
                      - Might require to upgrade to more powerful tow vehicle

    </prompt_examples> 
    """

    # -----------------------------------------------------------------------------
    # Pydantic Model
    # -----------------------------------------------------------------------------

    class OutputModel(BaseModel):
        """
        Pydantic model for the answers.
        """
        label: str

    # -----------------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------------

    @classmethod
    def get_messages(cls) -> list[dict]:
        """
        Get the messages.
        """
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": "{{text}}"},
        ]

    @classmethod
    def compile_messages(cls, text: str) -> list[dict]:
        """
        Compile the messages with the given context by replacing template variables.

        Args:
            text: The text content for the user message

        Returns:
            list[dict]: List of message dictionaries with populated template variables
        """
        messages = cls.get_messages()

        # Replace template variables in user message
        messages[1]["content"] = messages[1]["content"].replace("{{text}}", text)

        return messages
