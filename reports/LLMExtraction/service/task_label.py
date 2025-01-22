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
    You are technical scientific expert, spocialized in extracting information from scientific papers 
    </prompt_objective>  

    <prompt_rules>  
    - ALWAYS list only title of publication as is in provided text
    - NEVER change title compared to one in provided paper
    - ALWAYS seperate keywords with "; "
    - NEVER list multiple keywords without "; " between them  
    - ALWAYS list at least 3 keywords
    - NEVER list more than 8 keywords
    - ALWAYS list "Energy efficiency" "Pneumatic" "Vehicles" "Braking" "Suspension" "Fuel Consumption" as keywords when it matches the paper text
    - NEVER list other similar keywords when one of "Energy efficiency" "Pneumatic" "Vehicles" "Braking" "Suspension" "Fuel Consumption" would also fit
    - ALWAYS Remain consice as much as possible when creasting summary of paper.
    </prompt_rules>  

    <prompt_examples>  
    USER: A car trailer manufacturer aims to increase the payload capacity of their flatbed trailers. 
    By using high-strength steel in the frame construction, they have successfully increased the maximum load by 20%.
    The new design, while more capable, has resulted in a 15% increase in overall trailer weight.
    This additional weight reduces fuel efficiency for the towing vehicle and may require drivers
    to upgrade to more powerful tow vehicles for maximum capacity loads.  
    AI: Action: increasing payload capacity of flatbed trailers by using high-strength steel
    Positive Effect:  20% increase in maximum load
    
    Negative Effect:  15% increase in overall trailer weight;
                    Reduced fuel efficiency;
                    Might require to upgrade to more powerful tow vehicle;
    
    USER: Pneumatic suspension needs quiceker deaeration times. Manufacturer implements less restrictive silencer. 
    Thanks to that time to deaerate decreases 10%. This however made noise increase 20% and created possible issues with ingress protection.
    AI: Action: increasing payload capacity of flatbed trailers by using high-strength steel
    Positive Effect:  10% deacrease in deaeration time;
    
    Negative Effect:  20% increase in noise;
                      Possibly worse ingress protection;


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
