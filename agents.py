import os
from dotenv import load_dotenv
from anthropic import Anthropic
import json
import re

load_dotenv()
client = Anthropic()

#HELPER FUNCS 
def extract_section(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# TOOLS AND SCHEMAS
submit_offer_schema = {
    "name": "submit_offer",
    "description": (
        "Record your official negotiating position for this turn. You MUST call this "
        "every turn. This tool call is your binding, authoritative position. Your message "
        "and this tool call must always state the same number — never contradict yourself. "
        "(If they ever accidentally differ, the tool call is treated as authoritative, but "
        "this should never happen.) Returns your action and associated amount."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["discuss", "hold", "propose", "accept", "walk_away"],
                "description": (
                    "Your move this turn. "
                    "'discuss': talking with no number on the table yet (opening moves); number MUST be null. "
                    "'propose': a new or changed offer; number MUST be a positive whole dollar amount. "
                    "'hold': restate your existing offer unchanged; number MUST equal your last proposed amount. "
                    "'accept': agree to a number already offered by the other party; number MUST exactly match their standing offer. "
                    "'walk_away': quit permanently, no deal is possible; number MUST be null and this is irreversible."
                )
            },
            "number": {
                "type": ["integer", "null"],
                "description": (
                    "The whole-dollar amount for this position. MUST be null when action is "
                    "'discuss' or 'walk_away'. MUST be a positive whole number (no decimals, no "
                    "negatives) when action is 'propose', 'hold', or 'accept'."
                )
            }
        },
        "required": ["action", "number"]
    }
}

def submit_offer(action, number):
    valid_actions = ["discuss", "hold", "propose", "accept", "walk_away"]
    if action not in valid_actions:
        raise ValueError(f"Unsupported action: {action}")
    if action in ["discuss","walk_away"] and number is not None:
        raise ValueError(f"{action} requires number to be null")
    if isinstance(number, float) and number.is_integer():
        number = int(number)
    if action in ["propose", "hold", "accept"] and not isinstance(number,int):
        raise ValueError(f"{action} requires an integer, but got {number!r} (type: {type(number).__name__})")
    if action in ["propose", "hold", "accept"] and number < 0:
        raise ValueError(f"{number} must be a positive integer") 
    if action in ["propose", "hold", "accept"] and isinstance(number,bool):
        raise ValueError(f"number cannot be a boolean; this is your proposed dollar offer")
    return {"number": number, "action": action}


#AGENT CLASS
class NegotiatingAgent:
    def __init__(self, system_prompt):
        self.system_prompt= system_prompt
        self.messages = []

    def reply(self, incoming_message):
        self.messages.append({"role": "user", "content": incoming_message})

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=5000,
                system=self.system_prompt,
                messages=self.messages,
                tools=[submit_offer_schema],
                tool_choice={"type": "auto"},
            )

            # sort response blocks by type
            text_blocks = [b for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    
            # pull the prose message out of the text
            combined_text = "".join(b.text for b in text_blocks)
            your_message = extract_section(combined_text, "your_message")
            thoughts_on_counterpart = extract_section(combined_text, "thoughts_on_counterpart")
            reasoning = extract_section(combined_text, "reasoning")
            fair_value_estimate = extract_section(combined_text,"fair_value_estimate")
            opponent_limit_estimate = extract_section(combined_text,"opponent_limit_estimate")

            if your_message is None or reasoning is None or thoughts_on_counterpart is None or fair_value_estimate is None:                
                print("DEBUG - raw model text on missing-tags failure:")
                print(combined_text)
                print("---")
                self.messages.append({
                    "role": "assistant",
                    "content": response.content})
                
                correction_blocks = []
                # If it called the tool, we must answer that tool_use — truthfully, no error.
                for block in tool_use_blocks:
                    correction_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Received. But your text response was missing required tags — see below.",
                        "is_error": False,
                    })
                #plus the correciton as text
                correction_blocks.append({
                    "type": "text",
                    "text": "You did not include the required <fair_value_estimate>,<opponent_limit_estimate>,<thoughts_on_counterpart>, <reasoning>, and <your_message> tags. You must include all five. Please redo this turn with them.",
                })
                self.messages.append({"role":"user", "content": correction_blocks})
                last_error = "missing tags, reasoning or thoughts on counterpart"
                continue

            # always append the assistant's full response first (protocol requirement)
            self.messages.append({"role": "assistant", "content": response.content})

            # --- CASE: wrong number of tool calls ---
            if len(tool_use_blocks) == 0:
                # no tool_use to attach a tool_result to; send a plain correction message
                self.messages.append({
                    "role": "user",
                    "content": "You did not call submit_offer. You MUST call submit_offer exactly once this turn. Please try again.",
                })
                last_error = "no tool call"
                continue

            if len(tool_use_blocks) > 1:
                # every tool_use must be answered with a tool_result, so answer them all with an error
                error_results = []
                for block in tool_use_blocks:
                    error_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Error: you called submit_offer more than once. Call it exactly once. Please try again.",
                        "is_error": True,
                    })
                self.messages.append({"role": "user", "content": error_results})
                last_error = "more than one tool call"
                continue

            # --- exactly one tool call: validate it ---
            tool_use = tool_use_blocks[0]
            tool_input = tool_use.input
            action = tool_input.get("action")   # .get() -> None if missing, avoids KeyError
            number = tool_input.get("number")

        
            try:
                tool_output = submit_offer(action=action, number=number)
            except ValueError as e:
                last_error = e
                if action is None:
                    self.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "Your submit_offer call was missing the required 'action' field. You MUST include both 'action' and 'number' every time you call submit_offer. Please try again.",
                            "is_error": True,
                        }],
                    })
                    continue
                # invalid values (or missing action) -> send error tool_result, retry
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: {e}. Please call submit_offer again with valid arguments.",
                        "is_error": True,
                    }],
                })
                continue

            # --- success: append the good tool_result and return ---
            self.messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(tool_output),
                    "is_error": False,
                }],
            })
            return {"message": your_message, "action": action, "number": number, 
                    "thoughts_on_counterpart": thoughts_on_counterpart,
                    "reasoning": reasoning,
                    "fair_value_estimate": fair_value_estimate,
                    "opponent_limit_estimate": opponent_limit_estimate}

        # exhausted all retries without a valid tool call
        raise RuntimeError(f"Model failed to produce a valid submit_offer call after retries. Last error: {last_error}")