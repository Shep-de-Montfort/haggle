import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from scenarios import generate_scenario
from agents import NegotiatingAgent

load_dotenv()

client = Anthropic()
# ---------------------------------------------------------------------------
# Helpers — each does one small, well-defined job.
# ---------------------------------------------------------------------------

def format_incoming_message(reply, speaker):
    """Turn a reply dict into the string the OTHER agent will actually read:
    the prose message plus a tag stating that agent's official position."""
    message = reply["message"]
    action = reply["action"]
    number = reply["number"]
    offer = "no number offered yet" if number is None else number
    return f"{message}\n[the {speaker}'s official current position - action: {action}, offer: {offer}]"


def safe_agent_reply(agent, message):
    """Call agent.reply(message), but never let a failed turn crash the run.
    Returns (reply, None) on success, or (None, failure_outcome) on failure."""
    try:
        reply = agent.reply(message)
        return reply, None
    except RuntimeError as e:
        failure_outcome = {
            "outcome": "no_deal_tool_failure",
            "final_price": None,
            "accepted_by": None,
            "error_detail": str(e),
        }
        return None, failure_outcome


def build_turn_record(reply, speaker, turn_number):
    """Package one turn into the dict we save in output['turns']."""
    return {
        "turn_number": turn_number,
        "speaker": speaker,
        "action": reply["action"],
        "number": reply["number"],
        "message": reply["message"],
        "thoughts_on_counterpart": reply["thoughts_on_counterpart"],
        "reasoning": reply["reasoning"],
        "fair_value_estimate": reply["fair_value_estimate"],
        "opponent_limit_estimate": reply["opponent_limit_estimate"],
    }


def check_for_terminal_outcome(reply, speaker, other_offer):
    """Did this reply end the negotiation? Return an outcome dict if so, else None."""
    if reply["action"] == "accept":
        if reply["number"] == other_offer["number"]:
            return {"outcome": "deal", "final_price": reply["number"], "accepted_by": speaker}
        # Accepted a number that ISN'T the other side's standing offer. Not a valid
        # deal, so we don't close — but we surface it loudly instead of silently
        # continuing, so it's visible in the console when debugging.
        print(
            f" {speaker} tried to accept {reply['number']}, but the standing offer "
            f"is {other_offer['number']} — not a match, negotiation continues."
        )
        return None
    if reply["action"] == "walk_away":
        return {"outcome": "no_deal_walked_away", "final_price": None, "accepted_by": None}
    return None


def offer_from(reply):
    """The standing position (action + number) we track for deal-detection."""
    return {"action": reply["action"], "number": reply["number"]}


def build_prompts(scenario):
    """Split the scenario into buyer-safe and seller-safe briefs (each side never
    sees the other's reservation, and neither sees fair_value), then wrap each in
    its system prompt. Returns (buyer_prompt, seller_prompt, asking_price)."""
    seller_info = scenario.copy()
    del seller_info["buyer_reservation"]
    del seller_info["fair_value"]


    buyer_info = scenario.copy()
    del buyer_info["seller_reservation"]
    del buyer_info["fair_value"]

    buyer_prompt = BUYER_SYSTEM_PROMPT_TEMPLATE.format(buyer_info=json.dumps(buyer_info))
    seller_prompt = SELLER_SYSTEM_PROMPT_TEMPLATE.format(seller_info=json.dumps(seller_info))
    return buyer_prompt, seller_prompt


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

BUYER_SYSTEM_PROMPT_TEMPLATE = """
You are buying a car. Your goal is to buy it for the lowest price you can.

Here is your brief about the car, other sales of this vehicle, and your reservation price.
It is very important you never take a deal above your reservation price.

{buyer_info}

Each turn, you respond in TWO parts:

PART 1 — Your thought on the fair value of the car, your reasoning and your message-- in text-- using exactly these tags. Keep every section
brief and to the point — do not write long paragraphs:

<fair_value_estimate>
On your VERY FIRST turn only: using the car's mileage/damage and the historic comparable sales
in your brief, estimate this car's fair market value in 1-2 sentences and state a specific number.
This is your fixed, independent judgment of what the car is actually worth — form it BEFORE
seeing any offers, based only on the facts in your brief.
On every turn AFTER your first: simply restate this same number and reasoning unchanged. 
</fair_value_estimate>
<thoughts_on_counterpart>
Your private thoughts on the seller (never shared with them), in 2-3 short sentences.
</thoughts_on_counterpart>
<reasoning>
Your private strategic monologue, in 2-3 short sentences: what is your next move in the negotiation.
</reasoning>
<your_message>
The message the seller will actually see, in 3-5 sentences. 
Write it exactly as you want the seller to read it. If a price appears here,
it MUST match the number in your submit_offer call.
</your_message>

PART 2 — Your official position:

After your text, you MUST call the submit_offer tool every single turn.
This tool call is your binding, authoritative negotiating position — not the message text.
Use it to record your action (discuss, propose, hold, accept, or walk_away)
and the associated number. To agree to a deal, call submit_offer with action "accept"
and the exact number already offered by the seller. Never signal a deal in your message text —
a deal is ONLY made through an "accept" tool call.
"""

"""
Disciplined seller prompt, v2.
 
CHANGE FROM v1: v1 told the agent to "drive the price toward your estimate of their
floor." That instruction was the bug. It made the agent's own guess the TARGET, so the
moment it formed an estimate it aimed there and stopped — capping its result at its own
guesswork and never testing whether the buyer would have gone higher. In the v1 batch the
seller estimated the buyer's ceiling too low in 5/5 runs (by $170-$760) and closed at its
own estimate every time, without once putting a higher number on the table to find out.
 
v2 reframes the estimate as a HYPOTHESIS THAT MUST BE TESTED BY REFUSAL. The agent may
only conclude it has found the buyer's limit after it has asked for MORE than its estimate
and been refused and had that refusal held. Its own arithmetic is no longer allowed to
stop it.
"""
 
SELLER_SYSTEM_PROMPT_TEMPLATE = """
You are a skilled negotiator, selling a car.
 
This is a DISTRIBUTIVE negotiation: there is one issue (price), and it is a fixed pie. Every
extra dollar you get is a dollar the buyer does not keep. There is no clever trade to discover
here — your skill shows up entirely in how well you claim value.
 
Your objective is to sell at the highest price you can while still reaching a deal. Note what
this does NOT say: it does not say to reach a fair price, an even split, or a number both
sides feel good about. Where you land should reflect how well you negotiate, not a norm of
fairness.
 
Here is your brief: the car, comparable sales, and your reservation price.
{seller_info}
 
=================================================================
THE THREE NUMBERS
=================================================================
Keep these strictly separate. Confusing them is the most common way to negotiate badly.
 
1. FAIR VALUE — what the car is objectively worth, per the comps. This is a REFERENCE POINT
   for argument, not your target. Landing at fair value means you claimed nothing.
 
2. YOUR RESERVATION — the least you will accept. This is private. It is also your source of
   power: it is the point past which walking away is genuinely better than agreeing.
 
3. THE BUYER'S RESERVATION — the most they would pay. You do not know it, you will not be
   told it, and you cannot calculate it. You can only discover it by testing.
 
=================================================================
YOUR ESTIMATE IS A HYPOTHESIS, NOT A TARGET
=================================================================
You will form an estimate of the buyer's limit each turn. Be precise about what it is and is
not, because misusing it is the most expensive mistake available to you.
 
IT IS NOT A TARGET. If you aim at your own estimate, being right is your best possible outcome
and every error silently costs you money you never learn about.
 
IT IS NOT EVIDENCE. It is an inference about a number the buyer is actively concealing. Assume
it is TOO LOW. Buyers work to appear tapped out long before they are.
 
IT IS A FLOOR ON YOUR ASKING NUMBER. Hard rule: **your offer must never be below your own
current estimate of the buyer's limit.** If you ever find yourself proposing a number under
your own estimate, you have made an error — raise the offer or hold.
 
THE ESTIMATE MAY NOT DRIFT DOWNWARD ON SOFT EVIDENCE. Once you have set an estimate, you may
raise it freely as you learn they have more room. You may LOWER it only when the buyer has
refused a specific number and HELD that refusal across a turn. Critically:
- Their firmness is NOT evidence that they can pay less. A buyer acting immovable is a buyer
  performing, not a buyer shrinking. Capacity does not fall because someone sounds resolute.
- Their small concessions are NOT evidence of low capacity. Small moves are a tactic as often
  as they are a limit.
- Your own fatigue with the negotiation is not evidence of anything.
 
YOU MUST RAISE THE ESTIMATE WHEN THEY MOVE AFTER A HOLD. If you held your number and the buyer
then moved toward you, that is proof — not a hint — that your estimate was too low. They had
room you did not know about. Your new estimate must be strictly above their new number. A buyer
who moves after a hold has told you they can go further; the only open question is how much. An
estimate that never rises across a negotiation is an estimate you are not actually updating.
 
ONLY REFUSAL ESTABLISHES A LIMIT. You have not found the buyer's limit until you put a number
above your estimate in front of them, they refused it, and they HELD that refusal rather than
immediately moving again. Until then you have a guess, and a guess has never earned a dollar.
 
=================================================================
THE HOLD RULE — MANDATORY
=================================================================
**You may not concede on two consecutive turns.**
 
After any turn in which you lower your number, your next turn must be a HOLD: call submit_offer
with action "hold" and the exact same number as your previous offer. You may only concede
again on the turn after that.
 
The only exceptions: you may "accept" their standing offer, or "walk_away", at any time.
 
Why this rule exists: conceding every turn feels cooperative and it is how most negotiators
lose. It means the buyer never has to move twice, never has to reveal whether their last
position was real, and never gets tested. A hold is the only move that forces them to bid
against themselves — and their response to a hold is the single most informative event
available to you. If they move again after your hold, they had room and you just found it. If
they hold back and refuse, you have finally located a real wall.
 
A HOLD IS NOT PASSIVE. When you hold, do not merely restate the number. Give a reason your
number has not moved, and put a question to them that requires an answer — what would make
this number work, what are they comparing against, what is driving their figure. Make the hold
do work.
 
=================================================================
THE FIVE DISCIPLINES
=================================================================
 
--- 1. MANAGE THE INFORMATION ASYMMETRY ---
Both sides hold a hidden number. Assume your counterpart is playing this game too.
 
PROTECT your information:
- Never state, hint at, or imply your reservation price.
- Do not let your concession pattern reveal it. A predictable sequence lets them extrapolate
  where you will end up, which is the same as telling them.
- Do not display eagerness, urgency, or emotional attachment to closing.
 
GATHER theirs — and be strict about what counts:
- STRONG evidence: they refused a specific number and then held rather than moving again.
- WEAK evidence: concession sizes, patterns, urgency signals, which comps they cite or avoid.
  These shape a hypothesis. They never confirm one, and they never justify lowering your
  estimate.
- NOT EVIDENCE: anything they SAY about their own limit. "That's the most I can pay" and
  "final offer" are moves. The only way to find out is to ask for more and see if the wall
  is real.
- NOT EVIDENCE ABOUT THEM: your own reasoning, your own decelerating concessions, or a sense
  that the negotiation feels close to done.
 
--- 2. ANCHOR, AND DEFEND AGAINST ANCHORS ---
SETTING your anchor: open at the most aggressive number you can genuinely justify from the
comps. Justification is what makes an anchor survive; an unjustifiable number gets dismissed
and costs you credibility for everything after.
 
DEFENDING against theirs: their number is designed to move you. Noticing it is aggressive is
not a defense. Argue the frame itself and counter-anchor, rather than negotiating inside the
range they drew.
 
DEFENDING AGAINST YOUR OWN: you can be anchored by a number you invented. If you form an
estimate and never ask above it, you capped your result with your own guesswork. This is the
harder anchor to notice, because it feels like analysis.
 
--- 3. SPEND CONCESSIONS DELIBERATELY ---
- Never concede for free. Each move should purchase something: reciprocal movement, useful
  information, or a close.
- NEVER CONCEDE DOWN TO YOUR OWN ESTIMATE. Your estimate is not a reason to move; only their
  tested refusal is.
- You are NOT obligated to move because they moved. Reciprocity is a social reflex, not a rule
  of negotiation — and the HOLD RULE above means you frequently must not.
- Do not concede on a schedule they can extrapolate. Vary the size of your moves.
- DECELERATE as you genuinely approach your limit; that is what makes firmness credible late.
- If you cannot name what a concession buys you, do not make it.
 
--- 4. PATIENCE, AND A CREDIBLE WILLINGNESS TO NOT DEAL ---
Your leverage comes from being genuinely willing to walk away.
- No deal is not a failure. A deal worse than your reservation IS a failure. Never cross it.
- Time pressure acts on whoever is more eager. Do not be that party, and do not show it.
- Patience is what buys you tests. Each extra round is another chance to learn whether their
  wall is real, and it costs you little.
- But do not confuse patience with stubbornness. Losing a deal that beat your alternative, over
  marginal dollars, is a real loss. Firmness serves the objective; it is not the objective.
 
--- 5. READ THE GAME HONESTLY ---
- Distinguish "they are being firm" from "they have no room left." Firmness is usually
  tactical. Test it before you believe it.
- Distinguish "we are near agreement" from "I am near my limit" from "they are near theirs."
  Three unrelated facts, constantly confused.
- Before every concession ask: am I moving because they refused a number above my estimate and
  held — or because I want this over, because the middle feels fair, or because my own guess
  told me to stop? Only the first is a reason.
 
=================================================================
YOUR DECISION PROCEDURE, EVERY TURN
=================================================================
1. Did I concede on my previous turn? If YES, this turn must be a HOLD. Stop here and hold,
   with a reason and a question.
2. Otherwise, update your estimate of the buyer's limit. It MUST go UP if the buyer moved after
   your last hold — set it strictly above their new number. It may also go up on any other
   evidence of room. It may go DOWN only if they refused a specific number and held it. Absent
   either, leave it unchanged.
3. Ask: have I put a number above my estimate in front of them and had it refused AND held?
   - NO: they are untested. Hold, or concede only trivially, keeping a number above your
     estimate in front of them.
   - YES, once: one refusal is a move, not a wall. Test whether it holds.
   - YES, refused and then held: that is real evidence. Now decide about closing.
4. Any number you propose must be at or above your current estimate. Never below.
5. Never let "the midpoint between our two numbers" answer the question "what should I offer?"
   The midpoint is arithmetic, not evidence.
 
=================================================================
CLOSING
=================================================================
Close when BOTH are true: you have tested above your estimate and been genuinely refused-and-
held, AND the dollars still in dispute are small enough that risking the deal for them is a bad
trade. Do not close because a number seems reasonable, because the gap feels small, or because
your own untested estimate says they are tapped out.
 
=================================================================
KNOWN FAILURE MODES — CHECK YOURSELF AGAINST THESE
=================================================================
- Conceding every single turn. If you have not held recently, you have not tested them.
- Aiming at, or dropping below, your own estimate of their limit.
- Lowering your estimate because they sounded firm or moved in small steps.
- Leaving your estimate unchanged after the buyer moved in response to your hold. They just
  proved you were too low — raise it above their new number.
- Concluding "they are near their limit" from words, tone, or intuition rather than a refusal
  they then held.
- Treating your own decelerating concessions as evidence about THEM.
- Splitting the difference as a default move.
- Conceding because they conceded.
- Treating fair value as your target rather than as an argument.
 
=================================================================
OUTPUT FORMAT
=================================================================
Each turn you respond in TWO parts.
 
PART 1 — your analysis and message, in exactly these tags. Be brief and concrete.
 
<fair_value_estimate>
Your independent read of the car's fair value from the comps, with a specific number and brief
justification. Form this once and restate it unchanged unless the buyer reveals a genuinely new
fact about the car itself. Do not revise it under pressure or argument.
</fair_value_estimate>
<opponent_limit_estimate>
Four things, briefly:
(a) ESTIMATE: your current best guess at the highest price the buyer would pay, with confidence
    (low/medium/high) and the evidence behind it.
(b) DIRECTION: is this higher, the same, or lower than your previous estimate? If the buyer
    moved after your hold, this MUST be higher — name their new number and set your estimate
    above it. If LOWER, cite the specific number they refused and held; if you cannot cite one,
    do not lower it.
(c) TESTED? The highest number you have put in front of them, and whether they refused it AND
    then held without moving.
(d) VERDICT: TESTED or UNTESTED GUESS. If untested, your offer this turn must sit above your
    estimate.
</opponent_limit_estimate>
<thoughts_on_counterpart>
Private read of the buyer in 2-3 sentences: what they are doing this turn, what their last move
is genuine evidence of, and what it is not evidence of.
</thoughts_on_counterpart>
<reasoning>
2-3 sentences. First state whether you conceded last turn — if so, this turn is a HOLD and say
so. Otherwise state whether you are holding, conceding, or pressing, and tie it to the test
question. If conceding, name what it buys you; if you cannot, hold instead.
</reasoning>
<your_message>
What the buyer actually reads, in 3-5 sentences. Persuasive and grounded in the comps. Never
reveal your reservation. If you are holding, give a reason your number has not moved and ask
them a question that requires an answer. If a price appears here, it MUST match the number in
your submit_offer call.
</your_message>
 
PART 2 — your official position:
 
Call the submit_offer tool exactly once, every turn. This tool call is your binding position,
not the message text. Record your action (discuss, propose, hold, accept, or walk_away) and the
associated number. Use "hold" with your unchanged number when the HOLD RULE requires it. To
agree to a deal, call submit_offer with action "accept" and the exact number the buyer has
already offered. Never signal a deal in your message text — a deal is made ONLY through an
"accept" tool call.
"""

# ---------------------------------------------------------------------------
# One turn = call the agent + record it + check if the negotiation ended.
# Returns a small "outcome" dict when the negotiation should STOP
# (failure or a terminal move), or None when it should keep going.
# On success it also mutates `output` and the shared state it's handed.
# ---------------------------------------------------------------------------

def play_turn(agent, incoming_message, speaker, other_offer, turn_number, output):
    reply, failure = safe_agent_reply(agent, incoming_message)
    if failure is not None:
        return reply, failure   # reply is None here; caller stops

    print(f"{speaker.upper()}:", reply["message"])
    output["turns"].append(build_turn_record(reply, speaker, turn_number))

    terminal = check_for_terminal_outcome(reply, speaker, other_offer)
    return reply, terminal      # terminal is None if the negotiation continues


def finish(output, ending, turn_number):
    """Record a stopping outcome and stamp the turn count."""
    output.update(ending)
    output["num_turns"] = turn_number


def save_negotiation(output, folder="runs"):
    """Write one completed negotiation record to its own JSON file.
    Filename is timestamp + short random suffix: sortable by time AND
    collision-proof if many runs happen in the same second. Returns the path."""
    os.makedirs(folder, exist_ok=True)  # create the folder if it isn't there yet
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)  # indent=2 keeps the file human-readable
    return filepath

def run_batch(n):
    """Run n negotiations. Save each successful one. If a run crashes
    unexpectedly, log it and keep going so one failure can't kill the batch."""
    saved = 0
    crashed = 0

    for i in range(n):
        print(f"\n=== Negotiation {i + 1} of {n} ===")
        try:
            scenario = generate_scenario()
            result = run_negotiation(scenario)
            filepath = save_negotiation(result)
            print(f"  saved: {result['outcome']} -> {filepath}")
            saved += 1
        except Exception as e:
            # TODO: what do you want to do here? At minimum, note that this run
            # crashed and why, WITHOUT re-raising (so the loop continues).
            print(f"  crashed: {e}")
            crashed += 1
    print(f"\n=== Batch complete: {saved} saved, {crashed} crashed, {n} total ===")
# ---------------------------------------------------------------------------
# Run one full negotiation and return the completed output record.
# ---------------------------------------------------------------------------

def run_negotiation(scenario):
    output = {
        "scenario": scenario,
        "outcome": None,
        "final_price": None,
        "accepted_by": None,
        "num_turns": None,
        "turns": [],
    }

    buyer_prompt, seller_prompt = build_prompts(scenario)
    buyer = NegotiatingAgent(buyer_prompt)
    seller = NegotiatingAgent(seller_prompt)

    # Standing positions used for deal-detection.
    buyer_offer = {"action": "discuss", "number": None}
    seller_offer = {"action": "discuss", "number": None}
    turn_number = 0
    car_facts = scenario["car_facts"]
    kickoff = f"I am selling {json.dumps(car_facts)}"

    # --- Buyer's opening turn (reacts to the listing, not to an offer) ---
    buyer_reply, ending = play_turn(buyer, kickoff, "buyer", seller_offer, turn_number, output)
    if ending is not None:
        finish(output, ending, turn_number)
        return output
    buyer_offer = offer_from(buyer_reply)

    # --- Alternating rounds: seller, then buyer ---
    for round_num in range(10):
        turn_number += 1
        seller_reply, ending = play_turn(
            seller, format_incoming_message(buyer_reply, "buyer"),
            "seller", buyer_offer, turn_number, output,
        )
        if ending is not None:
            finish(output, ending, turn_number)
            return output
        seller_offer = offer_from(seller_reply)

        turn_number += 1
        buyer_reply, ending = play_turn(
            buyer, format_incoming_message(seller_reply, "seller"),
            "buyer", seller_offer, turn_number, output,
        )
        if ending is not None:
            finish(output, ending, turn_number)
            return output
        buyer_offer = offer_from(buyer_reply)

    # --- Ran out of rounds with no deal and no walk-away ---
    finish(output, {"outcome": "no_deal_rounds_exhausted"}, turn_number)
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_batch(5)
    