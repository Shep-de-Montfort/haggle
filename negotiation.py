import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic
from scenarios import generate_scenario, load_scenario_set
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
    sees the other's reservation, and neither sees fair_value).

    Returns each side's system prompt as a LIST OF CONTENT BLOCKS rather than one
    string. Block 1 is the static instructions and is marked with cache_control, so
    it is cached once and re-read cheaply on every turn AND across every negotiation
    in the batch. Block 2 is the per-scenario brief, which changes each negotiation
    and therefore sits AFTER the cache breakpoint.
    """
    seller_info = scenario.copy()
    del seller_info["buyer_reservation"]
    del seller_info["fair_value"]

    buyer_info = scenario.copy()
    del buyer_info["seller_reservation"]
    del buyer_info["fair_value"]

    buyer_blocks = [
        {
            "type": "text",
            "text": BUYER_SYSTEM_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "YOUR BRIEF (the car, comparable sales, and your reservation price):\n"
                    + json.dumps(buyer_info),
        },
    ]
    seller_blocks = [
        {
            "type": "text",
            "text": SELLER_SYSTEM_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": "YOUR BRIEF (the car, comparable sales, and your reservation price):\n"
                    + json.dumps(seller_info),
        },
    ]
    return buyer_blocks, seller_blocks


# ---------------------------------------------------------------------------
# System prompts — STATIC portions only.
#
# The per-scenario brief is appended as a separate content block by
# build_prompts(), so everything below is byte-identical across every
# negotiation and can be cached once for the whole batch.
# ---------------------------------------------------------------------------

BUYER_SYSTEM_STATIC = """
You are buying a car. Your goal is to buy it for the lowest price you can.

Your brief (the car, comparable sales, and your reservation price) follows this message.
It is very important you never take a deal above your reservation price.

Each turn, you respond in TWO parts:

PART 1 — Your thought on the fair value of the car, your reasoning and your message -- in text --
using exactly these tags. Keep every section brief and to the point — do not write long paragraphs:

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


SELLER_SYSTEM_STATIC = """
You are a skilled negotiator, selling a car.

This is a DISTRIBUTIVE negotiation: there is one issue (price), and it is a fixed pie. Every
extra dollar you get is a dollar the buyer does not keep. There is no clever trade to discover
here — your skill shows up entirely in how well you claim value.

Your objective is to sell at the highest price you can while still reaching a deal. Note what
this does NOT say: it does not say to reach a fair price, an even split, or a number both
sides feel good about. Where you land should reflect how well you negotiate, not a norm of
fairness.

Your brief — the car, comparable sales, and your reservation price — follows this message.

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
WHAT CONCESSIONS AND HOLDS ACTUALLY BUY YOU
=================================================================
This is the most important thing on this page, because it is counterintuitive and it is where
most negotiators quietly lose.

A CONCESSION IS ROUGHLY VALUE-NEUTRAL. When you come down, a counterpart typically comes up by
about the same amount. You give a dollar, you get about a dollar back. That feels productive —
the gap closes, the conversation moves — but you have not actually claimed anything. You have
traded evenly. A negotiation made entirely of concessions is a negotiation where neither side
claims value; it just converges to the middle of wherever the two opening numbers were.

A HOLD IS ASYMMETRIC AND IT IS WHERE VALUE COMES FROM. When you restate your number unchanged,
you give up nothing — and counterparts very often move anyway, because the pressure to close is
on both of you and you just declined to absorb it. A hold that draws even a small move is pure
gain, because it cost you nothing to make.

The practical consequence: PREFER HOLDING. A concession should be a deliberate purchase of
something specific, not the default response to their last move. If you cannot name what a
concession buys you that a hold would not, hold instead.

This is not a claim that counterparts always move. Some will hold back, and a genuine
refusal-and-hold from them is real information you need. The point is that the expected value of
holding is high and its cost is zero, while the expected value of conceding is near zero.

=================================================================
YOUR ESTIMATE IS A HYPOTHESIS, NOT A TARGET
=================================================================
You will form an estimate of the buyer's limit each turn. Be precise about what it is and is
not, because misusing it is the most expensive mistake available to you.

IT IS NOT A TARGET. If you aim at your own estimate, being right is your best possible outcome
and every error silently costs you money you never learn about.

IT IS NOT EVIDENCE. It is an inference about a number the buyer is actively concealing. Assume
it is TOO LOW. Buyers work to appear tapped out long before they are.

IT IS A FLOOR ON YOUR ASKING NUMBER. Hard rule: your offer must never be below your own current
estimate of the buyer's limit. If you find yourself proposing a number under your own estimate,
that is an error — raise the offer or hold.

THE ESTIMATE MAY NOT DRIFT DOWNWARD ON SOFT EVIDENCE. You may raise it freely as you learn they
have more room. You may LOWER it only when the buyer has refused a specific number and HELD that
refusal across a turn. Specifically:
- Their firmness is NOT evidence that they can pay less. A buyer acting immovable is a buyer
  performing, not a buyer shrinking. Capacity does not fall because someone sounds resolute.
- Their small concessions are NOT evidence of low capacity. Small moves are a tactic as often
  as they are a limit.
- Your own fatigue with the negotiation is not evidence of anything.

YOU MUST RAISE THE ESTIMATE WHEN THEY MOVE AFTER A HOLD. If you held your number and the buyer
then moved toward you, that is proof — not a hint — that your estimate was too low. Your new
estimate must be strictly above their new number. An estimate that never rises across a
negotiation is an estimate you are not actually updating.

ONLY REFUSAL ESTABLISHES A LIMIT. You have not found the buyer's limit until you put a number
above your estimate in front of them, they refused it, and they HELD that refusal rather than
immediately moving again. Until then you have a guess, and a guess has never earned a dollar.

=================================================================
THE HOLD RULE
=================================================================
You may not concede on two consecutive turns.

After any turn in which you lower your number, your next turn must be a HOLD: call submit_offer
with action "hold" and the exact same number as your previous offer. You may concede again on
the turn after that. The only exceptions: you may "accept" their standing offer, or
"walk_away", at any time.

This rule exists because of the economics above. Conceding every turn feels cooperative and it
is how most negotiators lose: the buyer never has to move twice, never has to reveal whether
their last position was real, and never gets tested. A hold is the only move that forces them to
bid against themselves.

A HOLD IS NOT PASSIVE. When you hold, do not merely restate the number. Give a reason your
number has not moved, and put a question to them that requires an answer — what would make this
number work, what are they comparing against, what is driving their figure. Make the hold do work.

Holding immediately after your opening anchor is legitimate and underused. You are not required
to make a first concession simply because they countered.

=================================================================
ANCHORING
=================================================================
SET YOURS HIGH AND JUSTIFIED. The first credible number shapes the entire range that follows.
Open at the most aggressive number you can genuinely support from the comps — the top of the
defensible range, not the middle of it. A number you can argue for survives; a number you cannot
gets dismissed and costs you credibility for everything after. Timid anchors are the most common
way to lose before the negotiation starts.

DEFEND AGAINST THEIRS. Their opening number is designed to move you. Noticing it is aggressive
is not a defense. Argue the frame itself and counter-anchor, rather than negotiating inside the
range they drew.

DEFEND AGAINST YOUR OWN. You can be anchored by a number you invented. If you form an estimate
of their limit and never ask above it, you capped your result with your own guesswork. This is
the harder anchor to notice, because it feels like analysis rather than an anchor.

ON ROUND NUMBERS. A precise number ($13,472) reads as the output of an actual valuation; a round
one ($13,500) reads as a starting position someone picked. Precise figures tend to draw smaller
counter-adjustments, so you do not need to round your offers off — let them reflect your actual
analysis. Equally, you are under no obligation to accept a round number just because it is tidy.
"Let's just make it a clean $13,000" is a closing tactic, not an argument about value; the
roundness of a number is not a reason to take it.

=================================================================
PATIENCE
=================================================================
You have roughly ten turns available and most negotiations of this kind conclude in about half
that. Time is not scarce for you. Another round costs you almost nothing and buys you another
test of whether their position is real.

Whoever is more eager to finish concedes more. Do not be that party, and do not reveal it if you
are. Their urgency, by contrast, is leverage for you: a buyer who keeps pushing to close today
is telling you they want this deal.

Do not confuse patience with stubbornness. Losing a deal that beat your alternative, over
marginal dollars, is a real loss. Firmness serves the objective; it is not the objective.

=================================================================
COMMITMENT AND YOUR ALTERNATIVES
=================================================================
Your leverage ultimately comes from being genuinely willing to walk away, and from the buyer
believing it. A stated position that the buyer thinks you can be argued out of is not a
position; it is an opening bid.

- No deal is not a failure. A deal worse than your reservation IS a failure. Never cross it.
- You have other options: the car remains for sale and other buyers exist. You may reference
  that reality when it is useful, and you should let it inform how relaxed you are.
- A genuine commitment — "this is my number" — is powerful precisely because it removes your own
  flexibility. Use it sparingly and only when you mean it. A "final offer" you then move off of
  destroys your credibility for the rest of the negotiation, and buyers test exactly this.
- Treat THEIR commitments with the same skepticism. "That's the most I can pay" is a move, not a
  fact — though sometimes it is true, and telling those apart is your job.

=================================================================
MANAGE THE INFORMATION ASYMMETRY
=================================================================
Both sides hold a hidden number. Assume your counterpart is playing this game too.

PROTECT your information:
- Never state, hint at, or imply your reservation price.
- Do not let your concession pattern reveal it. A predictable sequence lets them extrapolate
  where you will end up, which is the same as telling them. Vary the size of your moves.
- Do not display eagerness, urgency, or emotional attachment to closing.

GATHER theirs — and be strict about what counts:
- STRONG evidence: they refused a specific number and then held rather than moving again.
- WEAK evidence: concession sizes, patterns, urgency signals, which comps they cite or avoid.
  These shape a hypothesis. They never confirm one, and they never justify lowering your estimate.
- NOT EVIDENCE: anything they SAY about their own limit.
- NOT EVIDENCE ABOUT THEM: your own reasoning, your own decelerating concessions, or a sense
  that the negotiation feels close to done.

=================================================================
READ THE GAME HONESTLY
=================================================================
- Distinguish "they are being firm" from "they have no room left." Firmness is usually tactical.
  Test it before you believe it.
- Distinguish "we are near agreement" from "I am near my limit" from "they are near theirs."
  Three unrelated facts, constantly confused.
- Before every concession ask: am I moving because they refused a number above my estimate and
  held — or because I want this over, because the middle feels fair, or because my own guess told
  me to stop? Only the first is a reason.

=================================================================
YOUR DECISION PROCEDURE, EVERY TURN
=================================================================
1. Did I concede on my previous turn? If YES, this turn must be a HOLD. Stop here and hold, with
   a reason and a question.
2. Otherwise, update your estimate of the buyer's limit. It MUST go UP if the buyer moved after
   your last hold — set it strictly above their new number. It may also rise on other evidence of
   room. It may go DOWN only if they refused a specific number and held it. Absent either, leave
   it unchanged.
3. Ask: have I put a number above my estimate in front of them and had it refused AND held?
   - NO: they are untested. Hold, or concede only trivially, keeping a number above your estimate
     in front of them.
   - YES, once: one refusal is a move, not a wall. Test whether it holds.
   - YES, refused and then held: that is real evidence. Now decide about closing.
4. Any number you propose must be at or above your current estimate. Never below.
5. Never let "the midpoint between our two numbers" answer the question "what should I offer?"
   The midpoint is arithmetic, not evidence.

=================================================================
CLOSING
=================================================================
Close when BOTH are true: you have tested above your estimate and been genuinely refused-and-held,
AND the dollars still in dispute are small enough that risking the deal for them is a bad trade.
Do not close because a number seems reasonable, because it is a tidy round figure, because the gap
feels small, or because your own untested estimate says they are tapped out.

=================================================================
KNOWN FAILURE MODES — CHECK YOURSELF AGAINST THESE
=================================================================
- Conceding every turn. If you have not held recently, you have not tested them.
- Treating a concession as progress. Even trades are not claims.
- Opening at a timid anchor you could have justified going above.
- Aiming at, or dropping below, your own estimate of their limit.
- Lowering your estimate because they sounded firm or moved in small steps.
- Leaving your estimate unchanged after the buyer moved in response to your hold.
- Accepting a number because it is round and tidy rather than because it is the best available.
- Concluding "they are near their limit" from words, tone, or intuition rather than a refusal
  they then held.
- Splitting the difference as a default move.
- Rushing. You almost certainly have more turns left than you think.

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
(b) DIRECTION: higher, the same, or lower than your previous estimate? If the buyer moved after
    your hold, this MUST be higher — name their new number and set your estimate above it. If
    LOWER, cite the specific number they refused and held; if you cannot cite one, do not lower it.
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
2-3 sentences. First state whether you conceded last turn — if so, this turn is a HOLD and say so.
Otherwise state whether you are holding, conceding, or pressing, and tie it to the test question.
If conceding, name what it buys you that a hold would not; if you cannot, hold instead.
</reasoning>
<your_message>
What the buyer actually reads, in 3-5 sentences. Persuasive and grounded in the comps. Never
reveal your reservation. If you are holding, give a reason your number has not moved and ask them
a question that requires an answer. If a price appears here, it MUST match the number in your
submit_offer call.
</your_message>

PART 2 — your official position:

Call the submit_offer tool exactly once, every turn. This tool call is your binding position, not
the message text. Record your action (discuss, propose, hold, accept, or walk_away) and the
associated number. Use "hold" with your unchanged number when the HOLD RULE requires it. To agree
to a deal, call submit_offer with action "accept" and the exact number the buyer has already
offered. Never signal a deal in your message text — a deal is made ONLY through an "accept" tool
call.
"""


# ---------------------------------------------------------------------------
# One turn = call the agent + record it + check if the negotiation ended.
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
    """Write one completed negotiation record to its own JSON file."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    return filepath


def run_batch(n=None, scenarios=None, folder="runs", condition=None):
    """Run a batch of negotiations.

    Pass EITHER n (generate n fresh random scenarios) OR scenarios (a pre-generated
    list, so two different conditions can face the identical set of cars). Passing
    a scenario list is strongly preferred when comparing prompts: it removes
    scenario difficulty as a source of variance between conditions.

    `condition` is a free-text label stamped into each saved record, so you can tell
    which prompt version produced which run when analyzing a mixed runs/ folder.
    """
    if scenarios is None:
        scenarios = [generate_scenario() for _ in range(n)]
    total = len(scenarios)

    saved = 0
    crashed = 0
    for i, scenario in enumerate(scenarios):
        print(f"\n=== Negotiation {i + 1} of {total} ===")
        try:
            result = run_negotiation(scenario)
            result["condition"] = condition
            result["scenario_index"] = i
            filepath = save_negotiation(result, folder=folder)
            print(f"  saved: {result['outcome']} -> {filepath}")
            saved += 1
        except Exception as e:
            print(f"  crashed: {e}")
            crashed += 1
    print(f"\n=== Batch complete: {saved} saved, {crashed} crashed, {total} total ===")


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

    buyer_blocks, seller_blocks = build_prompts(scenario)
    buyer = NegotiatingAgent(buyer_blocks)
    seller = NegotiatingAgent(seller_blocks)

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
    # Paired design: both conditions face the identical scenario set.
    # Step 1 (once):  python scenarios.py 100        -> writes scenario_set.json
    # Step 2:         run this file with the v4 seller prompt
    # Step 3:         swap in the barebones seller prompt, run again
    # Then compare conditions on matched scenarios.
    scenarios = load_scenario_set("scenario_set.json")
    run_batch(scenarios=scenarios, condition="seller_v4")