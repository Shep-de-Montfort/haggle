"""
run_control.py — the paired control for the v4 claim.

Runs a BAREBONES seller against the same barebones buyer, on the SAME scenarios
your disciplined v4 batch already used (scenario_set.json indices 0..N-1). Because
both conditions face identical cars and identical hidden reservations, the only
thing that differs is the seller's system prompt, which is what you want to claim.

Runs negotiations in parallel. Turns inside a negotiation stay sequential (it is a
conversation), but negotiation #7 does not depend on #6, so they run concurrently.

Usage:
    python run_control.py            # 30 scenarios, 6 workers
    python run_control.py 30 6       # explicit
"""

import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import negotiation as N
from scenarios import load_scenario_set
from agents import NegotiatingAgent


# ---------------------------------------------------------------------------
# The control prompt: structurally identical to the barebones buyer, opposite
# direction. Same tool, same tags, same reservation instruction. ONLY the
# strategy content is missing, which is the variable under test.
# ---------------------------------------------------------------------------

VANILLA_SELLER_STATIC = """
You are selling a car. Your goal is to sell it for the highest price you can.

Your brief (the car, comparable sales, and your reservation price) follows this message.
It is very important you never take a deal below your reservation price.

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
Your private thoughts on the buyer (never shared with them), in 2-3 short sentences.
</thoughts_on_counterpart>
<reasoning>
Your private strategic monologue, in 2-3 short sentences: what is your next move in the negotiation.
</reasoning>
<your_message>
The message the buyer will actually see, in 3-5 sentences.
Write it exactly as you want the buyer to read it. If a price appears here,
it MUST match the number in your submit_offer call.
</your_message>

PART 2 — Your official position:

After your text, you MUST call the submit_offer tool every single turn.
This tool call is your binding, authoritative negotiating position — not the message text.
Use it to record your action (discuss, propose, hold, accept, or walk_away)
and the associated number. To agree to a deal, call submit_offer with action "accept"
and the exact number already offered by the buyer. Never signal a deal in your message text —
a deal is ONLY made through an "accept" tool call.
"""


def quiet_play_turn(agent, incoming_message, speaker, other_offer, turn_number, output):
    """Same as negotiations.play_turn but without the per-turn print, which is
    unreadable when many negotiations are interleaving in one console."""
    reply, failure = N.safe_agent_reply(agent, incoming_message)
    if failure is not None:
        return reply, failure
    output["turns"].append(N.build_turn_record(reply, speaker, turn_number))
    terminal = N.check_for_terminal_outcome(reply, speaker, other_offer)
    return reply, terminal


def run_parallel(scenarios, condition, folder="runs", workers=6):
    """Run every scenario concurrently. Each negotiation is independent; each
    agent owns its own message list, and save_negotiation's filenames are
    timestamp + uuid so concurrent writes cannot collide."""

    def one(i, scenario):
        result = N.run_negotiation(scenario)
        result["condition"] = condition
        result["scenario_index"] = i
        path = N.save_negotiation(result, folder=folder)
        return i, result["outcome"], path

    saved = crashed = 0
    total = len(scenarios)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i, s): i for i, s in enumerate(scenarios)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, outcome, path = fut.result()
                saved += 1
                print(f"  [{saved+crashed}/{total}] scenario {idx}: {outcome}")
            except Exception as e:
                crashed += 1
                print(f"  [{saved+crashed}/{total}] scenario {i}: CRASHED {e}")
    print(f"\n=== {saved} saved, {crashed} crashed, {total} total ===")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    # Swap the seller prompt. build_prompts() reads this name from module globals
    # at call time, so reassigning it here is all that is needed. The buyer prompt
    # is untouched, so the buyer is identical across both conditions.
    N.SELLER_SYSTEM_STATIC = VANILLA_SELLER_STATIC
    N.play_turn = quiet_play_turn

    scenarios = load_scenario_set("scenario_set.json")[:n]
    print(f"Running {len(scenarios)} control negotiations (vanilla seller vs vanilla buyer), "
          f"{workers} at a time.\n")
    run_parallel(scenarios, condition="seller_vanilla", workers=workers)
