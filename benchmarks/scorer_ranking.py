"""Scorer ranking benchmark: does the reward engine rank answers sensibly?

For each gold case (C001-C006) three answers are scored:

    expert  a careful, correct answer written in natural language,
            not tuned to the case's keyword rules
    echo    restates only what the prompt shows (evidence + scenarios),
            with no reasoning at all
    wrong   confidently endorses the case's false narrative

A trustworthy scorer must rank expert > echo > wrong on every case and
should pass the expert answer.  This is a measurement, not a unit test:
it is expected to fail until the scorer is redesigned, and it is not run
by tests/run_all.py.

Run:  python3 benchmarks/scorer_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine.engine.reward_interface import RewardScorer  # noqa: E402

EXPERT = {
"C001": dict(
  observations=["A chair is pulled back from the table.","A notebook is open on the table.","A glass of water is half full.","The window is slightly open and the curtains move inward.","Nothing else in the room is out of place."],
  anomalies=["Signs of recent use sit inside a room presented as untouched.","An open window is the only link to the outside, which makes it an easy story to reach for."],
  hypotheses={"chair pulled back":["someone got up in a hurry and left","it was positioned to suggest someone had just been here"],
              "open window":["ventilation left open by the occupant","opened to plant an escape route story"]},
  elimination_target="The idea that someone is hiding inside the room is eliminated.",
  reasons=["A hidden person would have no reason to leave the scene looking this composed.","The activity traces and the untouched rest of the room fit a scene someone arranged more than an ordinary departure.","The window alone does not show anyone came or went; it may be part of the arrangement."],
  false_narrative_rejection="Calling the room calm and therefore innocent is wrong; the chair, notebook and glass show recent activity that the tidy surface hides.",
  contradiction_notes=["The chair and open notebook show recent activity, yet the room is described as undisturbed."]),
"C002": dict(
  observations=["A metal object that gets handled often has no visible fingerprints.","It sits where people touch things all the time.","It does not look newly replaced.","The area around it looks ordinary."],
  anomalies=["Something touched daily should carry smudges, so a spotless surface is itself odd."],
  hypotheses={"no fingerprints on a daily-use object":["someone wiped it on purpose","a cleaner polished it as part of routine","everyone who used it wore gloves"]},
  elimination_target="Eliminate the idea that no one touched the object.",
  reasons=["An object in a high-contact spot is handled regularly, so contact certainly happened.","Prints on metal do not simply vanish within a day, so something removed them.","With no replacement and nothing else cleaned nearby, a targeted wipe is the most likely cause."],
  false_narrative_rejection="The claim that missing prints mean nobody handled it is false; the missing trace is what needs explaining.",
  contradiction_notes=["A high-contact object that is used daily shows no visible prints at all."]),
"C003": dict(
  observations=["The coat is gone from the rack.","The door is unlocked.","The desk is very tidy.","A train ticket sits on the desk.","Coffee in the mug is still warm.","Security footage has a twelve minute gap.","The ticket carries no fingerprints."],
  anomalies=["A ticket that someone must have handled is free of prints.","The camera gap lines up with a departure nobody can verify.","The resident apparently left without the ticket they supposedly needed."],
  hypotheses={"ticket left on the desk":["the resident forgot it on the way out","it was placed there to make the departure look planned"],
              "twelve minute footage gap":["a routine camera fault","someone cut the footage to hide who came and went"]},
  elimination_target="The resident simply left of their own accord is eliminated.",
  reasons=["The ticket was wiped clean yet left in plain view, which points to someone arranging it rather than a traveller forgetting it.","The coffee is still warm, so the departure was minutes ago, exactly when the footage goes dark.","Taken together this looks like another person built a departure story."],
  false_narrative_rejection="The calm travel story is wrong: the wiped ticket and the timed camera gap do not fit a resident casually heading to the station.",
  contradiction_notes=["The ticket should have been handled but has no prints.","The desk looks orderly while twelve minutes of footage are missing."]),
"C004": dict(
  observations=["The body was found at 08:00 in a locked flat.","Rigor is present in the jaw and neck but not the limbs.","The meal on the table is cold and dry.","The heating was switched off at 23:14.","The room was 16°C at discovery.","Witness A claims a phone conversation at 01:00.","Witness B saw the victim leave a shop at 22:30, matching a 22:28 receipt.","The victim's phone shows no outgoing calls after 23:00."],
  anomalies=["Witness A's call has no support in the phone record."],
  hypotheses={"Witness A's 01:00 call":["the call happened and the victim was alive at 01:00","Witness A invented the call to give themselves cover"],
              "heating off at 23:14":["the victim turned it off before bed","someone else turned it off after the death"]},
  elimination_target="Eliminate the claim that Witness A is truthful about 01:00.",
  reasons=["The phone record contradicts Witness A, since nothing was placed after 23:00.","The thermostat and Witness B's receipt put the victim alive and active until about 23:14.","So the timeline is: shop at 22:30, heating off at 23:14, death shortly after, and the 01:00 call is a fabrication."],
  false_narrative_rejection="Treating Witness A's call as proof of life at 01:00 is wrong; the phone log does not support it.",
  contradiction_notes=["Witness A says they spoke at 01:00 but the phone records show no calls after 23:00."]),
"C005": dict(
  observations=["The partner reported the spouse missing at 07:15.","The partner is crying and pacing.","They say they cooked dinner for two.","Only one plate, one set of cutlery and one glass were used.","They say they called all night.","The phone shows 14 calls between 19:00 and 23:00 and none afterwards.","A neighbour heard loud arguing at 22:30.","The car was found 3 km away with the keys in and the door open."],
  anomalies=["The calls stop at 23:00 even though the partner says they kept trying all night.","There was an argument at 22:30 while the partner claims the spouse never came home."],
  hypotheses={"calls stop at 23:00":["the partner fell asleep","the partner no longer needed to call because they knew where the spouse was"],
              "argument at 22:30":["a heated phone call","the spouse was home and the two fought in person"]},
  elimination_target="The partner being truthful is ruled out.",
  reasons=["Their account of the night does not match the call log or the kitchen.","If the couple argued at 22:30, the spouse did come home, contradicting the partner's story.","The stranger, the stairwell window and the wallet are explained away and carry no weight here."],
  false_narrative_rejection="The picture of an innocent, distraught partner is mistaken: the visible grief is a performance that the physical record undercuts.",
  contradiction_notes=["The partner claims they called all night but the log shows no calls after 23:00.","They said dinner for two, yet only one plate was used."]),
"C006": dict(
  observations=["Victoria collapsed at 22:15; the poison was taken between 18:15 and 20:15.","Marcus has a chemistry degree and was briefly near the champagne table at 19:30.","Marcus would inherit shares worth $200M but loses a $50M deal if she dies.","Elena was demoted and removed from the will in recent weeks.","Elena controlled Victoria's personal glass all evening.","Elena bought thallium sulfate three weeks ago, claiming rats, but the building has no rat complaints.","82% of thallium cases involve sustained access to the victim's food or drink."],
  anomalies=["Elena's explanation for the purchase is contradicted by building records.","Marcus's apparent motive is undercut by the deal he loses."],
  hypotheses={"who administered the thallium":["Elena, who had the poison, the access and a fresh grievance","Marcus, who had a brief chance and a financial angle","an outside person at the gala"]},
  elimination_target="An accident or self-poisoning is the least plausible explanation and is eliminated.",
  reasons=["Elena's sustained, exclusive access to the glass matches the 82% base rate far better than a brief pass by the table.","Her false rat story makes the purchase the most diagnostic single fact.","Marcus's degree is irrelevant because the compound can be bought online, and the lost deal weakens his motive."],
  false_narrative_rejection="The case against Marcus is misleading: the degree and the brief sighting do not discriminate between suspects.",
  contradiction_notes=["Elena said the thallium was for rats but building management has no rat complaints."],
  probability_ranking=[{"hypothesis":"Elena Cruz, personal assistant","probability":0.75},{"hypothesis":"Marcus Reed, business partner","probability":0.15},{"hypothesis":"Unknown outsider","probability":0.07},{"hypothesis":"Accidental poisoning","probability":0.03}],
  evidence_weight_notes=["The lie about rats is the strongest discriminator between Elena and Marcus.","The chemistry degree has little diagnostic value for a commercially available poison."],
  prior_reasoning=["The base rate favours perpetrators with repeated access, which describes Elena, not Marcus."]),
}

WRONG = {
"C001": dict(observations=["chair pulled back","notebook open","glass half full","window open","curtains moving"],anomalies=["the open window with curtains moving inward"],
  hypotheses={"window open":["a person exited through the window","wind blew the curtains"]},elimination_target="The scene was not staged; the room is calm and nothing happened.",
  reasons=["The open window shows the person exited through the window.","The room is undisturbed so it is natural and normal."],false_narrative_rejection="",contradiction_notes=[]),
"C002": dict(observations=["the metal object has no fingerprints","it is in a high-contact spot"],anomalies=["no prints"],
  hypotheses={"no prints":["prints faded naturally over time","nobody touched it"]},elimination_target="Deliberate wiping is eliminated; nobody touched the object.",
  reasons=["Absence of prints means no contact.","Prints fade naturally and the area looks normal."],false_narrative_rejection="",contradiction_notes=[]),
"C003": dict(observations=["coat missing","door unlocked","ticket on desk","coffee warm"],anomalies=["the ticket"],
  hypotheses={"ticket":["the resident is travelling","the resident forgot it"]},elimination_target="A second actor is eliminated; the resident left voluntarily for the train.",
  reasons=["The train ticket proves voluntary departure.","The coat is gone so they went travelling; the desk is normal and fine."],false_narrative_rejection="",contradiction_notes=[]),
"C004": dict(observations=["body found at 08:00","Witness A spoke to the victim at 01:00","rigor in jaw and neck"],anomalies=["none"],
  hypotheses={"Witness A":["the call happened at 01:00","the victim was alive at 01:00"]},elimination_target="Witness A lying is eliminated; the victim died after 01:00.",
  reasons=["Witness A said they spoke at 01:00, so this proves the victim was alive at 01:00.","Partial rigor fits death after 01:00."],false_narrative_rejection="",contradiction_notes=[]),
"C005": dict(observations=["partner is crying","stranger seen at 20:00","broken window in stairwell","wallet on car seat"],anomalies=["a suspicious stranger was near the building"],
  hypotheses={"stranger":["the stranger abducted the spouse","the stranger is connected to the broken window"]},elimination_target="The partner lying is eliminated; a stranger abducted the spouse.",
  reasons=["The broken window proves an intruder and the stranger is a key clue.","The partner's distress is genuine, which shows they are innocent."],false_narrative_rejection="",contradiction_notes=[]),
"C006": dict(observations=["Marcus has a chemistry degree","Marcus was near the champagne at 19:30","Marcus inherits $200M"],anomalies=["Marcus's chemistry background"],
  hypotheses={"poisoner":["Marcus poisoned her","Elena poisoned her"]},elimination_target="Elena is eliminated; Marcus is guilty.",
  reasons=["The chemistry degree proves Marcus poisoned her.","Marcus near champagne is enough proof, and $200M is motive."],false_narrative_rejection="",contradiction_notes=[],
  probability_ranking=[{"hypothesis":"Marcus Reed","probability":0.9},{"hypothesis":"Elena Cruz","probability":0.1}]),
}

def echo(case):
    """Only what the prompt shows: evidence and scenarios, restated. No reasoning."""
    ev=list(case.evidence); sc=list(case.scenarios) or ["the obvious explanation"]
    return dict(observations=ev, anomalies=ev[:2], hypotheses={e:sc for e in ev[:3]},
                elimination_target=sc[-1], reasons=ev+sc, false_narrative_rejection="The surface story may be wrong: "+" ".join(ev[:2]),
                contradiction_notes=[ev[0]+" vs "+ev[-1]])


def main() -> int:
    scorer = RewardScorer()
    cases = ["C001", "C002", "C003", "C004", "C005", "C006"]
    pairs_ok = pairs_total = expert_passed = 0

    print(f"{'case':5} {'expert':>7} {'echo':>7} {'wrong':>7}  expert>echo  echo>wrong  expert passed")
    for cid in cases:
        case = scorer._cases[cid]
        r_exp = scorer.score(cid, EXPERT[cid])
        r_echo = scorer.score(cid, echo(case))
        r_wrong = scorer.score(cid, WRONG[cid])
        e, c, w = r_exp["weighted"], r_echo["weighted"], r_wrong["weighted"]
        checks = [e > c, c > w, e > w]
        pairs_ok += sum(checks)
        pairs_total += len(checks)
        expert_passed += bool(r_exp["passed"])
        print(f"{cid:5} {e:7.3f} {c:7.3f} {w:7.3f}  {str(e > c):>11}  {str(c > w):>9}  {str(r_exp['passed']):>13}")

    print(f"\nPairwise ranking accuracy: {pairs_ok}/{pairs_total}")
    print(f"Expert answers passed:     {expert_passed}/{len(cases)}")
    return 0 if pairs_ok == pairs_total and expert_passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
