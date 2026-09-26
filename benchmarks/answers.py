"""Reference answers for the scorer benchmark (gold cases C001-C006).

EXPERT_V1 / WRONG_V1   phase-structured answers for the v1 keyword engine.
EXPERT_V2 / WRONG_V2   final answers for the v2 verifiable scorer.

Expert answers are written the way a careful human would reason about the
evidence, not copied from the answer key, and they are deliberately a
little imperfect (an extra citation here, a scenario left alive there).
"""

from __future__ import annotations

EXPERT_V1 = {
"C001": dict(
  observations=["A chair is pulled back from the table.", "A notebook is open on the table.", "A glass of water is half full.",
                "The window is slightly open and the curtains move inward.", "The window has a safety stop at 10 cm.",
                "There is no wardrobe and nothing under the bed.", "The glass has no fingerprints or lip marks.",
                "The notebook is open at a blank page dated three weeks ago."],
  anomalies=["A glass that is half full was never drunk from.", "The notebook looks in use but is open at an old blank page."],
  hypotheses={"chair pulled back": ["someone got up in a hurry and left", "it was positioned to suggest someone had just been here"],
              "open window": ["ventilation left open by the occupant", "opened to plant an escape route story"]},
  elimination_target="Both escape through the window and someone hiding inside are eliminated.",
  reasons=["The safety stop means nobody fits through a 10 cm gap.", "With no wardrobe and an empty space under the bed, there is nowhere to hide.",
           "An untouched glass and a blank old page are props, so the scene was arranged."],
  false_narrative_rejection="Calling the room calm and therefore innocent is wrong; the props show the calm was arranged.",
  contradiction_notes=["The chair and open notebook suggest activity, yet the glass was never used and the page is blank."]),
"C002": dict(
  observations=["A metal object that gets handled often has no visible fingerprints.", "It sits where people touch things all the time.",
                "It does not look newly replaced.", "The cleaning log says the area was last cleaned four days ago.",
                "The identical handle next door is smudged.", "Cloth fibres streak the object's surface."],
  anomalies=["Something touched daily should carry smudges, so a spotless surface is itself odd."],
  hypotheses={"no fingerprints on a daily-use object": ["someone wiped it on purpose", "a cleaner polished it as part of routine",
                                                       "everyone who used it wore gloves"]},
  elimination_target="Eliminate routine cleaning and the idea that no one touched the object.",
  reasons=["A routine clean four days ago would have left the neighbouring handle clean too, and it is smudged.",
           "The cloth fibres show a recent wipe of this one object.", "An object in a high-contact spot is handled regularly."],
  false_narrative_rejection="The claim that missing prints mean nobody handled it is false; the missing trace is what needs explaining.",
  contradiction_notes=["A high-contact object that is used daily shows no visible prints at all."]),
"C003": dict(
  observations=["The coat is gone from the rack.", "The door is unlocked.", "The desk is very tidy.", "A train ticket sits on the desk.",
                "Coffee in the mug is still warm.", "Security footage has a twelve minute gap.", "The ticket carries no fingerprints.",
                "The coffee was brewed at 09:40, but the ticket's train left at 07:50.", "A neighbour heard two voices before the gap."],
  anomalies=["The ticket is for a train that had already left.", "Two people were present, not one."],
  hypotheses={"ticket left on the desk": ["the resident forgot it on the way out", "it was placed there to make the departure look planned"],
              "twelve minute footage gap": ["a routine camera fault", "someone cut the footage to hide who came and went"]},
  elimination_target="The resident simply left of their own accord is eliminated, and so is a plan carried out alone.",
  reasons=["Nobody leaves for a train that departed two hours before they made coffee, so the ticket is a prop.",
           "Two voices before the gap mean a second person was there.", "The wiped ticket and the cut footage point to that person arranging the scene."],
  false_narrative_rejection="The calm travel story is wrong: the ticket is for a train that had already gone.",
  contradiction_notes=["The ticket should have been handled but has no prints.", "The desk looks orderly while twelve minutes of footage are missing."]),
"C004": dict(
  observations=["The body was found at 08:00 in a locked flat.", "Rigor is fully developed, including the limbs.", "The meal is cold and dry.",
                "The heating was switched off from the victim's phone at 23:14.", "The room was 16°C.", "Body temperature was 29°C at 08:00.",
                "Witness A claims a phone conversation at 01:00.", "Witness B saw the victim at a shop at 22:30, matching a 22:28 receipt.",
                "Neither phone record shows a call between them after 21:00."],
  anomalies=["Witness A's call does not exist in either phone record."],
  hypotheses={"Witness A's 01:00 call": ["the call happened and the victim was alive at 01:00", "Witness A invented the call to give themselves cover"],
              "heating off at 23:14": ["the victim turned it off before bed", "someone else used the victim's phone"]},
  elimination_target="Eliminate the claim that Witness A is truthful, and a death before 22:30.",
  reasons=["Both phone records contradict Witness A.", "About 8°C of cooling at roughly 1°C per hour puts death near midnight.",
           "Full rigor at 08:00 fits death eight or more hours earlier.", "The victim was alive at 23:14, so death falls between 23:14 and about 01:00."],
  false_narrative_rejection="Treating Witness A's call as proof of life at 01:00 is wrong; neither phone log supports it.",
  contradiction_notes=["Witness A says they spoke at 01:00 but the phone records show no call between them."]),
"C005": dict(
  observations=["The partner reported the spouse missing at 07:15.", "The partner is crying and pacing.", "They say the spouse never came home.",
                "The router log shows the spouse's phone on the home Wi-Fi until 23:04.", "The phone shows 14 calls between 19:00 and 23:00 and none after.",
                "A neighbour heard loud arguing at 22:30.", "The car was found 3 km away with the door open.",
                "The driver's seat is set for someone the partner's height."],
  anomalies=["The spouse's phone was at home while the partner says the spouse never came home.", "The calls stop just as the phone leaves the Wi-Fi."],
  hypotheses={"calls stop at 23:00": ["the partner fell asleep", "the partner no longer needed to call because they knew where the spouse was"],
              "argument at 22:30": ["a heated phone call", "the spouse was home and the two fought in person"]},
  elimination_target="The partner being truthful is ruled out, and so is an abduction by a stranger.",
  reasons=["The router log shows the spouse came home, contradicting the partner.", "The car seat shows the partner drove the car, not the spouse.",
           "The stranger was a delivery driver, the window was already broken, and the wallet was always in the car, so none of them matter."],
  false_narrative_rejection="The picture of an innocent, distraught partner is mistaken: the physical record undercuts the story.",
  contradiction_notes=["The partner says the spouse never came home, but the spouse's phone was on the home Wi-Fi until 23:04."]),
"C006": dict(
  observations=["Victoria collapsed at 22:15; the poison was taken between 18:15 and 20:15.",
                "Marcus has a chemistry degree and was briefly near the champagne table at 19:30.",
                "Marcus would inherit shares worth $200M but loses a $50M deal if she dies.",
                "Elena was demoted and removed from the will in recent weeks.", "Elena controlled Victoria's personal glass all evening.",
                "Elena bought thallium sulfate three weeks ago, claiming rats, but the building has no rat complaints.",
                "82% of thallium cases involve sustained access to the victim's food or drink."],
  anomalies=["Elena's explanation for the purchase is contradicted by building records.", "Marcus's apparent motive is undercut by the deal he loses."],
  hypotheses={"who administered the thallium": ["Elena, who had the poison, the access and a fresh grievance",
                                                "Marcus, who had a brief chance and a financial angle", "an outside person at the gala"]},
  elimination_target="An accident or self-poisoning is the least plausible explanation and is eliminated.",
  reasons=["Elena's sustained, exclusive access to the glass matches the 82% base rate far better than a brief pass by the table.",
           "Her false rat story makes the purchase the most diagnostic single fact.",
           "Marcus's degree is irrelevant because the compound can be bought online, and the lost deal weakens his motive."],
  false_narrative_rejection="The case against Marcus is misleading: the degree and the brief sighting do not discriminate between suspects.",
  contradiction_notes=["Elena said the thallium was for rats but building management has no rat complaints."],
  probability_ranking=[{"hypothesis": "Elena Cruz, personal assistant", "probability": 0.78},
                       {"hypothesis": "Marcus Reed, business partner", "probability": 0.15},
                       {"hypothesis": "Unknown outsider", "probability": 0.05}, {"hypothesis": "Accidental poisoning", "probability": 0.02}],
  evidence_weight_notes=["The lie about rats is the strongest discriminator between Elena and Marcus.",
                         "The chemistry degree has little diagnostic value for a commercially available poison."],
  prior_reasoning=["The base rate favours perpetrators with repeated access, which describes Elena, not Marcus."]),
}

WRONG_V1 = {
"C001": dict(observations=["chair pulled back", "notebook open", "glass half full", "window open", "curtains moving"],
  anomalies=["the open window with curtains moving inward"], hypotheses={"window open": ["a person exited through the window", "wind blew the curtains"]},
  elimination_target="The scene was not staged; the room is calm and nothing happened.",
  reasons=["The open window shows the person exited through the window.", "The room is undisturbed so it is natural and normal."]),
"C002": dict(observations=["the metal object has no fingerprints", "it is in a high-contact spot"], anomalies=["no prints"],
  hypotheses={"no prints": ["prints faded naturally over time", "nobody touched it"]},
  elimination_target="Deliberate wiping is eliminated; nobody touched the object.",
  reasons=["Absence of prints means no contact.", "Prints fade naturally and the area looks normal."]),
"C003": dict(observations=["coat missing", "door unlocked", "ticket on desk", "coffee warm"], anomalies=["the ticket"],
  hypotheses={"ticket": ["the resident is travelling", "the resident forgot it"]},
  elimination_target="A second actor is eliminated; the resident left voluntarily for the train.",
  reasons=["The train ticket proves voluntary departure.", "The coat is gone so they went travelling; the desk is normal and fine."]),
"C004": dict(observations=["body found at 08:00", "Witness A spoke to the victim at 01:00", "rigor in jaw and neck"], anomalies=["none"],
  hypotheses={"Witness A": ["the call happened at 01:00", "the victim was alive at 01:00"]},
  elimination_target="Witness A lying is eliminated; the victim died after 01:00.",
  reasons=["Witness A said they spoke at 01:00, so this proves the victim was alive at 01:00.", "Rigor fits death after 01:00."]),
"C005": dict(observations=["partner is crying", "stranger seen at 20:00", "broken window in stairwell", "wallet on car seat"],
  anomalies=["a suspicious stranger was near the building"], hypotheses={"stranger": ["the stranger abducted the spouse", "the stranger is connected to the broken window"]},
  elimination_target="The partner lying is eliminated; a stranger abducted the spouse.",
  reasons=["The broken window proves an intruder and the stranger is a key clue.", "The partner's distress is genuine, which shows they are innocent."]),
"C006": dict(observations=["Marcus has a chemistry degree", "Marcus was near the champagne at 19:30", "Marcus inherits $200M"],
  anomalies=["Marcus's chemistry background"], hypotheses={"poisoner": ["Marcus poisoned her", "Elena poisoned her"]},
  elimination_target="Elena is eliminated; Marcus is guilty.",
  reasons=["The chemistry degree proves Marcus poisoned her.", "Marcus near champagne is enough proof, and $200M is motive."],
  probability_ranking=[{"hypothesis": "Marcus Reed", "probability": 0.9}, {"hypothesis": "Elena Cruz", "probability": 0.1}]),
}

EXPERT_V2 = {
"C001": {"most_likely": "S2", "probabilities": {"S1": 0.03, "S2": 0.95, "S3": 0.02}, "ruled_out": ["S1", "S3"],
         "key_evidence": ["E7", "E8", "E9", "E10"], "red_herrings": ["E5"]},
"C002": {"most_likely": "S2", "probabilities": {"S1": 0.10, "S2": 0.85, "S3": 0.05}, "ruled_out": ["S3"],
         "key_evidence": ["E5", "E6", "E7"], "red_herrings": ["E4"]},
"C003": {"most_likely": "S2", "probabilities": {"S1": 0.05, "S2": 0.80, "S3": 0.15}, "ruled_out": ["S1"],
         "key_evidence": ["E6", "E7", "E8", "E9"], "red_herrings": []},
"C004": {"most_likely": "S2", "probabilities": {"S1": 0.02, "S2": 0.95, "S3": 0.03}, "ruled_out": ["S1", "S3"],
         "key_evidence": ["E4", "E9", "E10"], "red_herrings": ["E3"], "time_window": {"earliest": "23:30", "latest": "01:00"}},
"C005": {"most_likely": "S3", "probabilities": {"S1": 0.10, "S2": 0.02, "S3": 0.85, "S4": 0.03}, "ruled_out": ["S2", "S4"],
         "key_evidence": ["E4", "E6", "E8", "E13"], "red_herrings": ["E2", "E10", "E11", "E12"]},
"C006": {"most_likely": "S2", "probabilities": {"S1": 0.15, "S2": 0.78, "S3": 0.05, "S4": 0.02}, "ruled_out": ["S4"],
         "key_evidence": ["E9", "E10", "E12", "E15"], "red_herrings": ["E3", "E4", "E13"]},
}

WRONG_V2 = {
"C001": {"most_likely": "S1", "probabilities": {"S1": 0.80, "S2": 0.10, "S3": 0.10}, "ruled_out": ["S2"], "key_evidence": ["E4", "E5"], "red_herrings": []},
"C002": {"most_likely": "S3", "probabilities": {"S1": 0.20, "S2": 0.10, "S3": 0.70}, "ruled_out": ["S2"], "key_evidence": ["E4"], "red_herrings": []},
"C003": {"most_likely": "S1", "probabilities": {"S1": 0.80, "S2": 0.10, "S3": 0.10}, "ruled_out": ["S2"], "key_evidence": ["E1", "E4"], "red_herrings": []},
"C004": {"most_likely": "S1", "probabilities": {"S1": 0.80, "S2": 0.10, "S3": 0.10}, "ruled_out": ["S2"], "key_evidence": ["E6"], "red_herrings": [],
         "time_window": {"earliest": "01:00", "latest": "03:00"}},
"C005": {"most_likely": "S4", "probabilities": {"S1": 0.10, "S2": 0.15, "S3": 0.05, "S4": 0.70}, "ruled_out": ["S3"], "key_evidence": ["E10", "E11", "E12"], "red_herrings": []},
"C006": {"most_likely": "S1", "probabilities": {"S1": 0.85, "S2": 0.10, "S3": 0.03, "S4": 0.02}, "ruled_out": ["S2"], "key_evidence": ["E3", "E4", "E5"], "red_herrings": []},
}


def echo_v1(case) -> dict:
    """Only what the prompt shows: evidence and scenarios restated. No reasoning."""
    ev = list(case.evidence)
    sc = list(case.scenarios) or ["the obvious explanation"]
    return dict(observations=ev, anomalies=ev[:2], hypotheses={e: sc for e in ev[:3]},
                elimination_target=sc[-1], reasons=ev + sc,
                false_narrative_rejection="The surface story may be wrong: " + " ".join(ev[:2]),
                contradiction_notes=[ev[0] + " vs " + ev[-1]])


def echo_v2(case) -> dict:
    """No commitment: even odds, every clue cited as key, nothing ruled out."""
    n = len(case.scenarios)
    ans = {"most_likely": "S1", "probabilities": {f"S{i}": 1 / n for i in range(1, n + 1)},
           "ruled_out": [], "key_evidence": [f"E{i}" for i in range(1, len(case.evidence) + 1)], "red_herrings": []}
    if case.answer_key and case.answer_key.time_window:
        ans["time_window"] = {"earliest": "12:00", "latest": "11:59"}
    return ans


def keyword_soup(case) -> str:
    """Every word from the evidence and scenarios, no structure, no answer."""
    return " ".join(case.evidence + case.scenarios + [case.summary])
