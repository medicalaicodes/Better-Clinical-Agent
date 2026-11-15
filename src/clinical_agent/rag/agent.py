"""
DxAgent - a diagnostic agent that uses the medical library to CHECK its ideas, not to make them.

How one question flows through it:

    1. FRAME       One LLM call. The model reads the case (never the answer options) and writes
                   down the findings, what the question is really asking, a ranked list of
                   hypotheses, and for each hypothesis a couple of claims that can be checked.

    2. CHECK       The library is searched for every claim of every hypothesis at the same time,
                   so no hypothesis gets an unfair head start. Each claim comes back with short
                   "evidence cards" (see retrieval/evidence_tool.py).

    3. TURNS       The model reads the cards, marks each claim SUPPORTED / CONTRADICTED /
                   NOT_ADDRESSED, and can use four tools:
                       consult_reference   check one more claim
                       rule_out            drop a hypothesis - only with a real card or finding id
                       add_hypothesis      add a diagnosis nobody had thought of
                       request_critique    hand the work-up to the critic

    4. CRITIC      A second call with a fresh context plays the sceptical senior doctor: argues
                   for the best alternative, points at findings nothing explains, proposes missed
                   diagnoses. Its new diagnoses get checked in the library like everyone else's.

    5. ANSWER      The free-text answer is matched to an option. With final="select" the options
                   are shown only now, at the very end, together with a summary of the work-up.

The agent and critic could argue forever, so there are brakes: a fixed turn budget, only new
objections that name a different answer count, and we stop once the answer stays the same
after a critique. Whatever doubts are still open at the end are kept as "uncertainty".
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..llm import LLM, is_placeholder, parse_choice, parse_json
from ..retrieval import Retriever
from ..retrieval.evidence_tool import EvidenceTool
from . import agent_prompts as P
from . import prompts
from .methods import MCQ

CLAIM_STATUSES = ("SUPPORTED", "CONTRADICTED", "NOT_ADDRESSED")


# ----------------------------------------------------------------------------------------
# Small helpers for reading what the model wrote
# ----------------------------------------------------------------------------------------

def text_from(value) -> str:
    """A piece of text the model wrote, or "" if it is missing or just the template echoed back
    (8B models sometimes copy "<name>" from the example instead of filling it in)."""
    if not isinstance(value, str):
        return ""
    if is_placeholder(value):
        return ""
    return value.strip()


def id_from(value) -> str:
    """An id like "e3 " or "(F4)" turned into "E3" / "F4".
    This can't go through text_from(): an id has no real words, so it would look like a placeholder."""
    if not isinstance(value, str):
        return ""
    return value.strip().strip(".()[]").upper()


def list_from(parsed: dict, key: str) -> list:
    """parsed[key] if it is a list, otherwise an empty list - the model sometimes writes null."""
    value = parsed.get(key)
    if isinstance(value, list):
        return value
    return []


def question_sentence(question: str) -> str:
    """Just the question itself, e.g. "Which of the following is the most likely diagnosis?".
    The answer-matching step gets this line, but never the patient's story."""
    sentences = re.split(r"(?<=[.!?])\s+", question.strip())
    for sentence in reversed(sentences):
        if "?" in sentence:
            return sentence.strip()
    if sentences:
        return sentences[-1].strip()
    return ""


# ----------------------------------------------------------------------------------------
# The work-up: everything the agent has figured out so far about one case
# ----------------------------------------------------------------------------------------

@dataclass
class WorkUp:
    findings: List[str] = field(default_factory=list)       # "F1", "F2", ... in that order
    asked: str = ""                                          # what the question wants, in words
    hypotheses: Dict[str, dict] = field(default_factory=dict)   # name -> details, see add_hypothesis
    claims: Dict[str, dict] = field(default_factory=dict)       # "C1" -> claim, see add_claim
    cards: Dict[str, dict] = field(default_factory=dict)        # "E1" -> evidence card
    issues: Dict[str, dict] = field(default_factory=dict)       # "I1" -> one critic objection
    leading: Dict[str, str] = field(default_factory=dict)       # current best {"hypothesis", "answer"}
    turns: int = 0
    critiques: int = 0
    notes_for_agent: List[str] = field(default_factory=list)    # shown once on the next turn
    events: List[dict] = field(default_factory=list)            # a diary of what happened
    parse_failures: int = 0
    raw_failures: List[str] = field(default_factory=list)       # unreadable replies, kept to debug

    def log(self, event: str, **details) -> None:
        entry = {"turn": self.turns, "event": event}
        entry.update(details)
        self.events.append(entry)

    def live_hypotheses(self) -> List[str]:
        names = []
        for name, hypothesis in self.hypotheses.items():
            if hypothesis["status"] == "live":
                names.append(name)
        return names

    def open_issue_ids(self) -> List[str]:
        ids = []
        for issue_id, issue in self.issues.items():
            if not issue["resolved"]:
                ids.append(issue_id)
        return ids

    def claims_waiting_for_status(self) -> List[str]:
        """Claims that have their evidence but haven't been judged yet."""
        ids = []
        for claim_id, claim in self.claims.items():
            if claim["cards"] is not None and claim["status"] is None:
                ids.append(claim_id)
        return ids


class DxAgent:
    name = "dx_agent"

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None,
                 k: int = 4, claims: int = 2, max_rounds: int = 4, max_critiques: int = 2,
                 max_new_hypotheses: int = 2, library: bool = True, critic: bool = True,
                 article_first: bool = True, cards_per_claim: int = 2, min_score: float = 0.0,
                 final: str = "map", max_tokens: int = 1500):
        self.llm = llm
        self.k = k                                   # how many hypotheses in the first list
        self.claims = claims                         # claims per hypothesis, per turn
        self.max_rounds = max_rounds                 # agent turns after FRAME; 0 = frame + answer only
        self.max_new_hypotheses = max_new_hypotheses
        self.library = library                       # False = same loop, but the library stays shut
        self.max_tokens = max_tokens

        if critic:
            self.max_critiques = max_critiques
        else:
            self.max_critiques = 0

        # "map":    strictly no options - a separate call that never sees the case matches the
        #           free-text answer to an option.
        # "select": options stay hidden during the whole work-up and are shown only for the
        #           final pick. On MedQA the strict rule costs ~30 points (see IMPLEMENTATION_LOG).
        if final not in ("map", "select"):
            raise ValueError("final must be 'map' or 'select'")
        self.final = final

        self.tool = None
        if library and max_rounds > 0:
            if retriever is None:
                raise ValueError("dx_agent needs the retriever unless library=false or max_rounds=0.")
            self.tool = EvidenceTool(retriever, cards_per_claim=cards_per_claim, min_score=min_score,
                                     article_first=article_first)

    # ====================================================================================
    # One whole question, start to finish
    # ====================================================================================
    def __call__(self, q: MCQ) -> dict:
        workup = self.frame(q)
        first_guess = dict(workup.leading)          # kept so we can tell later if the loop helped
        shuffler = random.Random(q.id)              # hypothesis order is shuffled, but repeatably

        if self.max_rounds > 0:
            self.look_up_new_claims(workup)
            self.run_turns(q, workup, shuffler)

        # Both the first guess and the final answer get matched to options, so the trace
        # always shows how the strictly options-free version did.
        choices = self.match_to_options(q, workup, [first_guess["answer"], workup.leading["answer"]])
        first_choice = choices[0]
        open_choice = choices[1]

        if self.final == "select":
            picked = self.pick_option(q, workup)
            if picked is not None:
                return {"answer": picked, "trace": self.trace(workup, first_guess, first_choice, open_choice, False)}

        if open_choice is not None:
            return {"answer": open_choice, "trace": self.trace(workup, first_guess, first_choice, open_choice, False)}

        # Nothing matched the free-text answer. Answer the ordinary way and flag it, so we can
        # count how often the options-free route had to give up.
        reply = self.llm.chat(prompts.COT_USER.format(question=q.question, options=q.options_text()),
                              system=prompts.COT_SYSTEM, temperature=0.0)
        answer = parse_choice(reply, q.options)
        return {"answer": answer, "trace": self.trace(workup, first_guess, first_choice, open_choice, True)}

    # ====================================================================================
    # Talking to the model
    # ====================================================================================
    def ask(self, workup: WorkUp, prompt: str, system: str, max_tokens: Optional[int] = None) -> dict:
        """Sends one prompt that must come back as a JSON object.

        json_mode makes vLLM physically unable to write broken JSON. Before we had it, about
        half the replies in the first dev run were unreadable and the loop never really ran.
        If something still fails, the start of the reply is kept so we can see why.
        """
        reply = self.llm.chat(prompt, system=system, temperature=0.0,
                              max_tokens=max_tokens or self.max_tokens, json_mode=True)
        parsed = parse_json(reply)
        if parsed:
            return parsed

        workup.parse_failures += 1
        if len(workup.raw_failures) < 3:
            workup.raw_failures.append(str(reply)[:1500])
        return {}

    # ====================================================================================
    # Step 1 - FRAME: findings, what is asked, hypotheses, first guess
    # ====================================================================================
    def frame(self, q: MCQ) -> WorkUp:
        workup = WorkUp()
        prompt = P.FRAME_USER.format(question=q.question, k=self.k, claims=self.claims)
        parsed = self.ask(workup, prompt, P.AGENT_SYSTEM)

        for finding in list_from(parsed, "findings"):
            if text_from(finding):
                workup.findings.append(text_from(finding))

        workup.asked = text_from(parsed.get("asked"))
        if not workup.asked:
            workup.asked = question_sentence(q.question)

        for item in list_from(parsed, "hypotheses")[:self.k]:
            if isinstance(item, dict):
                self.add_hypothesis(workup, item, added_by="frame")

        # The first guess. If the model forgot to name one, its top hypothesis will do.
        leading = parsed.get("leading")
        if not isinstance(leading, dict):
            leading = {}
        hypothesis = text_from(leading.get("hypothesis"))
        answer = text_from(leading.get("answer")) or hypothesis
        if not answer and workup.hypotheses:
            hypothesis = next(iter(workup.hypotheses))
            answer = hypothesis
        workup.leading = {"hypothesis": hypothesis, "answer": answer}
        return workup

    # ====================================================================================
    # Keeping the work-up tidy
    # ====================================================================================
    def find_hypothesis(self, workup: WorkUp, name) -> Optional[str]:
        """The model rarely spells a name the same way twice ("TTP" vs "Thrombotic ... (TTP)"),
        so an exact match is tried first, then one name contained in the other."""
        name = text_from(name).lower()
        if not name:
            return None
        for known in workup.hypotheses:
            if known.lower() == name:
                return known
        for known in workup.hypotheses:
            if name in known.lower() or known.lower() in name:
                return known
        return None

    def add_claim(self, workup: WorkUp, hypothesis: str, text, about) -> bool:
        """Adds a claim to check. Returns False if it was empty or already there."""
        text = text_from(text)
        if not text:
            return False

        # The model likes to repeat itself; the same claim twice just wastes a lookup.
        for claim_id in workup.hypotheses[hypothesis]["claims"]:
            if workup.claims[claim_id]["text"].lower() == text.lower():
                return False

        claim_id = f"C{len(workup.claims) + 1}"
        workup.claims[claim_id] = {
            "hypothesis": hypothesis,
            "text": text,
            "about": text_from(about) or "other",     # presentation / tests / mechanism / treatment
            "cards": None,                            # None = not looked up yet, [] = nothing found
            "status": None,                           # filled in by the agent after reading cards
            "card": None,                             # which card the status rests on
        }
        workup.hypotheses[hypothesis]["claims"].append(claim_id)
        return True

    def add_hypothesis(self, workup: WorkUp, item: dict, added_by: str) -> Optional[str]:
        name = text_from(item.get("name"))
        if not name:
            return None

        known = self.find_hypothesis(workup, name)
        if known:
            # Already on the list. If the critic brings back one we ruled out, it gets another chance.
            if added_by == "critic" and workup.hypotheses[known]["status"] == "ruled_out":
                workup.hypotheses[known]["status"] = "live"
            return known

        workup.hypotheses[name] = {
            "why": text_from(item.get("why")),
            "worst_fit": text_from(item.get("worst_fit")),
            "status": "live",
            "origin": added_by,                       # frame / agent / critic
            "claims": [],
            "rule_out": None,
        }
        for claim in list_from(item, "claims")[:self.claims]:
            if isinstance(claim, dict):
                self.add_claim(workup, name, claim.get("text"), claim.get("about"))
        workup.log("add_hypothesis", by=added_by, name=name)
        return name

    def look_up_new_claims(self, workup: WorkUp) -> None:
        """Sends every claim that hasn't been looked up yet to the library, all in one go."""
        waiting = []
        for claim_id, claim in workup.claims.items():
            if claim["cards"] is None:
                waiting.append(claim_id)
        if not waiting:
            return

        if self.tool is None:
            # Library switched off (an ablation): the model has to judge from memory.
            for claim_id in waiting:
                workup.claims[claim_id]["cards"] = []
            return

        requests = []
        for claim_id in waiting:
            claim = workup.claims[claim_id]
            requests.append({"hypothesis": claim["hypothesis"], "claim": claim["text"], "about": claim["about"]})

        results = self.tool.consult(requests)

        for claim_id, cards in zip(waiting, results):
            card_ids = []
            for card in cards:
                card_id = f"E{len(workup.cards) + 1}"
                workup.cards[card_id] = card
                card_ids.append(card_id)
            workup.claims[claim_id]["cards"] = card_ids

    # ====================================================================================
    # Writing the work-up out as text for the prompts
    # ====================================================================================
    def findings_text(self, workup: WorkUp) -> str:
        if not workup.findings:
            return "(none extracted; read the case)"
        lines = []
        for number, finding in enumerate(workup.findings, start=1):
            lines.append(f"F{number}. {finding}")
        return "\n".join(lines)

    def dossier_text(self, workup: WorkUp, shuffler: random.Random) -> str:
        """Every hypothesis with its claims and cards. The order is shuffled each time, so the
        favourite isn't always sitting at the top where the model reads it first."""
        names = list(workup.hypotheses)
        shuffler.shuffle(names)

        blocks = []
        for name in names:
            hypothesis = workup.hypotheses[name]
            if hypothesis["status"] == "ruled_out":
                status = "RULED OUT"
            else:
                status = "live"

            lines = [f"## {name} [{status}] - explains: {hypothesis['why'] or '?'}"]
            if hypothesis["worst_fit"]:
                lines.append(f"   fits worst: {hypothesis['worst_fit']}")

            for claim_id in hypothesis["claims"]:
                claim = workup.claims[claim_id]

                # e.g.  C3: Haemolysis causes anaemia -> CONTRADICTED (E5)
                line = f"   {claim_id}: {claim['text']}"
                if claim["status"]:
                    line += f" -> {claim['status']}"
                    if claim["card"]:
                        line += f" ({claim['card']})"
                lines.append(line)

                if not self.library:
                    lines.append("      (library switched off: use your own knowledge)")
                    continue
                if not claim["cards"]:
                    lines.append("      NOT FOUND in the library")
                    continue
                for card_id in claim["cards"]:
                    card = workup.cards[card_id]
                    single = " [single source]" if card.get("single_source") else ""
                    lines.append(f"      {card_id} {card['source']}{single}: \"{card['text']}\"")

            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def open_issues_text(self, workup: WorkUp) -> str:
        if not workup.open_issue_ids():
            return ""
        lines = ["", "Open issues from the critic:"]
        for issue_id in workup.open_issue_ids():
            issue = workup.issues[issue_id]
            lines.append(f"{issue_id} ({issue['type']}): {issue['detail']} - to settle it: {issue['action']}")
        return "\n".join(lines) + "\n"

    def notes_text(self, workup: WorkUp) -> str:
        """Messages from the harness, like "rule-out rejected". Each is shown once, then cleared."""
        if not workup.notes_for_agent:
            return ""
        lines = ["", "Notes from the system:"]
        for note in workup.notes_for_agent:
            lines.append(f"- {note}")
        workup.notes_for_agent = []
        return "\n".join(lines) + "\n"

    # ====================================================================================
    # Steps 3 + 4 - the turns, with the critic in between
    # ====================================================================================
    def run_turns(self, q: MCQ, workup: WorkUp, shuffler: random.Random) -> None:
        answers_after_each_critique = []

        while workup.turns < self.max_rounds:
            wants_critique, asked_for_more = self.agent_turn(q, workup, shuffler)
            self.look_up_new_claims(workup)

            out_of_turns = workup.turns >= self.max_rounds
            if not wants_critique and not out_of_turns:
                continue

            if wants_critique and asked_for_more and not out_of_turns:
                # It asked for new evidence and for the critic at once. Let it read the evidence
                # first - otherwise the critic would review a half-finished work-up.
                workup.notes_for_agent.append("Your critique request was deferred: first read the new evidence.")
                continue

            if workup.critiques >= self.max_critiques:
                break

            blocking = self.critique(q, workup, shuffler)
            answers_after_each_critique.append(workup.leading["answer"].lower())

            if not blocking:
                break       # nothing serious left to object to
            if len(answers_after_each_critique) >= 2:
                if answers_after_each_critique[-1] == answers_after_each_critique[-2]:
                    break   # same answer after two reviews - arguing more won't change it
            if workup.turns >= self.max_rounds:
                break       # no turn left to respond; the objection stays on record as a doubt

            self.look_up_new_claims(workup)     # the critic's new diagnoses need evidence too

    def agent_turn(self, q: MCQ, workup: WorkUp, shuffler: random.Random):
        """One turn of the agent. Returns (wants the critic?, asked for new evidence?)."""
        workup.turns += 1
        has_open_issues = bool(workup.open_issue_ids())

        waiting = workup.claims_waiting_for_status()
        if waiting:
            to_judge = ", ".join(waiting)
        else:
            to_judge = "(none)"

        if self.library:
            status_rule = P.STATUS_RULE_LIBRARY
        else:
            status_rule = P.STATUS_RULE_NO_LIBRARY

        prompt = P.ROUND_USER.format(
            question=q.question,
            asked=workup.asked,
            findings=self.findings_text(workup),
            dossier=self.dossier_text(workup, shuffler),
            to_judge=to_judge,
            issues=self.open_issues_text(workup),
            messages=self.notes_text(workup),
            turns_left=self.max_rounds - workup.turns + 1,
            claims=self.claims,
            status_rule=status_rule,
            respond_rule=P.RESPOND_RULE if has_open_issues else "",
        )
        parsed = self.ask(workup, prompt, P.AGENT_SYSTEM)
        if not parsed:
            workup.notes_for_agent.append("Your last reply was not valid JSON. Reply with the JSON object only.")
            return False, False

        self.read_claim_statuses(workup, parsed, waiting)
        self.read_replies_to_critic(workup, parsed)
        wants_critique, new_claims, new_hypotheses = self.run_tools(workup, parsed)

        # A turn where it asks for nothing new means it's done checking - treat that as
        # "send it to the critic" rather than burning the next turn on the same thing.
        if new_claims == 0 and new_hypotheses == 0 and not wants_critique:
            wants_critique = True

        leading = parsed.get("leading")
        if isinstance(leading, dict) and text_from(leading.get("answer")):
            workup.leading = {"hypothesis": text_from(leading.get("hypothesis")),
                              "answer": text_from(leading.get("answer"))}

        workup.log("turn", critique=wants_critique, new_claims=new_claims, leading=workup.leading["answer"])
        asked_for_more = new_claims > 0 or new_hypotheses > 0
        return wants_critique, asked_for_more

    def read_claim_statuses(self, workup: WorkUp, parsed: dict, waiting: List[str]) -> None:
        for item in list_from(parsed, "claim_status"):
            if not isinstance(item, dict):
                continue
            claim_id = id_from(item.get("claim"))
            status = id_from(item.get("status"))
            # Only claims we actually asked about count. In the dev run the model once kept
            # inventing ids (C9, C10, ... C24) until it hit the length limit.
            if claim_id not in waiting or status not in CLAIM_STATUSES:
                continue
            workup.claims[claim_id]["status"] = status
            card = id_from(item.get("card"))
            if card in workup.cards:
                workup.claims[claim_id]["card"] = card

    def read_replies_to_critic(self, workup: WorkUp, parsed: dict) -> None:
        # Answering an objection closes it for now. The critic can reopen it next time,
        # but it sees the reply, so it has to engage with it rather than just repeat itself.
        for item in list_from(parsed, "responses"):
            if not isinstance(item, dict):
                continue
            issue_id = id_from(item.get("issue"))
            reply = text_from(item.get("reply"))
            if issue_id in workup.issues and reply:
                workup.issues[issue_id]["reply"] = reply
                workup.issues[issue_id]["resolved"] = True

    def run_tools(self, workup: WorkUp, parsed: dict):
        """Carries out the tool calls of one turn. Returns (critique wanted?, #claims, #hypotheses)."""
        wants_critique = False
        new_claims_per_hypothesis = {}
        new_hypotheses = 0

        for action in list_from(parsed, "actions"):
            if not isinstance(action, dict):
                continue
            tool = text_from(action.get("tool")).lower()

            if tool in ("request_critique", "finish"):
                wants_critique = True

            elif tool == "consult_reference":
                name = self.find_hypothesis(workup, action.get("hypothesis"))
                if name is None:
                    continue
                # Same cap for every hypothesis, so the favourite can't hog the library.
                if new_claims_per_hypothesis.get(name, 0) >= self.claims:
                    continue
                if self.add_claim(workup, name, action.get("claim"), action.get("about")):
                    new_claims_per_hypothesis[name] = new_claims_per_hypothesis.get(name, 0) + 1

            elif tool == "add_hypothesis":
                if new_hypotheses < self.max_new_hypotheses:
                    if self.add_hypothesis(workup, action, added_by="agent"):
                        new_hypotheses += 1

            elif tool == "rule_out":
                self.rule_out(workup, action)

        return wants_critique, sum(new_claims_per_hypothesis.values()), new_hypotheses

    def rule_out(self, workup: WorkUp, action: dict) -> None:
        """Drops a hypothesis - but only if the model points at something real: a card the
        library actually returned (E7) or a finding from the case (F3). "I just think so" is refused."""
        name = self.find_hypothesis(workup, action.get("hypothesis"))
        evidence = id_from(action.get("evidence"))

        cites_a_card = evidence in workup.cards
        cites_a_finding = False
        if re.fullmatch(r"F\d+", evidence):
            number = int(evidence[1:])
            cites_a_finding = 1 <= number <= len(workup.findings)

        problem = None
        if name is None:
            problem = "unknown hypothesis"
        elif workup.hypotheses[name]["status"] == "ruled_out":
            return
        elif not cites_a_card and not cites_a_finding:
            problem = f"'{evidence or 'nothing'}' is not a card or finding id"
        elif len(workup.live_hypotheses()) <= 1:
            problem = "it is the last live hypothesis (add another one first)"

        if problem:
            workup.notes_for_agent.append(f"Rule-out of '{action.get('hypothesis')}' rejected: {problem}.")
            workup.log("rule_out_rejected", why=problem)
            return

        workup.hypotheses[name]["status"] = "ruled_out"
        workup.hypotheses[name]["rule_out"] = {"evidence": evidence, "reason": text_from(action.get("reason"))}
        workup.log("rule_out", name=name, evidence=evidence)

    # ====================================================================================
    # The critic
    # ====================================================================================
    def critique(self, q: MCQ, workup: WorkUp, shuffler: random.Random) -> List[str]:
        """Runs the critic. Returns the ids of new objections serious enough to keep going."""
        workup.critiques += 1

        rule_outs = []
        for name, hypothesis in workup.hypotheses.items():
            if hypothesis["status"] == "ruled_out" and hypothesis["rule_out"]:
                reason = hypothesis["rule_out"]["reason"]
                cited = hypothesis["rule_out"]["evidence"]
                rule_outs.append(f"- {name}: {reason} (cites {cited})")

        # It sees its own earlier objections next to the agent's replies, so it can accept
        # a good answer instead of raising the same point again.
        previous = []
        for issue_id, issue in workup.issues.items():
            reply = issue.get("reply") or "(no reply)"
            previous.append(f"{issue_id}: {issue['detail']} -> colleague replied: {reply}")

        prompt = P.CRITIC_USER.format(
            question=q.question,
            asked=workup.asked,
            findings=self.findings_text(workup),
            dossier=self.dossier_text(workup, shuffler),
            rule_outs="\n".join(rule_outs) or "(none)",
            answer=workup.leading["answer"],
            hypothesis=workup.leading["hypothesis"] or "?",
            previous="\n".join(previous) or "(none)",
        )
        parsed = self.ask(workup, prompt, P.CRITIC_SYSTEM)
        if not parsed:
            return []

        # Earlier objections the critic still isn't happy with get reopened.
        for item in list_from(parsed, "previous_issues"):
            if isinstance(item, dict):
                issue_id = id_from(item.get("issue"))
                if issue_id in workup.issues and item.get("resolved") is False:
                    workup.issues[issue_id]["resolved"] = False

        blocking = self.record_new_issues(workup, parsed)

        # Missed diagnoses join the list and get checked like any other - no free pass.
        added = 0
        for item in list_from(parsed, "new_hypotheses"):
            if isinstance(item, dict) and added < self.max_new_hypotheses:
                if self.add_hypothesis(workup, item, added_by="critic"):
                    added += 1

        # Rule-outs the critic thinks were too hasty come back.
        for name in list_from(parsed, "restore"):
            known = self.find_hypothesis(workup, name)
            if known and workup.hypotheses[known]["status"] == "ruled_out":
                workup.hypotheses[known]["status"] = "live"
                workup.log("restored", name=known)

        strongest = parsed.get("strongest_alternative")
        if not isinstance(strongest, dict):
            strongest = {}
        workup.log("critique", blocking=blocking, new_hypotheses=added,
                   alternative=text_from(strongest.get("hypothesis")))
        return blocking

    def record_new_issues(self, workup: WorkUp, parsed: dict) -> List[str]:
        already_raised = set()
        for issue in workup.issues.values():
            already_raised.add(issue["detail"].lower())

        blocking = []
        for item in list_from(parsed, "issues"):
            if not isinstance(item, dict):
                continue
            detail = text_from(item.get("detail"))
            if not detail or detail.lower() in already_raised:
                continue

            # An objection only holds things up if the critic says what the better answer is,
            # and it's actually different. In the first dev run it flagged 296 of 298 reviews
            # as "could change the answer", usually without naming any alternative.
            alternative = text_from(item.get("alternative_answer"))
            different = alternative.lower() != workup.leading["answer"].lower()
            serious = item.get("changes_answer") is True and bool(alternative) and different

            issue_id = f"I{len(workup.issues) + 1}"
            workup.issues[issue_id] = {
                "type": text_from(item.get("type")) or "other",
                "detail": detail,
                "action": text_from(item.get("action")),
                "alternative_answer": alternative,
                "changes_answer": serious,
                "resolved": not serious,
                "critique": workup.critiques,
            }
            if serious:
                blocking.append(issue_id)
        return blocking

    # ====================================================================================
    # Step 5 - turning the answer into an option letter
    # ====================================================================================
    def match_to_options(self, q: MCQ, workup: WorkUp, answers: List[str]) -> List[Optional[str]]:
        """Which option means the same as each free-text answer. This call sees the options and
        the question sentence, but NOT the case - so it can't quietly solve the question itself."""
        unique = []
        for answer in answers:
            if answer and answer not in unique:
                unique.append(answer)
        if not unique:
            return [None] * len(answers)

        numbered = []
        for number, answer in enumerate(unique, start=1):
            numbered.append(f"{number}. {answer}")

        prompt = P.MAP_USER.format(asked=workup.asked, stem=question_sentence(q.question),
                                   options=q.options_text(), answers="\n".join(numbered))
        parsed = self.ask(workup, prompt, P.MAP_SYSTEM, max_tokens=200)

        letter_for = {}
        for item in list_from(parsed, "matches"):
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("answer"))
            except (TypeError, ValueError):
                continue
            letter = str(item.get("choice", "")).strip().upper().strip("().")
            if 1 <= number <= len(unique) and letter in q.options:
                letter_for[unique[number - 1]] = letter

        results = []
        for answer in answers:
            results.append(letter_for.get(answer))
        return results

    def pick_option(self, q: MCQ, workup: WorkUp) -> Optional[str]:
        """final="select": the options are revealed only here, after the whole work-up."""
        summary = []
        for name, hypothesis in workup.hypotheses.items():
            supported = 0
            contradicted = 0
            for claim_id in hypothesis["claims"]:
                if workup.claims[claim_id]["status"] == "SUPPORTED":
                    supported += 1
                elif workup.claims[claim_id]["status"] == "CONTRADICTED":
                    contradicted += 1
            if hypothesis["status"] == "ruled_out":
                status = "ruled out"
            else:
                status = "live"
            summary.append(f"- {name} [{status}]: {supported} claims supported, {contradicted} contradicted")

        doubts = []
        for issue_id in workup.open_issue_ids():
            doubts.append(f"- {workup.issues[issue_id]['detail']}")

        prompt = P.SELECT_USER.format(question=q.question, asked=workup.asked,
                                      hypothesis=workup.leading["hypothesis"] or "?",
                                      answer=workup.leading["answer"] or "?",
                                      workup="\n".join(summary) or "(none)",
                                      issues="\n".join(doubts) or "(none)",
                                      options=q.options_text())
        parsed = self.ask(workup, prompt, P.SELECT_SYSTEM, max_tokens=600)
        if not parsed:
            return None
        return parse_choice(json.dumps(parsed), q.options)

    # ====================================================================================
    # What gets saved for every question
    # ====================================================================================
    def trace(self, workup: WorkUp, first_guess: dict, first_choice: Optional[str],
              open_choice: Optional[str], used_fallback: bool) -> dict:
        # Claims written out in full under each hypothesis, so the trace reads on its own.
        hypotheses = {}
        for name, hypothesis in workup.hypotheses.items():
            record = dict(hypothesis)
            record["claims"] = []
            for claim_id in hypothesis["claims"]:
                record["claims"].append(workup.claims[claim_id])
            hypotheses[name] = record

        open_issues = []
        for issue_id in workup.open_issue_ids():
            open_issues.append(workup.issues[issue_id])

        first = dict(first_guess)
        first["choice"] = first_choice              # where the very first guess would have landed

        return {
            "findings": workup.findings,
            "asked": workup.asked,
            "first": first,
            "final": workup.leading,
            "open_choice": open_choice,             # the strictly options-free answer
            "final_step": self.final,
            "mapping_fallback": used_fallback,
            "turns": workup.turns,
            "critiques": workup.critiques,
            "open_issues": open_issues,
            "hypotheses": hypotheses,
            "cards": workup.cards,
            "issues": workup.issues,
            "events": workup.events,
            "parse_failures": workup.parse_failures,
            "raw_failures": workup.raw_failures,
        }
