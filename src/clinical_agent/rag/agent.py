"""
DxAgent: an options-free diagnostic agent that uses the library to VERIFY, not to decide.

    1. FRAME     (1 call)  findings, what is asked, k ranked hypotheses with checkable claims,
                           and a first answer. The answer options are never shown.
    2. ROUNDS    (<= max_rounds calls) all claims of all live hypotheses are checked in the
                           library at once (no hypothesis goes "first"); the agent marks each claim
                           SUPPORTED / CONTRADICTED / NOT_ADDRESSED and may:
                             consult_reference  check a new claim
                             rule_out           drop a hypothesis, ONLY citing a card or a finding
                             add_hypothesis     add a diagnosis that explains unexplained findings
                             request_critique   submit the answer to the critic
    3. CRITIC    (<= max_critiques calls, fresh context) argues for the best alternative, finds
                           unexplained findings, proposes overlooked diagnoses, restores weak
                           rule-outs. Its proposals are checked like any other hypothesis.
    4. MAP       (1 call)  the free-text answers (first and final) are matched to the options by a
                           call that never sees the case, so it cannot re-solve the question.

Loop safety (agent <-> critic can ping-pong forever):
  - one shared turn budget, shown to the agent every turn;
  - only NEW issues that could change the answer force another round;
  - the agent may rebut with evidence; repeated issues do not count;
  - stop when the answer did not change after a critique, or the caps are reached;
  - issues still open at the end are recorded as uncertainty (for escalation), not looped on.

Ablation switches (all via --set): max_rounds=0 (frame + map only: the cost of going
options-free), library=false (same loop, no retrieval), critic=false, article_first=false,
final=select (options hidden during the work-up, revealed only for the final choice).
"""

import json
import random
import re
from typing import List, Optional

from ..llm import LLM, is_placeholder, parse_choice, parse_json
from ..retrieval import Retriever
from ..retrieval.evidence_tool import EvidenceTool
from . import agent_prompts as P
from . import prompts
from .methods import MCQ

VALID_STATUS = {"SUPPORTED", "CONTRADICTED", "NOT_ADDRESSED"}


def clean(text) -> str:
    """A model-written string, or "" if it is missing or a template echo like "<name>"."""
    if not isinstance(text, str) or is_placeholder(text):
        return ""
    return text.strip()


def ident(text) -> str:
    """An id the model refers to ("e3", " F4 ", "C1") -> "E3", "F4", "C1".
    Not clean(): ids have no real words, so clean() would treat them as placeholders."""
    if not isinstance(text, str):
        return ""
    return text.strip().strip(".()[]").upper()


def question_stem(question: str) -> str:
    """The question sentence itself ("Which of the following is the most likely diagnosis?").
    The mapper sees this, but never the case."""
    sentences = re.split(r"(?<=[.!?])\s+", question.strip())
    for sentence in reversed(sentences):
        if "?" in sentence:
            return sentence.strip()
    return sentences[-1].strip() if sentences else ""


class DxAgent:
    name = "dx_agent"

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None,
                 k: int = 4, claims: int = 2, max_rounds: int = 4, max_critiques: int = 2,
                 max_new_hypotheses: int = 2, library: bool = True, critic: bool = True,
                 article_first: bool = True, cards_per_claim: int = 2, min_score: float = 0.0,
                 final: str = "map", max_tokens: int = 1500):
        self.llm = llm
        self.k = k                                  # hypotheses in the first differential
        self.claims = claims                        # claims per hypothesis per turn
        self.max_rounds = max_rounds                # agent turns after FRAME (0 = frame + map only)
        self.max_critiques = max_critiques if critic else 0
        self.max_new_hypotheses = max_new_hypotheses   # per turn, from the agent or the critic
        self.library = library
        # How the free-text conclusion becomes an option letter:
        #   "map"    the strict options-free rule: a case-blind call matches the answer to an option
        #   "select" options stay hidden during the whole work-up and are revealed only for the
        #            final choice, together with the case and a summary of the work-up
        if final not in ("map", "select"):
            raise ValueError("final must be 'map' or 'select'")
        self.final = final
        self.max_tokens = max_tokens
        self.tool = None
        if library and max_rounds > 0:
            if retriever is None:
                raise ValueError("dx_agent needs the retriever unless library=false or max_rounds=0.")
            self.tool = EvidenceTool(retriever, cards_per_claim=cards_per_claim, min_score=min_score,
                                     article_first=article_first)

    # ====================================================================================
    # The whole run for one question
    # ====================================================================================
    def __call__(self, q: MCQ) -> dict:
        state = self.frame(q)
        first = dict(state["leading"])              # remembered for harm / recovery analysis
        rng = random.Random(q.id)                   # random dossier order, repeatable per question

        if self.max_rounds > 0:
            self.check_pending_claims(state)
            self.run_loop(q, state, rng)

        # The open-ended answers (first and final) are always mapped, so the trace shows how
        # the strictly options-free agent did, whichever final step is used.
        mapped = self.map_to_options(q, state, [first["answer"], state["leading"]["answer"]])
        first_choice, final_choice = mapped[0], mapped[1]
        state["open_choice"] = final_choice

        if self.final == "select":
            selected = self.select_option(q, state)
            if selected is not None:
                return {"answer": selected, "trace": self.trace(state, first, first_choice, False)}

        fallback = False
        if final_choice is None:
            # No option means the same as the answer: answer the normal way and count it.
            reply = self.llm.chat(prompts.COT_USER.format(question=q.question, options=q.options_text()),
                                  system=prompts.COT_SYSTEM, temperature=0.0)
            final_choice = parse_choice(reply, q.options)
            fallback = True

        return {"answer": final_choice, "trace": self.trace(state, first, first_choice, fallback)}

    # ====================================================================================
    # Every structured call goes through here
    # ====================================================================================
    def ask(self, state: dict, prompt: str, system: str, max_tokens: Optional[int] = None) -> dict:
        """One LLM call that must return a JSON object. Returns {} if it still fails to parse.

        json_mode makes vLLM produce valid JSON (constrained decoding). If a reply still cannot
        be parsed, its beginning is kept in the trace, so failures can be read, not guessed.
        """
        reply = self.llm.chat(prompt, system=system, temperature=0.0,
                              max_tokens=max_tokens or self.max_tokens, json_mode=True)
        parsed = parse_json(reply)
        if parsed:
            return parsed
        state["parse_failures"] += 1
        if len(state["raw_failures"]) < 3:
            state["raw_failures"].append(str(reply)[:1500])
        return {}

    # ====================================================================================
    # Step 1: FRAME
    # ====================================================================================
    def frame(self, q: MCQ) -> dict:
        state = {"parse_failures": 0, "raw_failures": []}
        parsed = self.ask(state, P.FRAME_USER.format(question=q.question, k=self.k, claims=self.claims),
                          P.AGENT_SYSTEM)

        findings = []
        for item in parsed.get("findings", []):
            if clean(item):
                findings.append(clean(item))

        state.update({
            "findings": findings,
            "asked": clean(parsed.get("asked")) or question_stem(q.question),
            "hypotheses": {},          # name -> {"why", "worst_fit", "status", "origin", "claims": [ids]}
            "claims": {},              # "C1" -> {"hypothesis", "text", "about", "cards", "status", "card"}
            "cards": {},               # "E1" -> card dict
            "events": [],              # everything that happened, for the trace
            "messages": [],            # harness notes shown to the agent next turn (e.g. rejections)
            "issues": {},              # "I1" -> critic issue
            "critiques": 0,
            "turns": 0,
        })
        for item in parsed.get("hypotheses", [])[:self.k]:
            if isinstance(item, dict):
                self.add_hypothesis(state, item, origin="frame")

        leading = parsed.get("leading") if isinstance(parsed.get("leading"), dict) else {}
        state["leading"] = {"hypothesis": clean(leading.get("hypothesis")),
                            "answer": clean(leading.get("answer")) or clean(leading.get("hypothesis"))}
        if not state["leading"]["answer"] and state["hypotheses"]:
            name = next(iter(state["hypotheses"]))
            state["leading"] = {"hypothesis": name, "answer": name}
        return state

    # ====================================================================================
    # Bookkeeping helpers
    # ====================================================================================
    def find_hypothesis(self, state: dict, name: str) -> Optional[str]:
        """Matches a model-written name to a known hypothesis (case-insensitive, or contained)."""
        name = clean(name).lower()
        if not name:
            return None
        for known in state["hypotheses"]:
            if known.lower() == name:
                return known
        for known in state["hypotheses"]:
            if name in known.lower() or known.lower() in name:
                return known
        return None

    def add_claim(self, state: dict, hypothesis: str, text, about) -> None:
        text = clean(text)
        if not text:
            return
        claim_id = f"C{len(state['claims']) + 1}"
        state["claims"][claim_id] = {"hypothesis": hypothesis, "text": text,
                                     "about": clean(about) or "other", "cards": None,
                                     "status": None, "card": None}
        state["hypotheses"][hypothesis]["claims"].append(claim_id)

    def add_hypothesis(self, state: dict, item: dict, origin: str) -> Optional[str]:
        name = clean(item.get("name"))
        if not name:
            return None
        known = self.find_hypothesis(state, name)
        if known:                                  # already there: maybe bring it back
            if state["hypotheses"][known]["status"] == "ruled_out" and origin == "critic":
                state["hypotheses"][known]["status"] = "live"
            return known
        state["hypotheses"][name] = {"why": clean(item.get("why")), "worst_fit": clean(item.get("worst_fit")),
                                     "status": "live", "origin": origin, "claims": [], "rule_out": None}
        for claim in (item.get("claims") or [])[:self.claims]:
            if isinstance(claim, dict):
                self.add_claim(state, name, claim.get("text"), claim.get("about"))
        state["events"].append({"turn": state["turns"], "event": "add_hypothesis", "by": origin, "name": name})
        return name

    def check_pending_claims(self, state: dict) -> None:
        """Runs consult_reference for every claim not checked yet - all hypotheses together."""
        pending = []
        for cid, claim in state["claims"].items():
            if claim["cards"] is None:              # None = not looked up yet; [] = nothing found
                pending.append(cid)
        if not pending:
            return

        if self.tool is None:                       # library switched off
            for cid in pending:
                state["claims"][cid]["cards"] = []
            return

        requests = []
        for cid in pending:
            claim = state["claims"][cid]
            requests.append({"hypothesis": claim["hypothesis"], "claim": claim["text"], "about": claim["about"]})

        for cid, cards in zip(pending, self.tool.consult(requests)):
            ids = []
            for card in cards:
                card_id = f"E{len(state['cards']) + 1}"
                state["cards"][card_id] = card
                ids.append(card_id)
            state["claims"][cid]["cards"] = ids

    def render_findings(self, state: dict) -> str:
        if not state["findings"]:
            return "(none extracted; read the case)"
        return "\n".join(f"F{i + 1}. {f}" for i, f in enumerate(state["findings"]))

    def render_dossier(self, state: dict, rng: random.Random, with_cards: bool = True) -> str:
        """Every hypothesis with its claims and evidence cards, in a random order."""
        names = list(state["hypotheses"])
        rng.shuffle(names)
        blocks = []
        for name in names:
            h = state["hypotheses"][name]
            status = "RULED OUT" if h["status"] == "ruled_out" else "live"
            lines = [f"## {name} [{status}] - explains: {h['why'] or '?'}"]
            if h["worst_fit"]:
                lines.append(f"   fits worst: {h['worst_fit']}")
            for cid in h["claims"]:
                c = state["claims"][cid]

                # e.g. "C3: Haemolysis causes anaemia -> CONTRADICTED (E5)"
                verdict = ""
                if c["status"]:
                    verdict = f" -> {c['status']}"
                    if c["card"]:
                        verdict += f" ({c['card']})"
                lines.append(f"   {cid}: {c['text']}{verdict}")
                if not with_cards:
                    continue
                if not self.library:
                    lines.append("      (library switched off: use your own knowledge)")
                elif not c["cards"]:
                    lines.append("      NOT FOUND in the library")
                for card_id in c["cards"] or []:
                    card = state["cards"][card_id]
                    single = " [single source]" if card.get("single_source") else ""
                    lines.append(f"      {card_id} {card['source']}{single}: \"{card['text']}\"")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def open_issues(self, state: dict) -> List[str]:
        return [iid for iid, issue in state["issues"].items() if not issue["resolved"]]

    # ====================================================================================
    # Step 2 + 3: the verification loop with the critic
    # ====================================================================================
    def run_loop(self, q: MCQ, state: dict, rng: random.Random) -> None:
        answers_at_critique = []                    # leading answer each time the critic ran
        while state["turns"] < self.max_rounds:
            wants_critique, asked_new_claims = self.agent_turn(q, state, rng)
            self.check_pending_claims(state)

            budget_spent = state["turns"] >= self.max_rounds
            if not (wants_critique or budget_spent):
                continue
            if wants_critique and asked_new_claims and not budget_spent:
                # The agent asked for evidence AND for the critic in the same turn: let it read
                # the new evidence first, otherwise the critic reviews a half-finished work-up.
                state["messages"].append("Your critique request was deferred: first read the new evidence.")
                continue

            # --- critique, if allowed ---
            if state["critiques"] >= self.max_critiques:
                break
            new_blocking = self.critique(q, state, rng)
            answers_at_critique.append(state["leading"]["answer"].lower())

            if not new_blocking:
                break                               # the work-up survived review
            if len(answers_at_critique) >= 2 and answers_at_critique[-1] == answers_at_critique[-2]:
                break                               # the answer is stable across critiques: stop
            if state["turns"] >= self.max_rounds:
                break                               # no turn left to respond: issues stay open
            self.check_pending_claims(state)        # critic's new hypotheses get checked too

    def agent_turn(self, q: MCQ, state: dict, rng: random.Random):
        """One agent turn. Returns (requested critique?, requested new evidence?)."""
        state["turns"] += 1
        # Open critic issues the agent must answer this turn.
        open_ids = self.open_issues(state)
        issues_text = ""
        if open_ids:
            lines = ["", "Open issues from the critic:"]
            for iid in open_ids:
                issue = state["issues"][iid]
                lines.append(f"{iid} ({issue['type']}): {issue['detail']} - to settle it: {issue['action']}")
            issues_text = "\n".join(lines) + "\n"

        # Notes from the harness, e.g. "Rule-out rejected: E9 is not a card id". Shown once.
        messages_text = ""
        if state["messages"]:
            lines = ["", "Notes from the system:"]
            for message in state["messages"]:
                lines.append(f"- {message}")
            messages_text = "\n".join(lines) + "\n"
            state["messages"] = []

        prompt = P.ROUND_USER.format(
            question=q.question, asked=state["asked"], findings=self.render_findings(state),
            dossier=self.render_dossier(state, rng), issues=issues_text, messages=messages_text,
            turns_left=self.max_rounds - state["turns"] + 1, claims=self.claims,
            status_rule=P.STATUS_RULE_LIBRARY if self.library else P.STATUS_RULE_NO_LIBRARY,
            respond_rule=P.RESPOND_RULE if open_ids else "")
        parsed = self.ask(state, prompt, P.AGENT_SYSTEM)
        if not parsed:
            state["messages"].append("Your last reply was not valid JSON. Reply with the JSON object only.")
            return False, False

        # --- 1. claim statuses ---
        for item in parsed.get("claim_status", []) or []:
            if not isinstance(item, dict):
                continue
            cid, status = ident(item.get("claim")), ident(item.get("status"))
            if cid in state["claims"] and status in VALID_STATUS:
                state["claims"][cid]["status"] = status
                card = ident(item.get("card"))
                state["claims"][cid]["card"] = card if card in state["cards"] else None

        # --- 2. responses to critic issues: an answered issue counts as resolved for the agent;
        #        the critic may re-open it next time only by citing counter-evidence ---
        for item in parsed.get("responses", []) or []:
            if isinstance(item, dict):
                iid = ident(item.get("issue"))
                if iid in state["issues"] and clean(item.get("reply")):
                    state["issues"][iid]["reply"] = clean(item.get("reply"))
                    state["issues"][iid]["resolved"] = True

        # --- 3. tools ---
        wants_critique, new_claims, new_hypotheses = False, {}, 0
        for action in parsed.get("actions", []) or []:
            if not isinstance(action, dict):
                continue
            tool = clean(action.get("tool")).lower()
            if tool in ("request_critique", "finish"):
                wants_critique = True
            elif tool == "consult_reference":
                name = self.find_hypothesis(state, action.get("hypothesis"))
                if name and new_claims.get(name, 0) < self.claims:
                    before = len(state["claims"])
                    self.add_claim(state, name, action.get("claim"), action.get("about"))
                    if len(state["claims"]) > before:
                        new_claims[name] = new_claims.get(name, 0) + 1
            elif tool == "add_hypothesis" and new_hypotheses < self.max_new_hypotheses:
                if self.add_hypothesis(state, action, origin="agent"):
                    new_hypotheses += 1
            elif tool == "rule_out":
                self.rule_out(state, action)

        # A turn with no new evidence, hypotheses or critique request means the agent has
        # nothing left to check: treat it as "I am done" instead of wasting the next turn.
        if not new_claims and new_hypotheses == 0 and not wants_critique:
            wants_critique = True

        leading = parsed.get("leading") if isinstance(parsed.get("leading"), dict) else {}
        if clean(leading.get("answer")):
            state["leading"] = {"hypothesis": clean(leading.get("hypothesis")), "answer": clean(leading.get("answer"))}
        state["events"].append({"turn": state["turns"], "event": "turn", "critique": wants_critique,
                                "new_claims": sum(new_claims.values()), "leading": state["leading"]["answer"]})
        asked_new = bool(new_claims) or new_hypotheses > 0
        return wants_critique, asked_new

    def rule_out(self, state: dict, action: dict) -> None:
        """Accepted only with a real citation: an existing card id or an existing finding id."""
        name = self.find_hypothesis(state, action.get("hypothesis"))
        evidence = ident(action.get("evidence"))

        # Is the citation real? Either a card the library returned, or a finding of the case.
        is_card = evidence in state["cards"]
        is_finding = False
        if re.fullmatch(r"F\d+", evidence):
            is_finding = 1 <= int(evidence[1:]) <= len(state["findings"])

        live = 0
        for h in state["hypotheses"].values():
            if h["status"] == "live":
                live += 1

        problem = None
        if not name:
            problem = "unknown hypothesis"
        elif state["hypotheses"][name]["status"] == "ruled_out":
            return                                  # already out, nothing to do
        elif not (is_card or is_finding):
            problem = f"'{evidence or 'nothing'}' is not a card or finding id"
        elif live <= 1:
            problem = "it is the last live hypothesis (add another one first)"

        if problem:
            state["messages"].append(f"Rule-out of '{action.get('hypothesis')}' rejected: {problem}.")
            state["events"].append({"turn": state["turns"], "event": "rule_out_rejected", "why": problem})
            return
        state["hypotheses"][name]["status"] = "ruled_out"
        state["hypotheses"][name]["rule_out"] = {"evidence": evidence, "reason": clean(action.get("reason"))}
        state["events"].append({"turn": state["turns"], "event": "rule_out", "name": name, "evidence": evidence})

    def critique(self, q: MCQ, state: dict, rng: random.Random) -> List[str]:
        """Runs the critic. Returns the ids of NEW issues that could change the answer."""
        state["critiques"] += 1

        rule_outs = []
        for name, h in state["hypotheses"].items():
            if h["status"] == "ruled_out" and h["rule_out"]:
                rule_outs.append(f"- {name}: {h['rule_out']['reason']} (cites {h['rule_out']['evidence']})")

        # The critic sees its own earlier issues with the agent's replies, so it can
        # accept a rebuttal instead of raising the same point again.
        previous = []
        for iid, issue in state["issues"].items():
            previous.append(f"{iid}: {issue['detail']} -> colleague replied: {issue.get('reply') or '(no reply)'}")
        prompt = P.CRITIC_USER.format(
            question=q.question, asked=state["asked"], findings=self.render_findings(state),
            dossier=self.render_dossier(state, rng), rule_outs="\n".join(rule_outs) or "(none)",
            answer=state["leading"]["answer"], hypothesis=state["leading"]["hypothesis"] or "?",
            previous="\n".join(previous) or "(none)")
        parsed = self.ask(state, prompt, P.CRITIC_SYSTEM)
        if not parsed:
            return []

        # Previous issues the critic now considers unresolved are re-opened.
        for item in parsed.get("previous_issues", []) or []:
            if isinstance(item, dict):
                iid = ident(item.get("issue"))
                if iid in state["issues"] and item.get("resolved") is False:
                    state["issues"][iid]["resolved"] = False

        # New issues; only those that could change the answer block finishing.
        known = {i["detail"].lower() for i in state["issues"].values()}
        blocking = []
        for item in parsed.get("issues", []) or []:
            if not isinstance(item, dict) or not clean(item.get("detail")):
                continue
            if clean(item.get("detail")).lower() in known:
                continue                            # a repeated issue is not a new one
            iid = f"I{len(state['issues']) + 1}"

            # An issue blocks only if the critic names a DIFFERENT answer it would give.
            # In the first dev run the critic marked 296 of 298 reviews as answer-changing,
            # mostly without saying what the better answer was; that kept every loop open.
            alternative = clean(item.get("alternative_answer"))
            current = state["leading"]["answer"].lower()
            changes = (item.get("changes_answer") is True and bool(alternative)
                       and alternative.lower() != current)

            state["issues"][iid] = {"type": clean(item.get("type")) or "other", "detail": clean(item.get("detail")),
                                    "action": clean(item.get("action")), "alternative_answer": alternative,
                                    "changes_answer": changes, "resolved": not changes,
                                    "critique": state["critiques"]}
            if changes:
                blocking.append(iid)

        # Overlooked diagnoses enter the differential and will be checked like any other.
        added = 0
        for item in parsed.get("new_hypotheses", []) or []:
            if isinstance(item, dict) and added < self.max_new_hypotheses:
                if self.add_hypothesis(state, item, origin="critic"):
                    added += 1
        for name in parsed.get("restore", []) or []:
            known_name = self.find_hypothesis(state, name)
            if known_name and state["hypotheses"][known_name]["status"] == "ruled_out":
                state["hypotheses"][known_name]["status"] = "live"
                state["events"].append({"turn": state["turns"], "event": "restored", "name": known_name})

        alternative = parsed.get("strongest_alternative") if isinstance(parsed.get("strongest_alternative"), dict) else {}
        state["events"].append({"turn": state["turns"], "event": "critique", "blocking": blocking,
                                "new_hypotheses": added, "alternative": clean(alternative.get("hypothesis"))})
        return blocking

    # ====================================================================================
    # Step 4: MAP (the case is never shown here)
    # ====================================================================================
    def map_to_options(self, q: MCQ, state: dict, answers: List[str]) -> List[Optional[str]]:
        unique = []
        for answer in answers:
            if answer and answer not in unique:
                unique.append(answer)
        if not unique:
            return [None] * len(answers)

        numbered = []
        for number, answer in enumerate(unique, start=1):
            numbered.append(f"{number}. {answer}")

        prompt = P.MAP_USER.format(asked=state["asked"], stem=question_stem(q.question), options=q.options_text(),
                                   answers="\n".join(numbered))
        parsed = self.ask(state, prompt, P.MAP_SYSTEM, max_tokens=200)
        choice_of = {}
        for item in parsed.get("matches", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("answer"))
            except (TypeError, ValueError):
                continue
            letter = str(item.get("choice", "")).strip().upper().strip("().")
            if 1 <= number <= len(unique) and letter in q.options:
                choice_of[unique[number - 1]] = letter
        return [choice_of.get(answer) for answer in answers]

    def select_option(self, q: MCQ, state: dict) -> Optional[str]:
        """final="select": the options are revealed only now, after the whole work-up."""
        lines = []
        for name, h in state["hypotheses"].items():
            supported = contradicted = 0
            for cid in h["claims"]:
                if state["claims"][cid]["status"] == "SUPPORTED":
                    supported += 1
                elif state["claims"][cid]["status"] == "CONTRADICTED":
                    contradicted += 1
            status = "ruled out" if h["status"] == "ruled_out" else "live"
            lines.append(f"- {name} [{status}]: {supported} claims supported, {contradicted} contradicted")

        open_issues = []
        for iid in self.open_issues(state):
            open_issues.append(f"- {state['issues'][iid]['detail']}")

        prompt = P.SELECT_USER.format(question=q.question, asked=state["asked"],
                                      hypothesis=state["leading"]["hypothesis"] or "?",
                                      answer=state["leading"]["answer"] or "?",
                                      workup="\n".join(lines) or "(none)",
                                      issues="\n".join(open_issues) or "(none)",
                                      options=q.options_text())
        parsed = self.ask(state, prompt, P.SELECT_SYSTEM, max_tokens=600)
        return parse_choice(json.dumps(parsed), q.options) if parsed else None

    # ====================================================================================
    # Trace (everything the analysis needs)
    # ====================================================================================
    def trace(self, state: dict, first: dict, first_choice: Optional[str], fallback: bool) -> dict:
        # Each hypothesis with its claims written out in full (instead of claim ids).
        hypotheses = {}
        for name, h in state["hypotheses"].items():
            record = dict(h)
            record["claims"] = [state["claims"][cid] for cid in h["claims"]]
            hypotheses[name] = record

        open_issues = [state["issues"][iid] for iid in self.open_issues(state)]

        first_record = dict(first)
        first_record["choice"] = first_choice      # the option the FIRST answer maps to

        return {
            "findings": state["findings"],
            "asked": state["asked"],
            "first": first_record,
            "final": state["leading"],
            "mapping_fallback": fallback,
            "turns": state["turns"],
            "critiques": state["critiques"],
            "open_issues": open_issues,
            "hypotheses": hypotheses,
            "cards": state["cards"],
            "issues": state["issues"],
            "events": state["events"],
            "parse_failures": state["parse_failures"],
            "raw_failures": state["raw_failures"],
            "open_choice": state.get("open_choice"),       # the strictly options-free answer
            "final_step": self.final,
        }
