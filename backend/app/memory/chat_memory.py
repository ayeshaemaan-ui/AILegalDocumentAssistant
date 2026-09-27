"""Chat memory (Tier 1 M2: Chat Memory + Module 6: Memory).

Naive memory replays the whole transcript on every turn. That breaks twice
over: input cost grows quadratically with conversation length, and eventually
the history crowds out the retrieved excerpts, which are the part that actually
answers the question.

The strategy here is **windowed buffer + rolling summary**:

    [ rolling summary of turns 1..n-6 ]  +  [ last 6 turns verbatim ]

Recent turns stay exact because follow-ups depend on their precise wording
("what about clause 9.2?"). Older turns compress into a summary that keeps the
facts a follow-up might need - which documents are in play, what was already
asked and answered - and drops the phrasing. Summarisation runs on Haiku, off
the critical path, and costs a fraction of a cent.

Memory is also what makes query rewriting work: "and the other one?" is
meaningless alone, but resolvable against the summary plus the last few turns.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from app.config import Settings, get_settings
from app.memory.session import Message, Session, get_session_store

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = """You maintain a running summary of a conversation between a user \
and a contract-research assistant.

Write a compact factual summary that a later turn could rely on. Keep: which \
documents or contracts are under discussion, what the user has asked, what was found \
(including specific clause references, figures and dates), and any filter or scope the \
user set. Drop: pleasantries, the assistant's phrasing, and anything already \
superseded by a later turn.

Write it as prose in the third person, under 200 words. If an earlier summary is \
provided, merge the new turns into it rather than appending - the result should read \
as one summary, not a log."""


class ConversationMemory:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = get_session_store(self.settings)

    # ------------------------------------------------------------ reading
    def recent_turns(self, session: Session) -> list[Message]:
        window = self.settings.max_history_turns * 2  # a "turn" is user + assistant
        return session.messages[-window:] if session.messages else []

    def as_api_messages(self, session: Session) -> list[dict[str, Any]]:
        """History in Messages API shape, ready to prepend to the current turn."""
        out: list[dict[str, Any]] = []
        for message in self.recent_turns(session):
            if message.role not in ("user", "assistant"):
                continue
            content = message.content.strip()
            if not content:
                continue
            # Trim long assistant answers - the citations and reasoning in an
            # old answer are rarely needed, but its conclusion often is.
            if message.role == "assistant" and len(content) > 1500:
                content = content[:1500] + " […]"
            out.append({"role": message.role, "content": content})

        # The API requires the first message to be from the user, and forbids a
        # trailing assistant turn before our new one.
        while out and out[0]["role"] != "user":
            out.pop(0)
        while out and out[-1]["role"] == "assistant" and len(out) % 2 == 1:
            break
        return out

    def context_summary(self, session: Session) -> str | None:
        """What the query rewriter and answer prompt see as prior context."""
        parts: list[str] = []
        if session.summary:
            parts.append(session.summary)
        recent = self.recent_turns(session)
        if recent:
            tail = "\n".join(
                f"{m.role}: {m.content[:300]}" for m in recent[-4:] if m.content.strip()
            )
            if tail:
                parts.append(f"Most recent turns:\n{tail}")
        return "\n\n".join(parts) if parts else None

    # ------------------------------------------------------------ writing
    def should_summarize(self, session: Session) -> bool:
        turns = sum(1 for m in session.messages if m.role == "user")
        return turns > self.settings.summarize_history_after

    async def maybe_summarize(self, session: Session) -> str | None:
        """Refresh the rolling summary once the conversation outgrows the window."""
        if not self.should_summarize(session):
            return session.summary

        from app.core.cost import get_cost_tracker
        from app.core.llm import get_llm_client

        window = self.settings.max_history_turns * 2
        older = session.messages[:-window] if len(session.messages) > window else []
        if not older:
            return session.summary

        transcript = "\n".join(f"{m.role}: {m.content[:600]}" for m in older)
        user_turn = (
            (f"Existing summary:\n{session.summary}\n\n" if session.summary else "")
            + f"New turns to fold in:\n{transcript}"
        )

        client = get_llm_client(self.settings)
        try:
            result = await client.complete(
                system=SUMMARY_SYSTEM,
                messages=[{"role": "user", "content": user_turn}],
                model=self.settings.resolved_fast_model,
                max_tokens=512,
                effort="low",
            )
        except Exception as exc:  # memory is a nicety, never a hard failure
            logger.warning("History summarisation failed: %s", exc)
            return session.summary

        get_cost_tracker(self.settings).record(
            result.usage, stage="memory.summarize", session_id=session.session_id
        )
        summary = result.text.strip()
        if summary:
            self.store.set_summary(session.session_id, summary)
            session.summary = summary
        return summary

    def derive_title(self, first_question: str) -> str:
        """Session title from the opening question - no model call needed."""
        title = " ".join(first_question.split())
        return title[:77] + "…" if len(title) > 80 else title


def build_langchain_memory(session_id: str, settings: Settings | None = None):
    """Expose the same history as a LangChain `BaseChatMessageHistory`.

    Lets any LangChain chain or agent in this project read and write the exact
    same conversation store the FastAPI path uses, instead of keeping a second,
    divergent copy of the history.
    """
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    store = get_session_store(settings)

    class SessionBackedHistory(BaseChatMessageHistory):
        @property
        def messages(self) -> list[BaseMessage]:
            session = store.get(session_id)
            if session is None:
                return []
            out: list[BaseMessage] = []
            for m in session.messages:
                if m.role == "user":
                    out.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    out.append(AIMessage(content=m.content))
            return out

        def add_message(self, message: BaseMessage) -> None:
            role = "user" if message.type == "human" else "assistant"
            store.append_message(session_id, role, str(message.content))

        def clear(self) -> None:
            store.delete(session_id)

    return SessionBackedHistory()
