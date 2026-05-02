"""LiveKit voice agent with MongoDB Atlas integration.

Demonstrates five integration patterns:

1. RAG with $vectorSearch              -> search_knowledge tool
2. Agentic memory tools                -> remember/recall/forget/search_memories
3. Identify + pre-load context         -> preload_user (entrypoint)
4. Function-tool CRUD                  -> @function_tool methods
5. Session report persistence          -> on_session_end
"""

from dotenv import load_dotenv
load_dotenv(".env.local")

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RunContext,
    ToolError,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.llm import ChatMessage, StopResponse
from livekit.agents.voice import UserInputTranscribedEvent
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from bson.objectid import ObjectId

# Track agent's own corrections with timestamps to avoid loops
# Format: {normalized_text: timestamp}
agent_corrections = {}

def normalize_text(t: str) -> str:
    """Unified normalization for A.T.E. (lowercase alphanumeric + spaces)."""
    return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).strip()

from db.client import aclose, get_db
from tools.embeddings import embed_text
from tools.memory import forget, list_memories, recall, remember, search_memory
from tools.fact_checker import (
    process_claim_for_facts,
    reason_on_claim,
    speak_correction,
    set_livekit_llm_instance,
    el_play,
)

load_dotenv(".env.local")

# --- DEMO UI: ASCII ART & LOG CLEANUP ---
def print_banner():
    banner = r"""
     ██████╗██╗   ██╗██████╗ ██╗██╗     
    ██╔════╝╚██╗ ██╔╝██╔══██╗██║██║     
    ╚█████╗  ╚████╔╝ ██████╔╝██║██║     
     ╚═══██╗  ╚██╔╝  ██╔══██╗██║██║     
    ██████╔╝   ██║   ██████╔╝██║███████╗
    ╚═════╝    ╚═╝   ╚═════╝ ╚═╝╚══════╝
    SYBIL: The Truth-Engine | May 2, 2026
    ------------------------------------
    """
    print(banner)

# Suppress noisy logs from standard libraries
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("voyage").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# Fallback identity used only when ctx.job.metadata is absent, e.g. when
# running `uv run src/agent.py console`. The frontend always provides a
# real per-browser user_id via agent dispatch metadata.
DEFAULT_USER_ID = "user_1"
DEFAULT_TENANT_ID = "default"

# Allow-list for identity fields that `update_profile` can write to the
# `users` document. Values are dotted paths so nested preferences work
# without a second tool. Anything outside this map belongs in `memories`.
_PROFILE_PATHS = {
    "name": "name",
    "email": "email",
    "preferred_language": "preferences.language",
    "timezone": "preferences.timezone",
}


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def _vector_search_knowledge(
    db: AsyncDatabase, query: str, limit: int = 3
) -> list[dict]:
    """Run the shared knowledge vector search and return {title, content} docs."""
    query_embedding = await embed_text(query, input_type="query")
    pipeline = [
        {
            "$vectorSearch": {
                "index": "knowledge_embedding_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": limit,
            }
        },
        {"$project": {"title": 1, "content": 1, "_id": 0}},
    ]
    cursor = await db.knowledge.aggregate(pipeline)
    return await cursor.to_list(length=limit)


import subprocess

def control_media(action: str):
    """Robust media control for macOS Chrome/Safari."""
    # Use 'key code 49' (Space) for a more universal toggle if JS fails
    # But for demo precision, we target the browser directly
    browser = "Google Chrome" # Default to Chrome
    
    script = f'''
    tell application "{browser}"
        if (count windows) > 0 then
            execute front window's active tab javascript "var v = document.querySelector('video'); if (v) v.{action}();"
        end if
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True)
    except Exception:
        pass

class MongoAgent(Agent):
    """Voice agent that wires MongoDB Atlas into the LiveKit pipeline."""

    def __init__(
        self, *, chat_ctx: ChatContext, user_id: str, tenant_id: str, livekit_llm: inference.LLM
    ) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=(
                "You are SYBIL, an invisible background monitor. "
                "YOU ARE NOT A CHATBOT. NEVER GENERATE TEXT. NEVER SPEAK TO THE USER. "
                "Your internal response loop is DISABLED. "
                "Only the background fact-checker will play pre-cached audio bytes."
            ),
        )
        self._user_id = user_id
        self._tenant_id = tenant_id
        self._livekit_llm = livekit_llm
        self._is_speaking_correction = False
        self._last_spoke_at = 0.0
        self._primary_speaker_id = None

    async def on_enter(self) -> None:
        logger.info("SYBIL initialized in background-only mode.")

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """MUZZLE: Stop any response from the agent pipeline."""
        raise StopResponse()

    @function_tool()
    async def lookup_order(self, context: RunContext, order_id: str) -> str:
        """Look up an order by its ID. Returns items, total, and status."""
        db = await get_db()
        order = await db.orders.find_one({"order_id": order_id})
        if not order:
            raise ToolError(f"Order {order_id} not found.")
        return json.dumps(
            {
                "order_id": order["order_id"],
                "items": order["items"],
                "total": order["total"],
                "status": order["status"],
            }
        )

    @function_tool()
    async def search_knowledge(
        self, context: RunContext, query: str
    ) -> str:
        """Search the shared knowledge base for facts the user asks about.

        Use when the user asks a question about voice agents, MongoDB,
        LiveKit, STT/LLM/TTS providers, session handling, or anything
        else you are not confident answering from prior context. Returns
        a JSON object with a `results` array of `{title, content}`.
        """

        async def _speak_status_update(delay: float = 0.5) -> None:
            await asyncio.sleep(delay)
            await context.session.generate_reply(
                instructions=(
                    f"You are searching the knowledge base for '{query}' "
                    "but it is taking a moment. Give the user a brief, "
                    "one-sentence update that you are looking it up."
                )
            )

        status_task = asyncio.create_task(_speak_status_update(0.5))
        try:
            db = await get_db()
            results = await _vector_search_knowledge(db, query, limit=3)
        finally:
            status_task.cancel()
        return json.dumps({"results": results})

    @function_tool()
    async def update_profile(
        self, context: RunContext, field: str, value: str
    ) -> str:
        """Update an identity field on the user's profile.

        Use for name, email, preferred_language, or timezone. These are
        first-class profile fields stored on the `users` document and
        loaded at session start. For anything else, use remember_detail.
        """
        if field not in _PROFILE_PATHS:
            raise ToolError(
                f"Unknown profile field '{field}'. "
                f"Allowed: {sorted(_PROFILE_PATHS)}"
            )
        db = await get_db()
        now = _now()
        await db.users.update_one(
            {"user_id": self._user_id},
            {
                "$set": {_PROFILE_PATHS[field]: value, "updated_at": now},
                "$setOnInsert": {
                    "user_id": self._user_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        return f"Updated {field} to {value}."

    @function_tool()
    async def remember_detail(
        self, context: RunContext, memory_type: str, content: str
    ) -> str:
        """Store or replace a fact under memory_type.

        Use for preferences, allergies, pronouns, or anything the user
        volunteers. Pick a short specific label like 'favorite_color'
        rather than a generic one like 'preferences'.
        """
        db = await get_db()
        return await remember(
            db, self._user_id, self._tenant_id, memory_type, content
        )

    @function_tool()
    async def recall_detail(
        self, context: RunContext, memory_type: str
    ) -> str:
        """Return the value stored under memory_type by exact label.

        Returns 'No memory found' when the label is not set. Prefer
        search_memories when you are unsure which label stores the fact.
        """
        db = await get_db()
        return await recall(db, self._user_id, self._tenant_id, memory_type)

    @function_tool()
    async def forget_detail(
        self, context: RunContext, memory_type: str
    ) -> str:
        """Delete the value stored under memory_type."""
        db = await get_db()
        return await forget(db, self._user_id, self._tenant_id, memory_type)

    @function_tool()
    async def search_memories(self, context: RunContext, query: str) -> str:
        """Find memories by meaning using hybrid vector and text search.

        Use when you do not know the exact label or when pulling
        background context. Returns a list of {memory_type, content}
        objects so you can follow up with recall_detail or forget_detail.
        """
        db = await get_db()
        results = await search_memory(
            db, self._user_id, self._tenant_id, query, limit=3
        )
        return json.dumps({"results": results})

    @function_tool()
    async def list_user_memories(self, context: RunContext) -> str:
        """Return every slot stored for this user, newest first."""
        db = await get_db()
        results = await list_memories(db, self._user_id, self._tenant_id)
        return json.dumps(
            {
                "results": [
                    {"memory_type": r["memory_type"], "content": r["content"]}
                    for r in results
                ]
            }
        )


async def automated_fact_check(session: AgentSession, db: AsyncDatabase, text: str, livekit_llm: inference.LLM, mongo_agent: Optional[Any] = None, fast_path_only: bool = False):
    """Background task to identify and process factual claims from a transcript."""
    # Use the passed mongo_agent or look it up from session
    active_agent = mongo_agent or getattr(session, '_agent', None)
    
    # --- STRICT STATE GATE ---
    if not active_agent: return
    if active_agent._is_speaking_correction: return
    
    import time
    if (time.time() - active_agent._last_spoke_at) < 5.0:
        return

    # Only log for final checks to keep logs clean
    if not fast_path_only: logger.info(f"Automated check for transcript: {text}")
    now = _now()
    
    # --- FAST-PATH: Normalized Exact Match (< 20ms) ---
    fast_key = normalize_text(text)
    
    try:
        # Check if we have an EXACT match for this normalized string
        fast_match = await db.claims.find_one({
            "fast_key": fast_key,
            "status": "completed",
            "audio_bytes": {"$exists": True}
        })
        
        if fast_match:
            print(f"\n[⚡ FAST-PATH HIT] Key: {fast_key}")
            print(f" -> MATCH: {fast_match['text']}")
            control_media("pause")
            if active_agent: active_agent._is_speaking_correction = True
            
            # Blacklist our own correction before we say it
            if fast_match.get("correction"):
                agent_corrections[normalize_text(fast_match["correction"])] = time.time()

            try:
                el_play.play(fast_match["audio_bytes"])
            finally:
                import time
                if active_agent:
                    active_agent._is_speaking_correction = False
                    active_agent._last_spoke_at = time.time()
                
                # Small delay before resume to allow MacBook CPU to settle
                time.sleep(0.5) 
                control_media("play")
            return # EXIT IMMEDIATELY
    except Exception as e:
        logger.error(f"Fast-path check failed: {e}")

    # If we are only doing fast-path (interim updates), exit here
    if fast_path_only:
        return

    # --- PROACTIVE MEMORY PASS (Vector Search fallback) ---
    try:
        # Embed the raw transcript immediately
        transcript_embedding = await embed_text(text)
        
        # Search for semantically identical verified claims
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "claims_embedding_index",
                    "path": "embedding",
                    "queryVector": transcript_embedding,
                    "numCandidates": 10,
                    "limit": 1,
                }
            },
            {
                "$match": {
                    "status": "completed",
                    "verdict": "False",
                    "audio_bytes": {"$exists": True}
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    "correction": 1,
                    "audio_bytes": 1,
                    "text": 1,
                    "fast_key": 1
                }
            }
        ]
        
        cursor = await db.claims.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if results and results[0]["score"] > 0.95: # Increased to 0.95 for absolute precision
            match = results[0]
            print(f"\n[🎯 PROACTIVE HIT] Score: {match['score']:.2f}")
            print(f" -> MATCH: {match['text']}")
            control_media("pause")
            
            # Update cache with fast_key for next time
            await db.claims.update_one(
                {"_id": match["_id"]},
                {"$set": {"fast_key": normalize_text(text)}}
            )
            
            if active_agent: 
                active_agent._is_speaking_correction = True
                # Blacklist the correction so we don't fact-check our own voice
                if match.get("correction"): 
                    import time
                    agent_corrections[normalize_text(match["correction"])] = time.time()
                    logger.info(f"Blacklisted Super-Nitro correction: {match['correction']}")
            
            try:
                el_play.play(match["audio_bytes"])
            finally:
                import time
                if active_agent:
                    active_agent._is_speaking_correction = False
                    active_agent._last_spoke_at = time.time()
                
                # Small delay before resume
                time.sleep(0.5)
                control_media("play")
                
            return # Skip LLM extraction entirely
            
    except Exception as e:
        logger.error(f"Proactive memory pass failed: {e}")

    # --- LLM EXTRACTION PASS (If no proactive hit) ---
    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="user",
        content=(
            "Analyze the following transcript segment. It may contain conversational filler and multiple factual claims: \n"
            f"'{text}'\n\n"
            "Task: Extract EVERY checkable factual claim. "
            "A claim is any statement that can be verified as true or false. "
            "Return the results as a JSON list of strings. "
            "Example: ['London population is 9 million', 'The sky is green'] "
            "If no claims exist, return []."
        )
    )

    stream = livekit_llm.chat(chat_ctx=chat_ctx)
    raw_extraction = ""
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            raw_extraction += chunk.delta.content
    
    # Process multiple claims
    try:
        # Clean up common LLM formatting issues
        cleaned = raw_extraction.strip()
        if "```" in cleaned: cleaned = cleaned.split("```")[1].replace("json", "").strip()
        claims_to_check = json.loads(cleaned)
        if not isinstance(claims_to_check, list): claims_to_check = [str(claims_to_check)]
    except:
        claims_to_check = []

    for claim_text in claims_to_check:
        if not claim_text or claim_text == "NO_CLAIM": continue
        
        print(f"\n[🔍 NEW CLAIM] {claim_text}")
        claim_embedding = await embed_text(claim_text)
        
        claim_doc = {
            "text": claim_text,
            "fast_key": normalize_text(claim_text), # STORE THE FAST KEY
            "embedding": claim_embedding, # STORE THE EMBEDDING
            "timestamp_ingested": now,
            "status": "pending",
            "whisper_transcript_segment": text,
            "retrieval_log_ids": [],
            "source_domains_consulted": [],
            "last_updated": now,
        }
        claim_result = await db.claims.insert_one(claim_doc)
        
        # Trigger fact-checking process in the background
        asyncio.create_task(process_claim_for_facts(db, claim_result.inserted_id, livekit_llm, active_agent))


async def check_and_resume_incomplete_claims(db: AsyncDatabase, livekit_llm: inference.LLM, mongo_agent: Optional[Any] = None):
    logger.info("Checking for and resuming incomplete claims...")
    incomplete_statuses = ["pending", "retrieving", "reasoning", "speaking", "failed_speaking"]
    cursor = db.claims.find({"status": {"$in": incomplete_statuses}})

    async for claim in cursor:
        claim_id = claim["_id"]
        status = claim["status"]
        logger.info(f"Resuming claim {claim_id} with status: {status}")

        # Trigger the appropriate function based on status
        if status == "pending":
            asyncio.create_task(process_claim_for_facts(db, claim_id, livekit_llm, mongo_agent))
        elif status == "retrieving": # If it was retrieving, restart fact checking
            asyncio.create_task(process_claim_for_facts(db, claim_id, livekit_llm, mongo_agent))
        elif status == "reasoning":
            asyncio.create_task(reason_on_claim(db, claim_id, livekit_llm, mongo_agent))
        elif status == "failed_speaking":
            # Correction failed previously. Don't auto-retry at startup to avoid backlog dump.
            logger.info(f"Claim {claim_id} previously failed speaking; skipping auto-resume.")
            pass
        elif status == "speaking" and claim.get("correction"): # Only speak if correction exists
            asyncio.create_task(speak_correction(db, claim_id, claim["correction"], mongo_agent))
    logger.info("Finished checking for incomplete claims.")


async def preload_user(user_id: str, tenant_id: str) -> ChatContext:
    """Pattern 3: load user data into the chat context before the session.

    Upserts the `users` row so every connected user (including anonymous
    cookie visitors) has a stable profile document, then appends any
    memory slots the agent has learned for this (user_id, tenant_id) in
    prior sessions. Both writes land as assistant messages so the LLM
    sees them before the first reply.
    """
    db = await get_db()
    now = _now()
    user = await db.users.find_one_and_update(
        {"user_id": user_id},
        {
            "$set": {"last_seen_at": now},
            "$setOnInsert": {"user_id": user_id, "created_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    chat_ctx = ChatContext()
    name = user.get("name")
    email = user.get("email")
    prefs = user.get("preferences", {})
    if name or email or prefs:
        chat_ctx.add_message(
            role="assistant",
            content=(
                f"User profile: name={name or 'unknown'}, "
                f"email={email or 'unknown'}, preferences={prefs}."
            ),
        )
    else:
        chat_ctx.add_message(
            role="assistant",
            content=(
                f"No stored profile fields yet for user_id {user_id}. "
                "Greet them as a new user, then ask for their name and "
                "call update_profile with field='name' so it persists."
            ),
        )
    if not name:
        chat_ctx.add_message(
            role="assistant",
            content=(
                "No name on file for this user. Ask them for their name "
                "and call update_profile with field='name' to save it."
            ),
        )

    memories = await list_memories(db, user_id, tenant_id)
    if memories:
        lines = "\n".join(
            f"- {m['memory_type']}: {m['content']}" for m in memories
        )
        chat_ctx.add_message(
            role="assistant",
            content=f"Remembered facts from prior sessions:\n{lines}",
        )
    return chat_ctx


async def on_session_end(ctx: JobContext) -> None:
    """Pattern 5: persist a session report to MongoDB on hangup."""
    try:
        report = ctx.make_session_report()
        db = await get_db()
        user_id = ctx.proc.userdata.get("user_id", DEFAULT_USER_ID)
        tenant_id = ctx.proc.userdata.get("tenant_id", DEFAULT_TENANT_ID)
        await db.sessions.insert_one(
            {
                "session_id": ctx.room.name,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "room_name": ctx.room.name,
                "report": report.to_dict(),
            }
        )
        logger.info("Persisted session report for %s", ctx.room.name)
    except Exception:
        logger.exception("Failed to persist session report")
    finally:
        await aclose()


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent", on_session_end=on_session_end)
async def my_agent(ctx: JobContext) -> None:
    print_banner()
    ctx.log_context_fields = {"room": ctx.room.name}

    # Pattern 3 setup: identify the user from agent dispatch metadata.
    meta: dict[str, str] = {}
    if hasattr(ctx, 'job') and getattr(ctx, 'job', None) and ctx.job.metadata:
        try:
            meta = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            logger.warning("ctx.job.metadata was not valid JSON; using defaults")

    user_id = meta.get("user_id", DEFAULT_USER_ID)
    tenant_id = meta.get("tenant_id", DEFAULT_TENANT_ID)
    ctx.proc.userdata["user_id"] = user_id
    ctx.proc.userdata["tenant_id"] = tenant_id

    await ctx.connect()

    initial_ctx = await preload_user(user_id, tenant_id)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-5.3-chat-latest"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            preemptive_generation={"enabled": True},
        ),
    )

    # Initialize the agent instance early so we can use its state
    mongo_agent = MongoAgent(
        chat_ctx=initial_ctx,
        user_id=user_id,
        tenant_id=tenant_id,
        livekit_llm=session.llm,
    )
    # Stash the agent instance on the session for event listeners to find
    setattr(session, '_agent', mongo_agent)

    # Get DB instance here to pass to resume claims function
    db_instance = await get_db()

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev: UserInputTranscribedEvent):
        if not ev.transcript:
            return

        import time
        now = time.time()
        
        # 1. Voice-Set Identity (Denoising)
        identity = ev.speaker_id
        if mongo_agent._primary_speaker_id is None and identity:
            mongo_agent._primary_speaker_id = identity
            logger.info(f"Primary speaker identified: {identity}")
        
        if identity and identity != mongo_agent._primary_speaker_id:
            return

        # 2. State & Cooldown
        if mongo_agent._is_speaking_correction: return
        if (now - mongo_agent._last_spoke_at) < 4.0: return

        # 3. Hybrid Strategy
        if not ev.is_final:
            # FAST-PATH ONLY for speed
            asyncio.create_task(automated_fact_check(session, db_instance, ev.transcript, session.llm, mongo_agent, fast_path_only=True))
        else:
            # FULL-CHECK for precision
            asyncio.create_task(automated_fact_check(session, db_instance, ev.transcript, session.llm, mongo_agent, fast_path_only=False))

    # Check and resume any incomplete claims from previous runs in the background
    asyncio.create_task(check_and_resume_incomplete_claims(db_instance, session.llm, mongo_agent))

    await session.start(
        agent=mongo_agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L
                ),
            ),
        ),
    )


if __name__ == "__main__":
    print_banner()
    cli.run_app(server)
