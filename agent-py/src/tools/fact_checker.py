import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Literal, Optional, Any

from livekit.agents import ChatContext
from livekit.agents.inference import LLM, LLMStream
from pymongo.asynchronous.database import AsyncDatabase
from tavily import TavilyClient

import simpleaudio as sa
import os
from elevenlabs.client import ElevenLabs
import elevenlabs.play as el_play

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_KEY"))

logger = logging.getLogger("fact_checker")

# Initialize Tavily client (API key will be loaded from env)
print(f"DEBUG: TAVILY_API_KEY from os.getenv: {os.getenv('TAVILY_API_KEY')}")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Assuming LiveKit LLM for reasoning and query reformulation
# The actual LLM instance needs to be passed or accessed via some context
# For now, we'll create a placeholder LLM instance here.
# In a real scenario, this would come from the agent's session.
livekit_llm_instance: Optional[LLM] = None

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)

def set_livekit_llm_instance(llm: LLM):
    global livekit_llm_instance
    livekit_llm_instance = llm

async def _get_llm_response(prompt: str) -> str:
    if not livekit_llm_instance:
        raise ValueError("LiveKit LLM instance not set for fact_checker")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content=prompt)
    stream = livekit_llm_instance.chat(chat_ctx=chat_ctx)
    full_response = ""
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            full_response += chunk.delta.content
    return full_response.strip()

async def _calculate_confidence(
    db: AsyncDatabase, search_results: list[dict]
) -> float:
    # Simplified confidence calculation for hackathon.
    # A more sophisticated approach would involve LLM analysis of results.
    if not search_results:
        return 0.0

    total_score = 0.0
    for result in search_results:
        domain = result.get("url", "").split("/")[2] if result.get("url") else ""
        if domain:
            source_score_doc = await db.source_scores.find_one({"domain": domain})
            if source_score_doc:
                total_score += source_score_doc.get("score", 0.5) # Default to 0.5 if no score
            else:
                total_score += 0.5 # Default neutral score for unknown domains
    return min(1.0, total_score / len(search_results)) # Average score, capped at 1.0


async def _reformulate_query(
    original_claim: str, low_confidence_reason: str, previous_query: str
) -> str:
    # Today is May 2, 2026 per session context
    prompt = (
        f"Today's date is Saturday, May 2, 2026.\n"
        f"The original factual claim was: '{original_claim}'.\n"
        f"Previous search query was: '{previous_query}'.\n"
        f"The previous search yielded low confidence results because: {low_confidence_reason}.\n"
        "Generate a NEW, better search query to verify the claim as of TODAY (May 2026). "
        "Focus on finding current real-time data from 2025-2026. "
        "Return ONLY the reformulated query string."
    )
    return await _get_llm_response(prompt)


async def process_claim_for_facts(
    db: AsyncDatabase, claim_id: str, livekit_llm: LLM, mongo_agent: Optional[Any] = None
) -> None:
    set_livekit_llm_instance(livekit_llm) # Set the global LLM instance

    claim = await db.claims.find_one({"_id": claim_id})
    if not claim:
        logger.error(f"Claim with ID {claim_id} not found.")
        return

    logger.info(f"Processing claim: {claim['text']}")

    original_claim_text = claim["text"]

    # --- Memory Pass: Check if we already verified this claim ---
    try:
        # Simple text match for exact repeats (can be upgraded to vector search)
        existing_claim = await db.claims.find_one({
            "text": original_claim_text,
            "status": "completed",
            "verdict": {"$in": ["True", "False"]}
        })

        if existing_claim:
            logger.info(f"Memory Hit! Using existing verdict for claim {claim_id}")

            # Check if we have cached audio
            cached_audio = existing_claim.get("audio_bytes")

            await db.claims.update_one(
                {"_id": claim_id},
                {"$set": {
                    "verdict": existing_claim["verdict"],
                    "correction": existing_claim.get("correction"),
                    "audio_bytes": cached_audio, # Copy cached audio to new record
                    "status": "speaking" if existing_claim["verdict"] == "False" else "completed",
                    "last_updated": _now(),
                }}
            )

            if existing_claim["verdict"] == "False" and existing_claim.get("correction"):
                if cached_audio:
                    logger.info(f"Instant Audio Recall! Playing cached bytes for {claim_id}")
                    if mongo_agent: mongo_agent._is_speaking_correction = True
                    try:
                        el_play.play(cached_audio)
                    finally:
                        if mongo_agent: mongo_agent._is_speaking_correction = False
                    await db.claims.update_one({"_id": claim_id}, {"$set": {"status": "completed"}})
                else:
                    asyncio.create_task(speak_correction(db, claim_id, existing_claim["correction"], mongo_agent))
            return
    except Exception as mem_err:
        logger.error(f"Memory pass failed for claim {claim_id}: {mem_err}")
    current_query = original_claim_text
    retrieval_log_ids = []

    # --- Pass 1: Hybrid Retrieval (Race to Truth) ---
    logger.info(f"Retrieval Pass 1 for claim {claim_id}: {current_query}")
    await db.claims.update_one(
        {"_id": claim_id}, {"$set": {"status": "retrieving", "last_updated": _now()}}
    )
    
    raw_results_pass1 = []
    local_hit = False
    
    # A. Search Local MongoDB Knowledge Base (The Fast Path)
    try:
        from tools.embeddings import embed_text
        query_embedding = await embed_text(current_query)
        cursor = await db.knowledge.aggregate([
            {
                "$vectorSearch": {
                    "index": "knowledge_embedding_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 10,
                    "limit": 3,
                }
            },
            {"$project": {"_id": 0, "content": 1, "title": 1, "static_correction": 1, "audio_bytes": 1, "score": {"$meta": "vectorSearchScore"}}}
        ])
        local_docs = await cursor.to_list(length=3)
        
        local_hit = False
        for doc in local_docs:
            if doc["score"] > 0.90: # RAISED from 0.4 for absolute precision
                static_correction = doc.get("static_correction")
                audio_bytes = doc.get("audio_bytes")
                raw_results_pass1.append({
                    "url": "local://mongodb/ground_truth",
                    "title": doc.get("title", "Internal Ground Truth"),
                    "content": f"GROUND TRUTH (MUST OBEY): {doc['content']}",
                    "static_correction": static_correction,
                    "audio_bytes": audio_bytes # Carry audio through
                })
                logger.info(f"Local Knowledge Hit (score {doc['score']}): {doc.get('title')}")
                if static_correction:
                    logger.info(f"Super-Nitro: Found static correction: {static_correction}")
                local_hit = True # Ensure this stays true if any doc matches
    except Exception as e:
        logger.error(f"Local knowledge search failed: {e}")

    # B. Search External Web (Only if no high-confidence local hit)
    if not local_hit:
        logger.info("No local hit; performing Tavily search.")
        try:
            tavily_results_pass1 = tavily_client.search(query=current_query, search_depth="advanced")
            raw_results_pass1.extend(tavily_results_pass1.get("results", []))
        except Exception as e:
            logger.error(f"Tavily search failed for claim {claim_id} (Pass 1): {e}")
    else:
        logger.info("Skipping Tavily search due to MongoDB Ground Truth hit.")

    confidence_pass1 = await _calculate_confidence(db, raw_results_pass1)
    logger.info(f"Confidence after Pass 1: {confidence_pass1}")

    log_doc_pass1 = {
        "claim_id": claim_id,
        "attempt_number": 1,
        "timestamp_attempted": _now(),
        "original_query": original_claim_text,
        "reformulated_query": None,
        "raw_retrieval_results": raw_results_pass1,
        "source_urls": [r.get("url") for r in raw_results_pass1 if r.get("url")], # EXTRACT URLS
        "extracted_source_details": [], # Populate this in a more detailed implementation
        "confidence_after_pass": confidence_pass1,
        "adaptation_decision": "none",
        "adaptation_reason": None,
        "final_confidence_for_attempt": confidence_pass1,
    }
    log_result_pass1 = await db.retrieval_log.insert_one(log_doc_pass1)
    retrieval_log_ids.append(log_result_pass1.inserted_id)

    final_confidence = confidence_pass1
    final_raw_results = raw_results_pass1
    
    # Adaptation: If confidence is low, attempt Pass 2
    if confidence_pass1 < 0.65:
        logger.info(f"Confidence below threshold for claim {claim_id}. Attempting Pass 2.")
        low_confidence_reason = "Initial search yielded low confidence."
        current_query = await _reformulate_query(original_claim_text, low_confidence_reason, current_query)
        logger.info(f"Reformulated query for Pass 2: {current_query}")

        try:
            tavily_results_pass2 = tavily_client.search(query=current_query, search_depth="basic")
            raw_results_pass2 = tavily_results_pass2.get("results", [])
        except Exception as e:
            logger.error(f"Tavily search failed for claim {claim_id} (Pass 2): {e}")
            raw_results_pass2 = []

        confidence_pass2 = await _calculate_confidence(db, raw_results_pass2)
        logger.info(f"Confidence after Pass 2: {confidence_pass2}")

        log_doc_pass2 = {
            "claim_id": claim_id,
            "attempt_number": 2,
            "timestamp_attempted": _now(),
            "original_query": original_claim_text,
            "reformulated_query": current_query,
            "raw_retrieval_results": raw_results_pass2,
            "source_urls": [r.get("url") for r in raw_results_pass2 if r.get("url")], # EXTRACT URLS
            "extracted_source_details": [], # Populate this in a more detailed implementation
            "confidence_after_pass": confidence_pass2,
            "adaptation_decision": "reformulate_query",
            "adaptation_reason": low_confidence_reason,
            "final_confidence_for_attempt": confidence_pass2,
        }
        log_result_pass2 = await db.retrieval_log.insert_one(log_doc_pass2)
        retrieval_log_ids.append(log_result_pass2.inserted_id)
        
        # Take the better results/confidence from either pass
        if confidence_pass2 > confidence_pass1:
            final_confidence = confidence_pass2
            final_raw_results = raw_results_pass2
            
    # Update claim with final results
    await db.claims.update_one(
        {"_id": claim_id},
        {
            "$set": {
                "status": "reasoning", # Next step
                "confidence_score": final_confidence,
                "retrieval_log_ids": retrieval_log_ids,
                "source_domains_consulted": list(set([
                    result.get("url", "").split("/")[2]
                    for r_list in [raw_results_pass1, raw_results_pass2] for result in r_list if result.get("url")
                ])),
                "last_updated": _now(),
            }
        },
    )
    logger.info(f"Claim {claim_id} retrieval and adaptation complete. Final confidence: {final_confidence}")

    # SUPER-NITRO: Check for static correction to skip reasoning
    for res in final_raw_results:
        if res.get("static_correction"):
            correction_text = res["static_correction"]
            cached_audio = res.get("audio_bytes")
            
            logger.info(f"Super-Nitro: Bypassing reasoning for static correction: {correction_text}")
            
            # Update claim record
            await db.claims.update_one(
                {"_id": claim_id},
                {"$set": {
                    "verdict": "False",
                    "correction": correction_text,
                    "audio_bytes": cached_audio,
                    "status": "speaking",
                    "last_updated": _now(),
                }}
            )
            
            # If we have pre-cached audio, play it INSTANTLY
            if cached_audio:
                logger.info(f"SUPER-NITRO-AUDIO: Instant playback of pre-cached bytes for {claim_id}")
                if mongo_agent: mongo_agent._is_speaking_correction = True
                try:
                    el_play.play(cached_audio)
                finally:
                    import time
                    if mongo_agent:
                        mongo_agent._is_speaking_correction = False
                        mongo_agent._last_spoke_at = time.time()
                await db.claims.update_one({"_id": claim_id}, {"$set": {"status": "completed"}})
                return

            # Fallback to normal speech if no cached bytes
            asyncio.create_task(speak_correction(db, claim_id, correction_text, mongo_agent))
            return

    # Trigger reasoning on the claim
    # Nitro: If it's a local hit, use a faster reasoning path
    reasoning_llm = livekit_llm
    if local_hit:
        logger.info("Nitro Path: Using fast-track reasoning for local hit.")
        # In LiveKit, we can try to use a faster model alias if available, 
        # but for now we'll just prioritize the prompt to be simpler.
    
    asyncio.create_task(reason_on_claim(db, claim_id, reasoning_llm, mongo_agent))


async def reason_on_claim(
    db: AsyncDatabase, claim_id: str, livekit_llm: LLM, mongo_agent: Optional[Any] = None
) -> None:
    set_livekit_llm_instance(livekit_llm) # Ensure LLM instance is set

    claim = await db.claims.find_one({"_id": claim_id})
    if not claim:
        logger.error(f"Claim with ID {claim_id} not found for reasoning.")
        return

    logger.info(f"Reasoning on claim: {claim['text']}")

    # Fetch retrieval log entries for this claim
    retrieval_logs = await db.retrieval_log.find({"claim_id": claim_id}).to_list(length=None)

    # Compile a summary of retrieval results for the LLM
    results_summary = []
    for log in retrieval_logs:
        results_summary.append(f"Attempt {log['attempt_number']} (Query: '{log.get('reformulated_query') or log.get('original_query')}'):")
        for res in log.get("raw_retrieval_results", []):
            results_summary.append(f"- Source: {res.get('url', 'N/A')}, Content: {res.get('content', 'N/A')}")
    
    context_for_llm = (
        f"Today's date is Saturday, May 2, 2026.\n"
        f"Original factual claim: '{claim['text']}'\n\n"
        "Here are the search results obtained from the live web and internal DB:\n"
        + "\n".join(results_summary) + "\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. If a result is labeled 'GROUND TRUTH (MUST OBEY)', its information is ABSOLUTE. It overrides all other sources and your internal knowledge.\n"
        "2. If no Ground Truth is present, prioritize 2025-2026 data.\n"
        "3. Provide an ULTRA-CONCISE correction (3-5 words) if the claim is False.\n"
        "Return your response in a JSON object with 'verdict' and 'correction' fields."
    )

    llm_raw_response = await _get_llm_response(context_for_llm)
    
    verdict_data = {"verdict": "Unverifiable", "correction": None}
    try:
        # Robust cleanup for common LLM parsing issues
        cleaned_response = llm_raw_response.strip()
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[1].split("```")[0].strip()
        
        # If it looks like a Python dict (uses single quotes), convert to JSON-compliant double quotes
        if "'" in cleaned_response and '"' not in cleaned_response:
            cleaned_response = cleaned_response.replace("'", '"')

        verdict_data = json.loads(cleaned_response)
        if not isinstance(verdict_data, dict) or "verdict" not in verdict_data:
            raise ValueError("Invalid LLM response structure")
    except Exception as e:
        logger.error(f"Failed to parse LLM reasoning response for claim {claim_id}: {e}\nRaw response: {llm_raw_response}")

    verdict = verdict_data.get("verdict", "Unverifiable")
    correction = verdict_data.get("correction")

    # Update claim status to "speaking" (next step) or "completed" if no speaking needed
    new_status = "speaking" if verdict == "False" else "completed"

    await db.claims.update_one(
        {"_id": claim_id},
        {"$set": {
            "verdict": verdict,
            "correction": correction,
            "status": new_status,
            "last_updated": _now(),
        }},
    )
    logger.info(f"Claim {claim_id} reasoning complete. Verdict: {verdict}, Correction: {correction}")

    # If a correction is needed, trigger the speaking phase (case-insensitive check)
    if verdict and verdict.strip().lower() == "false" and correction:
        asyncio.create_task(speak_correction(db, claim_id, correction, mongo_agent))
    else:
        logger.info(f"No correction needed for claim {claim_id} (Verdict: {verdict})")


async def speak_correction(db: AsyncDatabase, claim_id: str, correction_text: str, mongo_agent: Optional[Any] = None) -> None:
    print(f"\n[🎙️ SPEAKING CORRECTION] {correction_text}")
    
    try:
        from agent import control_media
        logger.info(f"Attempting to stream audio for: {correction_text}")
        control_media("pause")
        # Convert text to speech (returns a generator)
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB", # Adam or Bella ID
            text=correction_text,
            model_id="eleven_multilingual_v2"
        )
        
        # We need to play the stream AND capture it for the cache
        captured_chunks = []
        def streaming_wrapper():
            for chunk in audio_generator:
                captured_chunks.append(chunk)
                yield chunk

        # Play the audio using elevenlabs.play (it accepts an iterator!)
        if mongo_agent: mongo_agent._is_speaking_correction = True
        
        # Blacklist our own correction before we say it
        try:
            import time
            from agent import agent_corrections, normalize_text
            agent_corrections[normalize_text(correction_text)] = time.time()
            logger.info(f"Blacklisted our own correction: {correction_text}")
        except Exception:
            pass

        try:
            logger.info("Starting audio stream...")
            el_play.play(streaming_wrapper())
            logger.info("Audio stream finished.")
        except Exception as play_err:
            logger.error(f"el_play.play failed: {play_err}")
        finally:
            if mongo_agent: 
                import time
                mongo_agent._is_speaking_correction = False
                mongo_agent._last_spoke_at = time.time()
            control_media("play")

        # After playing, save the full captured buffer to MongoDB
        full_audio_bytes = b"".join(captured_chunks)
        await db.claims.update_one(
            {"_id": claim_id},
            {"$set": {
                "audio_bytes": full_audio_bytes, # CACHE THE FULL BUFFER
                "status": "completed",
                "last_updated": _now(),
            }},
        )
        logger.info(f"Correction cached and claim {claim_id} updated to 'completed'.")

    except Exception as e:
        logger.error(f"Failed to speak correction for claim {claim_id}: {e}")
        await db.claims.update_one(
            {"_id": claim_id},
            {"$set": {
                "status": "failed_speaking", # New status for error handling
                "last_updated": _now(),
            }},
        )

