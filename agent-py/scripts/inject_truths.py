import asyncio
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pymongo import MongoClient
from dotenv import load_dotenv
from tools.embeddings import embed_text
from elevenlabs.client import ElevenLabs

async def inject():
    # Load from the agent directory specifically
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.local")
    load_dotenv(env_path)
    
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "livekit_mongo_starter")
    el_key = os.getenv("ELEVEN_LABS_KEY")
    
    print(f"Connecting to MongoDB: {db_name}")
    client = MongoClient(uri)
    db = client[db_name]
    
    el_client = ElevenLabs(api_key=el_key)
    
    # --- CONFIGURE YOUR DEMO TRUTHS HERE ---
    FACTS = [
        {
            "claim": "The CEO of Apple is Steve Jobs", 
            "aliases": ["CEO of Apple is Steve Jobs", "Steve Jobs is Apple CEO", "Who is Apple CEO Steve Jobs"],
            "static_correction": "Tim Cook is CEO."
        },
        {
            "claim": "London population is 500 million", 
            "aliases": ["London has 500 million people", "Population of London is 500 million", "500 million people live in London"],
            "static_correction": "It is 9 million."
        },
        {
            "claim": "Water boils at 50 degrees", 
            "aliases": ["Water boils at fifty degrees", "Boiling point of water is 50", "Water is boiling at 50"],
            "static_correction": "Water boils at 100 degrees."
        },
        {
            "claim": "The earth is flat", 
            "aliases": ["Earth is flat", "The world is flat", "Is the earth flat", "The earth is a pancake"],
            "static_correction": "Earth is round."
        }
    ]

    def normalize(t): return "".join(filter(str.isalnum, t.lower()))

    print(f"Injecting {len(FACTS)} truths with {sum(len(f.get('aliases', [])) + 1 for f in FACTS)} total variations...")
    
    for f in FACTS:
        main_claim = f['claim']
        all_variations = [main_claim] + f.get("aliases", [])
        
        # 1. Pre-calculate main embedding (for vector fallback)
        embedding = await embed_text(main_claim)
        
        # 2. Pre-generate audio
        print(f" -> Processing: '{main_claim}'")
        audio_gen = el_client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            text=f['static_correction'],
            model_id="eleven_multilingual_v2"
        )
        audio_bytes = b"".join(audio_gen)
        
        # 3. Seed into KNOWLEDGE
        db.knowledge.update_one(
            {"title": f"Truth: {main_claim}"},
            {"$set": {
                "content": f"GROUND TRUTH: {main_claim}. The correct fact is: {f['static_correction']}",
                "static_correction": f['static_correction'],
                "audio_bytes": audio_bytes,
                "embedding": embedding,
                "category": "ground_truth"
            }},
            upsert=True
        )

        # 4. Seed all variations into CLAIMS (The 20ms Path)
        for variant in all_variations:
            db.claims.update_one(
                {"fast_key": normalize(variant)},
                {"$set": {
                    "text": variant,
                    "fast_key": normalize(variant),
                    "verdict": "False",
                    "correction": f['static_correction'],
                    "audio_bytes": audio_bytes,
                    "embedding": embedding,
                    "status": "completed"
                }},
                upsert=True
            )
    
    print("\n✅ Super-Nitro truths are live in Atlas.")
    client.close()

if __name__ == "__main__":
    asyncio.run(inject())
