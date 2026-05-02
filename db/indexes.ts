// Create MongoDB Atlas collections, indexes, and search indexes.
//
// Run once after configuring MONGODB_URI:
//
//     pnpm db:init
//
// Search indexes take ~1-2 minutes to become queryable on Atlas after
// creation.
//
// Requires MongoDB 8.0 or later because searchMemory uses the $rankFusion
// aggregation stage for hybrid vector + text retrieval. Atlas M10+ dedicated
// clusters run 8.0 by default; verify shared-tier clusters (M0/M2/M5) before
// running this script.

import { MongoClient } from 'mongodb';
import { loadEnv, requireEnv } from './lib/env';
import { EMBEDDING_DIMENSIONS } from './lib/voyage';

const DEFAULT_DB_NAME = 'livekit_mongo_starter';

interface SearchIndexSpec {
  name: string;
  type: 'vectorSearch' | 'search';
  definition: Record<string, unknown>;
}

function vectorIndex(name: string): SearchIndexSpec {
  return {
    name,
    type: 'vectorSearch',
    definition: {
      fields: [
        {
          type: 'vector',
          path: 'embedding',
          numDimensions: EMBEDDING_DIMENSIONS,
          similarity: 'cosine',
        },
        { type: 'filter', path: 'user_id' },
        { type: 'filter', path: 'tenant_id' },
      ],
    },
  };
}

// Full-text search index on `memory_type` + `content`. Paired with
// memories_embedding_index inside the $rankFusion pipeline in
// tools/memory.searchMemory. Token fields on user_id and tenant_id let
// the text branch $match per-user scope cheaply.
function memoriesTextIndex(): SearchIndexSpec {
  return {
    name: 'memories_text_index',
    type: 'search',
    definition: {
      mappings: {
        dynamic: false,
        fields: {
          memory_type: { type: 'string', analyzer: 'lucene.standard' },
          content: { type: 'string', analyzer: 'lucene.standard' },
          user_id: { type: 'token' },
          tenant_id: { type: 'token' },
        },
      },
    },
  };
}

function isAlreadyExistsError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  const msg = (err as { message?: string }).message?.toLowerCase() ?? '';
  return msg.includes('already exists') || msg.includes('duplicate');
}

async function createIndexes(): Promise<void> {
  const { source } = loadEnv();
  console.info(`Loaded env from ${source}`);

  const client = new MongoClient(requireEnv('MONGODB_URI'));
  await client.connect();
  try {
    const db = client.db(process.env.MONGODB_DB ?? DEFAULT_DB_NAME);

    const existing = new Set(
      (await db.listCollections({}, { nameOnly: true }).toArray()).map((c) => c.name),
    );
     for (const name of ['users', 'orders', 'sessions', 'knowledge', 'memories', 'claims', 'source_scores', 'retrieval_log']) {
       if (!existing.has(name)) {
         await db.createCollection(name);
         console.info(`${name}: collection created`);
       }
     }

     // New A.T.E. collections and their indexes
     await db.collection('claims').createIndex({ status: 1 });
     await db.collection('claims').createIndex({ timestamp_ingested: 1 });
     await db.collection('claims').createIndex({ text: 1 });
     console.info('claims: status, timestamp_ingested, text indexes ready');

     await db.collection('source_scores').createIndex({ domain: 1 }, { unique: true });
     console.info('source_scores: domain unique index ready');

     await db.collection('retrieval_log').createIndex({ claim_id: 1 });
     await db.collection('retrieval_log').createIndex({ timestamp_attempted: 1 });
     console.info('retrieval_log: claim_id, timestamp_attempted indexes ready');

     await db.collection('users').createIndex({ user_id: 1 }, { unique: true });
     console.info('users: user_id unique index ready');

    await db.collection('orders').createIndex({ order_id: 1 }, { unique: true });
    await db.collection('orders').createIndex({ user_id: 1 });
    console.info('orders: order_id unique + user_id indexes ready');

    await db.collection('sessions').createIndex({ session_id: 1 }, { unique: true });
    await db.collection('sessions').createIndex({ user_id: 1 });
    console.info('sessions: session_id unique + user_id indexes ready');

    await db
      .collection('memories')
      .createIndex(
        { user_id: 1, tenant_id: 1, memory_type: 1 },
        { unique: true, name: 'memories_slot_unique' },
      );
    console.info('memories: (user_id, tenant_id, memory_type) unique index ready');

    const searchIndexes: ReadonlyArray<{ collection: string; spec: SearchIndexSpec }> = [
      { collection: 'knowledge', spec: vectorIndex('knowledge_embedding_index') },
      { collection: 'memories', spec: vectorIndex('memories_embedding_index') },
      { collection: 'memories', spec: memoriesTextIndex() },
      { collection: 'claims', spec: vectorIndex('claims_embedding_index') },
    ];
    for (const { collection, spec } of searchIndexes) {
      try {
        await db.collection(collection).createSearchIndex(spec);
        console.info(`${collection}: ${spec.name} search index created`);
      } catch (err) {
        if (isAlreadyExistsError(err)) {
          console.info(`${collection}: ${spec.name} already exists`);
        } else {
          throw err;
        }
      }
    }

    console.info('Done. Search indexes need ~1-2 minutes to sync on Atlas.');
  } finally {
    await client.close();
  }
}

createIndexes().catch((err) => {
  console.error(err);
  process.exit(1);
});
