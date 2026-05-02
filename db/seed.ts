// Seed MongoDB with sample users, orders, and knowledge documents.
//
// Run after db/indexes.ts:
//
//     pnpm db:seed
//
// Sample data lives in db/lib/data.ts. Edit there to customize.

import { MongoClient } from 'mongodb';
import { loadEnv, requireEnv } from './lib/env';
import { KNOWLEDGE_INPUTS, ORDERS, USERS } from './lib/data';
import { embedTexts } from './lib/voyage';

const DEFAULT_DB_NAME = 'livekit_mongo_starter';

async function seedData(): Promise<void> {
  const { source } = loadEnv();
  console.info(`Loaded env from ${source}`);

  const client = new MongoClient(requireEnv('MONGODB_URI'));
  await client.connect();
  try {
    const db = client.db(process.env.MONGODB_DB ?? DEFAULT_DB_NAME);
    const now = new Date(); // Moved declaration here

    for (const name of ['users', 'orders', 'knowledge', 'memories', 'sessions', 'claims', 'source_scores', 'retrieval_log']) {
      await db.collection(name).deleteMany({});
    }
    console.info('Cleared existing data from all collections');

    // Seed initial source_scores
    const initialSourceScores = [
      { domain: "wikipedia.org", score: 0.8, evaluations_count: 0, positive_contributions: 0, negative_contributions: 0, created_at: now, last_evaluated: now },
      { domain: "nytimes.com", score: 0.7, evaluations_count: 0, positive_contributions: 0, negative_contributions: 0, created_at: now, last_evaluated: now },
      { domain: "example.com", score: 0.3, evaluations_count: 0, positive_contributions: 0, negative_contributions: 0, created_at: now, last_evaluated: now },
    ];
    await db.collection('source_scores').insertMany(initialSourceScores);
    console.info(`Inserted ${initialSourceScores.length} initial source scores`);

    const users = USERS.map((u) => ({ ...u, created_at: now }));
    await db.collection('users').insertMany(users);
    console.info(`Inserted ${users.length} users`);

    const orders = ORDERS.map((o) => ({ ...o, created_at: now }));
    await db.collection('orders').insertMany(orders);
    console.info(`Inserted ${orders.length} orders`);

    const embeddings = await embedTexts(
      KNOWLEDGE_INPUTS.map((doc) => doc.content),
      { inputType: 'document' },
    );
    const knowledgeDocs = KNOWLEDGE_INPUTS.map((doc, i) => ({
      ...doc,
      embedding: embeddings[i],
      created_at: now,
    }));
    await db.collection('knowledge').insertMany(knowledgeDocs);
    console.info(
      `Inserted ${knowledgeDocs.length} knowledge documents with embeddings`,
    );

    console.info('Seed complete.');
  } finally {
    await client.close();
  }
}

seedData().catch((err) => {
  console.error(err);
  process.exit(1);
});
