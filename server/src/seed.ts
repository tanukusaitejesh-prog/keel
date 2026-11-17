import "./env.js";
import { createPool, migrate } from "./db.js";
import { seedShop } from "./agents/shop-data.js";
import { docsStore, ingestRepoDocs } from "./agents/docs.js";

const db = createPool();
await migrate(db);
await seedShop(db);
const { rows } = await db.query(`SELECT (SELECT count(*) FROM shop.customers) c, (SELECT count(*) FROM shop.products) p, (SELECT count(*) FROM shop.orders) o, (SELECT count(*) FROM shop.order_items) i`);
console.log("seeded shop:", rows[0]);
const chunks = await ingestRepoDocs(db);
console.log(`ingested README.md into the docs knowledge base: ${chunks} chunks (${docsStore(db).backend})`);
await db.end();
