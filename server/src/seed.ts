import "./env.js";
import { createPool, migrate } from "./db.js";
import { seedShop } from "./agents/shop-data.js";

const db = createPool();
await migrate(db);
await seedShop(db);
const { rows } = await db.query(`SELECT (SELECT count(*) FROM shop.customers) c, (SELECT count(*) FROM shop.products) p, (SELECT count(*) FROM shop.orders) o, (SELECT count(*) FROM shop.order_items) i`);
console.log("seeded shop:", rows[0]);
await db.end();
