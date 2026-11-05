import type { Db } from "../db.js";

export const analystPassword = () => process.env.ANALYST_DB_PASSWORD ?? "keel_analyst_local";

/**
 * Deterministic synthetic e-commerce dataset in schema `shop`, plus a
 * SELECT-only role the analyst agent's queries run as. No real people.
 */
const FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vihaan", "Isha", "Arjun", "Saanvi", "Liam", "Emma", "Noah", "Olivia", "Lucas", "Mia", "Yuki", "Hana", "Mateo", "Sofia", "Omar", "Layla", "Chen", "Wei", "Amara", "Kofi"];
const LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Gupta", "Menon", "Smith", "Brown", "Garcia", "Muller", "Tanaka", "Sato", "Silva", "Rossi", "Haddad", "Okafor", "Li", "Zhang", "Kim", "Dubois"];
const COUNTRIES: [string, number][] = [["India", 34], ["United States", 22], ["United Kingdom", 11], ["Germany", 9], ["Japan", 8], ["Brazil", 7], ["France", 5], ["Nigeria", 4]];
const CATEGORIES: [string, [number, number]][] = [["Electronics", [80, 900]], ["Home & Kitchen", [15, 220]], ["Books", [8, 45]], ["Sports", [12, 300]], ["Beauty", [6, 80]], ["Toys", [10, 120]]];
const ADJ = ["Pro", "Classic", "Smart", "Eco", "Ultra", "Mini", "Deluxe", "Prime"];
const NOUN: Record<string, string[]> = {
  Electronics: ["Headphones", "Monitor", "Keyboard", "Speaker", "Tablet", "Camera", "Router"],
  "Home & Kitchen": ["Blender", "Kettle", "Pan Set", "Lamp", "Knife Set", "Toaster"],
  Books: ["Cookbook", "Novel", "Atlas", "Biography", "Workbook"],
  Sports: ["Yoga Mat", "Dumbbells", "Racket", "Bike Helmet", "Running Shoes"],
  Beauty: ["Face Serum", "Shampoo", "Sunscreen", "Lip Balm"],
  Toys: ["Puzzle", "Robot Kit", "Board Game", "Plush Bear"],
};

function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

export async function seedShop(db: Db): Promise<void> {
  const r = rng(2026);
  const pick = <T>(a: T[]) => a[Math.floor(r() * a.length)]!;
  const weighted = <T>(a: [T, number][]) => {
    const total = a.reduce((s, [, w]) => s + w, 0);
    let x = r() * total;
    for (const [v, w] of a) if ((x -= w) < 0) return v;
    return a[0]![0];
  };
  const day = (start: Date, end: Date) => new Date(start.getTime() + Math.floor(r() * ((end.getTime() - start.getTime()) / 864e5)) * 864e5).toISOString().slice(0, 10);

  const client = await db.connect();
  try {
    await client.query("BEGIN");
    await client.query(`DROP SCHEMA IF EXISTS shop CASCADE; CREATE SCHEMA shop;`);
    await client.query(`
      CREATE TABLE shop.customers (id int PRIMARY KEY, name text NOT NULL, country text NOT NULL, segment text NOT NULL, signup_date date NOT NULL);
      CREATE TABLE shop.products (id int PRIMARY KEY, name text NOT NULL, category text NOT NULL, price numeric(10,2) NOT NULL);
      CREATE TABLE shop.orders (id int PRIMARY KEY, customer_id int NOT NULL REFERENCES shop.customers(id), order_date date NOT NULL, status text NOT NULL, channel text NOT NULL);
      CREATE TABLE shop.order_items (order_id int NOT NULL REFERENCES shop.orders(id), product_id int NOT NULL REFERENCES shop.products(id), quantity int NOT NULL, unit_price numeric(10,2) NOT NULL, PRIMARY KEY (order_id, product_id));
      COMMENT ON TABLE shop.customers IS 'One row per customer account.';
      COMMENT ON COLUMN shop.customers.segment IS 'consumer or business';
      COMMENT ON TABLE shop.products IS 'Catalog. price is the current list price.';
      COMMENT ON TABLE shop.orders IS 'One row per order. Only status = completed counts toward revenue; returned and cancelled orders do not.';
      COMMENT ON COLUMN shop.orders.status IS 'completed, returned or cancelled';
      COMMENT ON COLUMN shop.orders.channel IS 'web, mobile or store';
      COMMENT ON TABLE shop.order_items IS 'Line items. Revenue of a line = quantity * unit_price (the price actually charged).';
    `);

    const customers: string[] = [];
    for (let i = 1; i <= 400; i++) {
      const name = `${pick(FIRST)} ${pick(LAST)}`;
      customers.push(`(${i}, '${name}', '${weighted(COUNTRIES)}', '${r() < 0.22 ? "business" : "consumer"}', '${day(new Date("2024-01-01"), new Date("2026-06-01"))}')`);
    }
    await client.query(`INSERT INTO shop.customers VALUES ${customers.join(",")}`);

    const products: { id: number; price: number; category: string }[] = [];
    const prodRows: string[] = [];
    let pid = 1;
    for (const [cat, [lo, hi]] of CATEGORIES) {
      const n = cat === "Electronics" ? 10 : 8;
      for (let k = 0; k < n; k++, pid++) {
        const price = Math.round((lo + r() * (hi - lo)) * 100) / 100;
        products.push({ id: pid, price, category: cat });
        prodRows.push(`(${pid}, '${pick(ADJ)} ${pick(NOUN[cat]!)} ${String.fromCharCode(65 + k)}', '${cat}', ${price})`);
      }
    }
    await client.query(`INSERT INTO shop.products VALUES ${prodRows.join(",")}`);

    // The last product of every category is never sold.
    const sellable = products.filter((p, i) => products[i + 1]?.category === p.category);
    const orders: string[] = [];
    const items: string[] = [];
    for (let o = 1; o <= 3200; o++) {
      const cust = 1 + Math.floor(Math.pow(r(), 1.6) * 400); // some customers buy much more
      const date = day(new Date("2025-01-01"), new Date("2026-07-01"));
      const month = Number(date.slice(5, 7));
      if ((month === 11 || month === 12) && r() < 0.15) continue; // keep holiday spike modest but present
      const status = r() < 0.07 ? "returned" : r() < 0.04 ? "cancelled" : "completed";
      orders.push(`(${o}, ${cust}, '${date}', '${status}', '${weighted([["web", 5], ["mobile", 4], ["store", 2]])}')`);
      const n = 1 + Math.floor(r() * r() * 4);
      const used = new Set<number>();
      for (let k = 0; k < n; k++) {
        const p = pick(sellable);
        if (used.has(p.id)) continue;
        used.add(p.id);
        const discount = r() < 0.2 ? 0.9 : 1;
        items.push(`(${o}, ${p.id}, ${1 + Math.floor(r() * r() * 3)}, ${(Math.round(p.price * discount * 100) / 100).toFixed(2)})`);
      }
    }
    await client.query(`INSERT INTO shop.orders VALUES ${orders.join(",")}`);
    await client.query(`INSERT INTO shop.order_items VALUES ${items.join(",")}`);

    // A separate LOGIN role: the analyst's sessions start unprivileged, so there
    // is no more powerful role they could SET ROLE back to.
    await client.query(`SELECT set_config('keel.analyst_pw', $1, true)`, [analystPassword()]);
    await client.query(`
      DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'keel_analyst') THEN
          EXECUTE format('CREATE ROLE keel_analyst LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD %L', current_setting('keel.analyst_pw'));
        ELSE
          EXECUTE format('ALTER ROLE keel_analyst LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD %L', current_setting('keel.analyst_pw'));
        END IF;
      END $$;
      REVOKE ALL ON SCHEMA public FROM keel_analyst;
      GRANT USAGE ON SCHEMA shop TO keel_analyst;
      GRANT SELECT ON ALL TABLES IN SCHEMA shop TO keel_analyst;
      ALTER ROLE keel_analyst SET search_path = shop;
      ALTER ROLE keel_analyst SET statement_timeout = '5s';
      ALTER ROLE keel_analyst SET timezone = 'UTC';
      ALTER ROLE keel_analyst SET default_transaction_read_only = on;
    `);
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}
