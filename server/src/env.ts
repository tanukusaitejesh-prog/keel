// Load the repository's .env (if present) for the API, worker and seed entry points.
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const file = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", ".env");
if (existsSync(file)) process.loadEnvFile(file);
