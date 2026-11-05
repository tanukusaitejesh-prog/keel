import { AgentRegistry } from "../core/define.js";
import { analyst } from "./analyst.js";
import { researcher, wikiResearcher } from "./research.js";

export const registry = () => new AgentRegistry([analyst, researcher, wikiResearcher]);
