import { tool } from "@langchain/core/tools";
import { z } from "zod";

/**
 * Arithmetic without eval(): + - * / % ^, parentheses, unary minus, decimals.
 * Models are unreliable at arithmetic, so agents are told to use this instead.
 */
export function evaluate(expr: string): number {
  const src = expr.replace(/,/g, "").replace(/\s+/g, "");
  let i = 0;
  const peek = () => src[i];
  const fail = (msg: string): never => {
    throw new Error(`${msg} at position ${i} in "${expr}"`);
  };
  const number = (): number => {
    const m = /^\d+(\.\d+)?(e[+-]?\d+)?|^\.\d+/i.exec(src.slice(i));
    if (!m) return fail("expected a number");
    i += m[0].length;
    return Number(m[0]);
  };
  const atom = (): number => {
    if (peek() === "(") {
      i++;
      const v = sum();
      if (peek() !== ")") fail("expected )");
      i++;
      return v;
    }
    if (peek() === "-") {
      i++;
      return -power();
    }
    if (peek() === "+") {
      i++;
      return power();
    }
    return number();
  };
  const power = (): number => {
    const base = atom();
    if (peek() === "^") {
      i++;
      return base ** power(); // right-associative
    }
    return base;
  };
  const product = (): number => {
    let v = power();
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const op = src[i++];
      const r = power();
      if ((op === "/" || op === "%") && r === 0) fail("division by zero");
      v = op === "*" ? v * r : op === "/" ? v / r : v % r;
    }
    return v;
  };
  const sum = (): number => {
    let v = product();
    while (peek() === "+" || peek() === "-") v = src[i++] === "+" ? v + product() : v - product();
    return v;
  };
  if (!src) fail("empty expression");
  const v = sum();
  if (i < src.length) fail(`unexpected "${src[i]}"`);
  if (!Number.isFinite(v)) fail("result is not finite");
  return v;
}

/** A LangChain tool (usable in any LangChain/LangGraph app, and in Keel via fromLangChainTool). */
export const calculator = tool(
  async ({ expression }) => {
    const v = evaluate(expression);
    return `${expression} = ${Number.isInteger(v) ? v : Number(v.toPrecision(12))}`;
  },
  {
    name: "calculate",
    description: "Evaluate an arithmetic expression exactly: + - * / % ^ and parentheses. Use it for every calculation instead of doing math yourself.",
    schema: z.object({ expression: z.string().min(1).max(500).describe("e.g. (1250 - 980) / 980 * 100") }),
  },
);
