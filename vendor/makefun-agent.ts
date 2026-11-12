// makefun agent/v1 entry shim for Node 22: one file, erasable TypeScript, node:fs and node:crypto only,
// so it runs as-is under the grader's `node /bot/main.ts`. Copy it to vendor/makefun-agent.ts in your
// repository:
//
//   import { Anthropic, readTrial, StoreClient, trialRng } from './vendor/makefun-agent.ts';
//   const trial = readTrial();                 // the brief, the URLs, the shop token, the seed, the now
//   const store = new StoreClient(trial);      // the shop's Admin GraphQL and the supplier's REST API
//   const llm = new Anthropic(trial);          // the trial's gateway, the only route to a model
//   const { shop } = await store.request<{ shop: { name: string } }>('{ shop { name currencyCode } }');
//   const reply = await llm.messages({ model: 'claude-haiku-5-5', max_tokens: 1024, messages: [{ role: 'user', content: trial.prompt }] });
//
// What a trial hands the entry (readTrial reads it; TrialConfigError names what is missing):
//   MAKEFUN_PROMPT_FILE        the merchant brief, a UTF-8 text file: trial.prompt holds its text
//   MAKEFUN_SHOP_GRAPHQL_URL   the shop's Admin GraphQL endpoint (POST, X-Shopify-Access-Token)
//   MAKEFUN_SHOP_TOKEN         the shop's access token
//   MAKEFUN_SUPPLIER_URL       the supplier's REST API root; when unset, /supplier/v1 on the shop's origin
//   MAKEFUN_TRIAL_SEED         the trial's seed, to seed every random choice from (trialRng)
//   MAKEFUN_TRIAL_NOW          the trial's fixed now: an RFC 3339 instant, or epoch milliseconds (trialNow)
//   MAKEFUN_ENV_URL            the env's origin, whatever the env simulates; when unset, the shop's origin (envUrl)
//   ANTHROPIC_BASE_URL          the gateway for Anthropic Messages (default https://api.anthropic.com)
//   OPENAI_BASE_URL             the gateway for OpenAI Chat Completions (default https://api.openai.com/v1)
//   ANTHROPIC_API_KEY, OPENAI_API_KEY   the trial's key, makefun-trial-<48 hex>, the same on every pass:
//                                       the gateway admits that key alone, drops it and sends the real one
//
// The determinism rule. Every trial runs twice: live, where each model call goes through the gateway and
// is recorded, then replay, where the gateway answers each request from that recording by the request's
// content, and the store must end byte-equal or the run is nondeterministic and unranked. So nothing that
// differs between two runs may reach a request, to the model, the shop or the supplier:
//   - no wall clock: never put Date.now(), new Date(), performance.now() or process.hrtime() into a
//     request; take any date from trialNow(), the trial's fixed now;
//   - no unseeded randomness: never Math.random(), crypto.randomUUID() or randomBytes(); draw from
//     trialRng(trial, label), the same values in both runs (and in agent/sdk/makefun_agent.py);
//   - one order: a StoreClient sends one request at a time, in the order they were made, so two
//     mutations started together reach the shop in the same order on replay. Model calls may run in
//     parallel (the replay serves identical requests in their recorded order), but what you do with the
//     answers must not depend on which came back first.
// Waiting is fine (time may pass); a time value must never be written into anything sent. This file
// never reads a clock or a random source itself.
//
// The shop's rate limit is the Admin API's leaky bucket on the store's own clock: each Admin request is
// one simulated second, restoring 50 of 1,000 points, and waiting restores nothing. A THROTTLED answer is
// retried at once, each retry restoring the next 50, up to maxThrottleRetries; every request, a retry or a
// refusal too, counts against the trial's 5,000. The model clients retry 408, 429, 5xx and 529 answers
// with fixed backoff (never a refusal of the gateway's: budget_exhausted, request_cap, model_not_allowed,
// max_tokens_required, which throw LlmRefusalError) and never a connection that failed before an answer,
// since a request the gateway never saw has no recording to replay.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export const HARNESS = 'agent/v1';

/** The variables a trial sets for the entry. */
export const ENV = Object.freeze({
  PROMPT_FILE: 'MAKEFUN_PROMPT_FILE',
  SHOP_GRAPHQL_URL: 'MAKEFUN_SHOP_GRAPHQL_URL',
  SHOP_TOKEN: 'MAKEFUN_SHOP_TOKEN',
  SUPPLIER_URL: 'MAKEFUN_SUPPLIER_URL',
  TRIAL_SEED: 'MAKEFUN_TRIAL_SEED',
  TRIAL_NOW: 'MAKEFUN_TRIAL_NOW',
  ENV_URL: 'MAKEFUN_ENV_URL',
  ANTHROPIC_BASE_URL: 'ANTHROPIC_BASE_URL',
  OPENAI_BASE_URL: 'OPENAI_BASE_URL',
  ANTHROPIC_API_KEY: 'ANTHROPIC_API_KEY',
  OPENAI_API_KEY: 'OPENAI_API_KEY',
});

export const DEFAULT_ANTHROPIC_BASE_URL = 'https://api.anthropic.com';
export const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1';
export const ANTHROPIC_VERSION = '2023-06-01';
/** The supplier API's path on the shop's origin, when MAKEFUN_SUPPLIER_URL is unset. */
export const SUPPLIER_PATH = '/supplier/v1';
/** The gateway's own refusals (its error `type`): final, never retried. */
export const GATEWAY_REFUSALS = Object.freeze(['budget_exhausted', 'request_cap', 'model_not_allowed', 'max_tokens_required'] as const);
export type GatewayRefusal = (typeof GATEWAY_REFUSALS)[number];
/** The model answers retried with backoff. */
export const RETRY_STATUSES: readonly number[] = Object.freeze([408, 429, 500, 502, 503, 504, 529]);
/** The domain separating trialRng's streams from any other use of the seed. */
export const RNG_DOMAIN = 'agent-fun/agent/rng/v1';

// ---------------------------------------------------------------- the trial

/** The trial's environment is missing a variable or holds one this shim cannot read. */
export class TrialConfigError extends Error {
  override name = 'TrialConfigError';
}

export interface LlmEndpoint {
  /** The API root, no trailing slash (Anthropic: the host, /v1/messages is appended; OpenAI: .../v1). */
  baseUrl: string;
  /** The key sent as x-api-key or a Bearer token; '' sends none. In a trial, the trial's key, which the gateway swaps for the real one. */
  apiKey: string;
}

export interface Trial {
  /** The merchant brief: the prompt file's text (UTF-8, a leading BOM dropped). */
  prompt: string;
  promptFile: string;
  shopGraphqlUrl: string;
  shopToken: string;
  /** The supplier's REST API root, no trailing slash. */
  supplierUrl: string;
  /** The env's origin, no trailing slash: MAKEFUN_ENV_URL, or the shop's origin when unset. */
  envUrl: string;
  /** MAKEFUN_TRIAL_SEED as given. */
  seed: string;
  /** The trial's now, epoch milliseconds. */
  nowMs: number;
  /** The trial's now as `YYYY-MM-DDTHH:MM:SS.sssZ`. */
  nowIso: string;
  anthropic: LlmEndpoint;
  openai: LlmEndpoint;
}

type Env = Readonly<Record<string, string | undefined>>;

function required(env: Env, name: string): string {
  const v = env[name];
  if (v === undefined || v.trim() === '') throw new TrialConfigError(`${name} is not set: run the entry under the agent/v1 harness (or set it as RULES.txt describes)`);
  return v;
}

function httpUrl(name: string, text: string): string {
  let u: URL;
  try {
    u = new URL(text);
  } catch {
    throw new TrialConfigError(`${name} is not a URL: ${JSON.stringify(text)}`);
  }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new TrialConfigError(`${name} is not an http(s) URL: ${JSON.stringify(text)}`);
  return text;
}

const stripSlash = (u: string): string => u.replace(/\/+$/, '');

const RFC3339 = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/;
const DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
/** The latest instant a Date holds, in ms. */
const MAX_MS = 8_640_000_000_000_000;

/** A trial now (an RFC 3339 instant with Z or an offset, or epoch milliseconds) as epoch ms; fractions past ms are cut. */
export function parseTrialNow(text: string): number {
  if (/^\d{1,16}$/.test(text)) {
    const ms = Number(text);
    if (ms <= MAX_MS) return ms;
  }
  const m = RFC3339.exec(text);
  const bad = (): never => {
    throw new TrialConfigError(`${ENV.TRIAL_NOW} is not an RFC 3339 instant (2026-11-02T09:00:00Z) or epoch milliseconds: ${JSON.stringify(text)}`);
  };
  if (m === null) return bad();
  const [y, mo, d, h, mi, s] = [m[1], m[2], m[3], m[4], m[5], m[6]].map(Number) as [number, number, number, number, number, number];
  const leap = (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
  if (y < 1970 || mo < 1 || mo > 12 || d < 1 || d > (mo === 2 && leap ? 29 : DAYS[mo - 1]!) || h > 23 || mi > 59 || s > 59) return bad();
  const ms = Number((m[7] ?? '').padEnd(3, '0').slice(0, 3));
  let offsetMin = 0;
  if (m[8] !== 'Z') {
    const oh = Number(m[8]!.slice(1, 3));
    const om = Number(m[8]!.slice(4, 6));
    if (oh > 23 || om > 59) return bad();
    offsetMin = (m[8]!.startsWith('-') ? -1 : 1) * (oh * 60 + om);
  }
  const at = Date.UTC(y, mo - 1, d, h, mi, s, ms) - offsetMin * 60_000;
  return at < 0 ? bad() : at;
}

/** Everything the trial hands the entry (see the header), read once; throws TrialConfigError. */
export function readTrial(env: Env = process.env, readFile: (path: string) => Uint8Array = (p) => readFileSync(p)): Trial {
  const promptFile = required(env, ENV.PROMPT_FILE);
  let prompt: string;
  try {
    prompt = new TextDecoder('utf-8', { fatal: true }).decode(readFile(promptFile));
  } catch (err) {
    throw new TrialConfigError(`cannot read the prompt ${promptFile} (${ENV.PROMPT_FILE}) as UTF-8 text: ${(err as Error).message}`);
  }
  const shopGraphqlUrl = httpUrl(ENV.SHOP_GRAPHQL_URL, required(env, ENV.SHOP_GRAPHQL_URL));
  const shopToken = required(env, ENV.SHOP_TOKEN);
  const supplierText = env[ENV.SUPPLIER_URL];
  const supplierUrl = stripSlash(supplierText === undefined || supplierText.trim() === '' ? new URL(SUPPLIER_PATH, shopGraphqlUrl).href : httpUrl(ENV.SUPPLIER_URL, supplierText));
  const envText = env[ENV.ENV_URL];
  const envUrl = stripSlash(envText === undefined || envText.trim() === '' ? new URL('/', shopGraphqlUrl).href : httpUrl(ENV.ENV_URL, envText));
  const seed = required(env, ENV.TRIAL_SEED);
  const nowMs = parseTrialNow(required(env, ENV.TRIAL_NOW));
  const endpoint = (base: string, key: string, fallback: string): LlmEndpoint => {
    const b = env[base];
    return Object.freeze({ baseUrl: stripSlash(b === undefined || b.trim() === '' ? fallback : httpUrl(base, b)), apiKey: env[key] ?? '' });
  };
  return Object.freeze({
    prompt,
    promptFile,
    shopGraphqlUrl,
    shopToken,
    supplierUrl,
    envUrl,
    seed,
    nowMs,
    nowIso: new Date(nowMs).toISOString(),
    anthropic: endpoint(ENV.ANTHROPIC_BASE_URL, ENV.ANTHROPIC_API_KEY, DEFAULT_ANTHROPIC_BASE_URL),
    openai: endpoint(ENV.OPENAI_BASE_URL, ENV.OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL),
  });
}

/** The trial's fixed now, a fresh Date each call: the trial's own, or MAKEFUN_TRIAL_NOW's. Never the wall clock. */
export function trialNow(trial?: Pick<Trial, 'nowMs'>, env: Env = process.env): Date {
  return new Date(trial === undefined ? parseTrialNow(required(env, ENV.TRIAL_NOW)) : trial.nowMs);
}

/** The env's origin, no trailing slash: the trial's own, or MAKEFUN_ENV_URL's (an entry that reads no trial). */
export function envUrl(trial?: Pick<Trial, 'envUrl'>, env: Env = process.env): string {
  return trial === undefined ? stripSlash(httpUrl(ENV.ENV_URL, required(env, ENV.ENV_URL))) : trial.envUrl;
}

// ---------------------------------------------------------------- the seeded RNG

const MASK64 = (1n << 64n) - 1n;
const TWO64 = 1n << 64n;
const rotl = (x: bigint, k: bigint): bigint => ((x << k) | (x >> (64n - k))) & MASK64;

/**
 * xoshiro256** seeded from sha256(RNG_DOMAIN, 0, seed, 0, label): the same seed and label give the same
 * values here and in makefun_agent.py's Rng. Use one label per purpose, so adding draws to one stream
 * never moves another.
 */
export class Rng {
  readonly seed: string;
  readonly label: string;
  #s: [bigint, bigint, bigint, bigint];

  constructor(seed: string | number | bigint, label = '') {
    if (typeof seed === 'number' && !Number.isSafeInteger(seed)) throw new RangeError(`an Rng seed number must be a safe integer, got ${seed}`);
    this.seed = String(seed);
    this.label = label;
    const h = createHash('sha256').update(RNG_DOMAIN, 'utf8').update('\0').update(this.seed, 'utf8').update('\0').update(label, 'utf8').digest();
    const s: [bigint, bigint, bigint, bigint] = [h.readBigUInt64BE(0), h.readBigUInt64BE(8), h.readBigUInt64BE(16), h.readBigUInt64BE(24)];
    if (s.every((w) => w === 0n)) s[0] = 1n;
    this.#s = s;
  }

  /** The next 64 random bits. */
  nextU64(): bigint {
    const [s0, s1, s2, s3] = this.#s;
    const result = (rotl((s1 * 5n) & MASK64, 7n) * 9n) & MASK64;
    const t = (s1 << 17n) & MASK64;
    const a2 = s2 ^ s0;
    const a3 = s3 ^ s1;
    this.#s = [s0 ^ a3, s1 ^ a2, a2 ^ t, rotl(a3, 45n)];
    return result;
  }

  /** A float in [0, 1) from the top 53 bits. */
  next(): number {
    return Number(this.nextU64() >> 11n) / 9_007_199_254_740_992;
  }

  /** An integer in [min, max], both safe integers, without modulo bias. */
  int(min: number, max: number): number {
    if (!Number.isSafeInteger(min) || !Number.isSafeInteger(max) || min > max) throw new RangeError(`Rng.int needs safe integers min <= max, got ${min}, ${max}`);
    const span = BigInt(max) - BigInt(min) + 1n;
    const limit = TWO64 - (TWO64 % span);
    for (;;) {
      const x = this.nextU64();
      if (x < limit) return Number(BigInt(min) + (x % span));
    }
  }

  /** One of the items, uniformly. */
  pick<T>(items: readonly T[]): T {
    if (items.length === 0) throw new RangeError('Rng.pick of an empty list');
    return items[this.int(0, items.length - 1)]!;
  }

  /** Shuffle in place (Fisher-Yates, from the end) and return the same array. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      const x = items[i]!;
      items[i] = items[j]!;
      items[j] = x;
    }
    return items;
  }
}

/** The trial's RNG stream named `label`: the only randomness that may reach a request. */
export function trialRng(trial: Pick<Trial, 'seed'>, label = ''): Rng {
  return new Rng(trial.seed, label);
}

// ---------------------------------------------------------------- the store: Admin GraphQL and the supplier

export interface GraphQLErrorEntry {
  message: string;
  path?: (string | number)[];
  locations?: { line: number; column: number }[];
  extensions?: Record<string, unknown>;
}
export interface GraphQLResponse<T = unknown> {
  data?: T | null;
  errors?: GraphQLErrorEntry[];
  extensions?: Record<string, unknown>;
}

/** The shop or the supplier answered something that is not its API's answer (a 4xx/5xx from the shop, a body that is not JSON). */
export class StoreHttpError extends Error {
  override name = 'StoreHttpError';
  readonly status: number;
  readonly body: string;
  constructor(message: string, status: number, body: string) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

/** A GraphQL answer carrying `errors` (StoreClient.request): the whole response is kept. */
export class GraphQLRequestError extends Error {
  override name = 'GraphQLRequestError';
  readonly errors: GraphQLErrorEntry[];
  readonly response: GraphQLResponse;
  constructor(response: GraphQLResponse) {
    const errors = response.errors ?? [];
    super(errors.map((e) => (typeof e.extensions?.code === 'string' ? `${e.extensions.code}: ${e.message}` : e.message)).join('; ') || 'GraphQL errors');
    this.errors = errors;
    this.response = response;
  }
}

export interface SupplierResponse<T = unknown> {
  status: number;
  body: T;
}

export interface StoreClientOptions {
  fetch?: typeof fetch;
  /** THROTTLED answers retried per call (default 25: an empty bucket refills a 1,000-point query in 20). */
  maxThrottleRetries?: number;
}

type FetchFn = typeof fetch;

function isThrottled(r: GraphQLResponse): boolean {
  return (r.errors ?? []).some((e) => e.extensions?.code === 'THROTTLED');
}

async function readJson(res: Response, what: string): Promise<{ text: string; json: unknown }> {
  const text = await res.text();
  try {
    return { text, json: JSON.parse(text) as unknown };
  } catch {
    throw new StoreHttpError(`${what} answered ${res.status} with a body that is not JSON: ${text.slice(0, 200)}`, res.status, text);
  }
}

/**
 * The shop's Admin GraphQL API and the supplier's REST API, one request at a time: every call waits for
 * the one before it, so calls reach the store in the order they were made (the determinism rule). Use
 * one StoreClient per trial; two would each keep their own order and race each other.
 */
export class StoreClient {
  readonly graphqlUrl: string;
  readonly supplierUrl: string;
  readonly #token: string;
  readonly #fetch: FetchFn;
  readonly #maxThrottleRetries: number;
  #tail: Promise<void> = Promise.resolve();
  #requests = 0;

  constructor(trial: Pick<Trial, 'shopGraphqlUrl' | 'shopToken' | 'supplierUrl'>, opts: StoreClientOptions = {}) {
    this.graphqlUrl = trial.shopGraphqlUrl;
    this.supplierUrl = stripSlash(trial.supplierUrl);
    this.#token = trial.shopToken;
    this.#fetch = opts.fetch ?? globalThis.fetch;
    this.#maxThrottleRetries = opts.maxThrottleRetries ?? 25;
  }

  /** HTTP requests sent so far, retries included (the store takes 5,000 per trial). */
  get requests(): number {
    return this.#requests;
  }

  #queue<T>(job: () => Promise<T>): Promise<T> {
    const run = this.#tail.then(job);
    this.#tail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  /** One GraphQL operation, its whole answer (data, errors, extensions.cost); THROTTLED is retried. Throws StoreHttpError when the shop answers other than 200. */
  graphql<T = unknown>(query: string, variables?: Record<string, unknown>, operationName?: string): Promise<GraphQLResponse<T>> {
    const body = JSON.stringify({ query, ...(variables === undefined ? {} : { variables }), ...(operationName === undefined ? {} : { operationName }) });
    return this.#queue(async () => {
      for (let attempt = 0; ; attempt++) {
        this.#requests++;
        const res = await this.#fetch(this.graphqlUrl, {
          method: 'POST',
          headers: { 'content-type': 'application/json', accept: 'application/json', 'x-shopify-access-token': this.#token },
          body,
        });
        const { text, json } = await readJson(res, 'the shop');
        if (res.status !== 200) throw new StoreHttpError(`the shop answered ${res.status}: ${text.slice(0, 300)}`, res.status, text);
        if (typeof json !== 'object' || json === null || Array.isArray(json)) throw new StoreHttpError(`the shop answered a JSON ${Array.isArray(json) ? 'array' : typeof json}, not a GraphQL response`, res.status, text);
        const out = json as GraphQLResponse<T>;
        if (!isThrottled(out) || attempt >= this.#maxThrottleRetries) return out;
      }
    });
  }

  /** One GraphQL operation's data; throws GraphQLRequestError when the answer carries errors. userErrors are data: check them. */
  async request<T = unknown>(query: string, variables?: Record<string, unknown>, operationName?: string): Promise<T> {
    const r = await this.graphql<T>(query, variables, operationName);
    if (r.errors !== undefined && r.errors.length > 0) throw new GraphQLRequestError(r as GraphQLResponse);
    return r.data as T;
  }

  /**
   * One supplier request: `path` under the supplier root ('/products', '/products/NW-100', '/quotes'),
   * `query` its parameters in the order given (undefined and null left out). Any status comes back
   * with its JSON body (a 409 is an answer: out of stock); a body that is not JSON throws StoreHttpError.
   */
  supplier<T = unknown>(method: 'GET' | 'POST', path: string, opts: { query?: Readonly<Record<string, string | number | boolean | null | undefined>>; body?: unknown } = {}): Promise<SupplierResponse<T>> {
    if (path !== '' && !path.startsWith('/')) throw new TypeError(`a supplier path starts with '/', got ${JSON.stringify(path)}`);
    const params = new URLSearchParams();
    for (const [k, v] of Object.entries(opts.query ?? {})) if (v !== undefined && v !== null) params.append(k, String(v));
    const qs = params.toString();
    const url = `${this.supplierUrl}${path}${qs === '' ? '' : `?${qs}`}`;
    const body = method === 'POST' ? JSON.stringify(opts.body ?? {}) : undefined;
    return this.#queue(async () => {
      this.#requests++;
      const res = await this.#fetch(url, { method, headers: body === undefined ? { accept: 'application/json' } : { 'content-type': 'application/json', accept: 'application/json' }, body });
      const { json } = await readJson(res, 'the supplier');
      return { status: res.status, body: json as T };
    });
  }

  supplierGet<T = unknown>(path: string, query?: Readonly<Record<string, string | number | boolean | null | undefined>>): Promise<SupplierResponse<T>> {
    return this.supplier<T>('GET', path, { query });
  }

  supplierPost<T = unknown>(path: string, body: unknown): Promise<SupplierResponse<T>> {
    return this.supplier<T>('POST', path, { body });
  }
}

// ---------------------------------------------------------------- the model clients

export type LlmRoute = 'anthropic.messages' | 'openai.chat' | 'models';

/** A model call that failed: an error answer (status, the vendor's error type) or no answer at all (status 0). */
export class LlmError extends Error {
  override name = 'LlmError';
  readonly route: LlmRoute;
  readonly status: number;
  /** The error's `type` (or `code`) from the vendor's error shape, when there is one. */
  readonly type: string | null;
  readonly body: unknown;
  constructor(message: string, route: LlmRoute, status: number, type: string | null, body: unknown) {
    super(message);
    this.route = route;
    this.status = status;
    this.type = type;
    this.body = body;
  }
}

/** The gateway refused the call (the trial's budget, request cap, model allowlist or max_tokens rule): final. */
export class LlmRefusalError extends LlmError {
  override name = 'LlmRefusalError';
}

export interface LlmClientOptions {
  fetch?: typeof fetch;
  /** Retries of a RETRY_STATUSES answer (default 2). */
  maxRetries?: number;
  /** The first backoff; each retry doubles it (default 1,000 ms, no jitter). */
  retryDelayMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

const defaultSleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

function errorOf(route: LlmRoute, status: number, text: string): LlmError {
  let body: unknown = text;
  try {
    body = JSON.parse(text) as unknown;
  } catch {
    // not JSON: kept as text
  }
  const e = typeof body === 'object' && body !== null ? (body as { error?: unknown }).error : undefined;
  const o = typeof e === 'object' && e !== null ? (e as Record<string, unknown>) : {};
  const type = typeof o.type === 'string' ? o.type : typeof o.code === 'string' ? o.code : null;
  const message = typeof o.message === 'string' ? o.message : typeof body === 'string' ? body.slice(0, 300) : '';
  const what = `${route} answered ${status}${type === null ? '' : ` ${type}`}${message === '' ? '' : `: ${message}`}`;
  const refusal = (GATEWAY_REFUSALS as readonly string[]).includes(type ?? '') || (typeof o.code === 'string' && (GATEWAY_REFUSALS as readonly string[]).includes(o.code));
  return refusal ? new LlmRefusalError(what, route, status, type, body) : new LlmError(what, route, status, type, body);
}

async function llmCall(route: LlmRoute, url: string, init: RequestInit, opts: Required<Omit<LlmClientOptions, 'fetch'>> & { fetch: FetchFn }): Promise<unknown> {
  for (let attempt = 0; ; attempt++) {
    let res: Response;
    try {
      res = await opts.fetch(url, init);
    } catch (err) {
      throw new LlmError(`${route}: cannot reach ${url}: ${(err as Error).message}`, route, 0, null, null);
    }
    const text = await res.text();
    if (res.ok) {
      try {
        return JSON.parse(text) as unknown;
      } catch {
        throw new LlmError(`${route} answered ${res.status} with a body that is not JSON`, route, res.status, null, text);
      }
    }
    const err = errorOf(route, res.status, text);
    if (err instanceof LlmRefusalError || !RETRY_STATUSES.includes(res.status) || attempt >= opts.maxRetries) throw err;
    await opts.sleep(opts.retryDelayMs * 2 ** attempt);
  }
}

function endpointOf(source: Pick<Trial, 'anthropic' | 'openai'> | LlmEndpoint, which: 'anthropic' | 'openai'): LlmEndpoint {
  const e = 'baseUrl' in source ? source : source[which];
  return { baseUrl: stripSlash(e.baseUrl), apiKey: e.apiKey ?? '' };
}

function resolved(opts: LlmClientOptions): Required<Omit<LlmClientOptions, 'fetch'>> & { fetch: FetchFn } {
  return { fetch: opts.fetch ?? globalThis.fetch, maxRetries: opts.maxRetries ?? 2, retryDelayMs: opts.retryDelayMs ?? 1000, sleep: opts.sleep ?? defaultSleep };
}

function positiveInt(v: unknown): boolean {
  return typeof v === 'number' && Number.isSafeInteger(v) && v > 0;
}

/** A content block, as sent and as answered: text, tool_use, tool_result, thinking, image... */
export interface AnthropicContentBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  content?: string | AnthropicContentBlock[];
  is_error?: boolean;
  [key: string]: unknown;
}
export interface AnthropicMessageParam {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}
export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  [key: string]: unknown;
}
export interface AnthropicRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessageParam[];
  system?: string | AnthropicContentBlock[];
  tools?: AnthropicTool[];
  tool_choice?: Record<string, unknown>;
  [key: string]: unknown;
}
export interface AnthropicMessage {
  id: string;
  type: 'message';
  role: 'assistant';
  model: string;
  content: AnthropicContentBlock[];
  /** end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal, model_context_window_exceeded. */
  stop_reason: string | null;
  stop_sequence: string | null;
  usage: Record<string, unknown>;
  [key: string]: unknown;
}

/** Anthropic Messages (POST {baseUrl}/v1/messages), JSON only: no streaming. */
export class Anthropic {
  readonly endpoint: LlmEndpoint;
  readonly version: string;
  readonly beta: readonly string[];
  readonly #opts: Required<Omit<LlmClientOptions, 'fetch'>> & { fetch: FetchFn };

  constructor(source: Pick<Trial, 'anthropic' | 'openai'> | LlmEndpoint, opts: LlmClientOptions & { version?: string; beta?: readonly string[] } = {}) {
    this.endpoint = endpointOf(source, 'anthropic');
    this.version = opts.version ?? ANTHROPIC_VERSION;
    this.beta = opts.beta ?? [];
    this.#opts = resolved(opts);
  }

  #headers(json: boolean): Record<string, string> {
    return {
      ...(json ? { 'content-type': 'application/json' } : {}),
      accept: 'application/json',
      'anthropic-version': this.version,
      ...(this.beta.length > 0 ? { 'anthropic-beta': this.beta.join(',') } : {}),
      ...(this.endpoint.apiKey === '' ? {} : { 'x-api-key': this.endpoint.apiKey }),
    };
  }

  /** One message. max_tokens is required (the gateway refuses a call without it); stream is refused. */
  async messages(body: AnthropicRequest): Promise<AnthropicMessage> {
    if (!positiveInt(body.max_tokens)) throw new TypeError('Anthropic.messages: max_tokens must be a positive integer (the gateway refuses a call without it)');
    if (body.stream === true) throw new TypeError('Anthropic.messages: this minimal client takes JSON answers only; stream is not supported');
    return (await llmCall('anthropic.messages', `${this.endpoint.baseUrl}/v1/messages`, { method: 'POST', headers: this.#headers(true), body: JSON.stringify(body) }, this.#opts)) as AnthropicMessage;
  }

  /** The models the endpoint serves (in a trial: the competition's allowlist). */
  async models(): Promise<unknown> {
    return llmCall('models', `${this.endpoint.baseUrl}/v1/models`, { method: 'GET', headers: this.#headers(false) }, this.#opts);
  }
}

export interface ChatToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}
export interface ChatMessage {
  role: 'system' | 'developer' | 'user' | 'assistant' | 'tool';
  content?: string | unknown[] | null;
  tool_calls?: ChatToolCall[];
  tool_call_id?: string;
  name?: string;
  [key: string]: unknown;
}
export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  max_completion_tokens?: number;
  max_tokens?: number;
  tools?: unknown[];
  [key: string]: unknown;
}
export interface ChatCompletion {
  id: string;
  object: string;
  model: string;
  choices: { index: number; message: ChatMessage; finish_reason: string | null }[];
  usage?: Record<string, unknown>;
  [key: string]: unknown;
}

/** OpenAI Chat Completions (POST {baseUrl}/chat/completions), JSON only: no streaming. */
export class OpenAI {
  readonly endpoint: LlmEndpoint;
  readonly #opts: Required<Omit<LlmClientOptions, 'fetch'>> & { fetch: FetchFn };

  constructor(source: Pick<Trial, 'anthropic' | 'openai'> | LlmEndpoint, opts: LlmClientOptions = {}) {
    this.endpoint = endpointOf(source, 'openai');
    this.#opts = resolved(opts);
  }

  #headers(json: boolean): Record<string, string> {
    return { ...(json ? { 'content-type': 'application/json' } : {}), accept: 'application/json', ...(this.endpoint.apiKey === '' ? {} : { authorization: `Bearer ${this.endpoint.apiKey}` }) };
  }

  /** One chat completion. max_completion_tokens (or max_tokens) is required; stream is refused. */
  async chat(body: ChatRequest): Promise<ChatCompletion> {
    if (!positiveInt(body.max_completion_tokens) && !positiveInt(body.max_tokens)) throw new TypeError('OpenAI.chat: max_completion_tokens (or max_tokens) must be a positive integer (the gateway refuses a call without it)');
    if (body.stream === true) throw new TypeError('OpenAI.chat: this minimal client takes JSON answers only; stream is not supported');
    return (await llmCall('openai.chat', `${this.endpoint.baseUrl}/chat/completions`, { method: 'POST', headers: this.#headers(true), body: JSON.stringify(body) }, this.#opts)) as ChatCompletion;
  }

  /** The models the endpoint serves (in a trial: the competition's allowlist). */
  async models(): Promise<unknown> {
    return llmCall('models', `${this.endpoint.baseUrl}/models`, { method: 'GET', headers: this.#headers(false) }, this.#opts);
  }
}
