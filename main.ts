// photo-geolocator-01 entry: a vision-model geolocation agent with a code-only fallback.
//
// For each of the instance's 25 photos, in order, it asks claude-sonnet-5-5 where the photo was
// taken and posts that answer to the env as soon as it has it, so partial progress survives a
// mid-trial failure. When the model cannot answer (no provider key pushed, a gateway refusal, a
// transient error that survives retries, or an answer with no coordinates), it posts a fixed
// code-only guess instead of leaving the photo unanswered: the geographic centroid of the 75
// practice photos (lat 22.171, lon 10.5056). An unanswered photo counts 20,015 km of error; the
// centroid guess averages ~8,560 km on the practice set, so it is always worth posting.
//
// Deterministic: one photo at a time, in photo order; every model request is a function of the
// photo's bytes and a fixed prompt; no wall clock, no unseeded randomness, nothing time- or
// order-dependent in any request body. Model calls run sequentially so a replay sends the same
// calls in the same order and leaves the same answers.
//
// Runs as `node /bot/main.ts` on Node 22 (erasable TypeScript, standard library only plus the
// vendored SDK in vendor/makefun-agent.ts), with the trial's variables: MAKEFUN_ENV_URL,
// ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY (the gateway and the key it swaps for the real one).
// Model calls run on the Anthropic key pushed to the competition before submission; with no key
// every call is answered 401 and every photo gets the centroid guess.

import { readFileSync } from 'node:fs';
import { Anthropic } from './vendor/makefun-agent.ts';

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 1024;
/** Fixed code-only guess: centroid of the 75 practice photos (mean lat/lon of their true places). */
const FALLBACK_LAT = 22.171;
const FALLBACK_LON = 10.5056;

const PROMPT = [
  'You are a geolocation expert. Look at this street-level photo carefully and work out where on Earth it was taken.',
  'Use every visible clue: the language and script on signs, the side of the road traffic drives on, road markings and signs, license plates, vegetation and climate, architecture and building materials, the landscape, and the sun and shadows.',
  'Reason briefly (2-4 sentences) naming the clues you used, then give your best estimate as one line of JSON and nothing else after it:',
  '{"lat": <degrees north, -90 to 90>, "lon": <degrees east, -180 to 180>}.',
  'If you are unsure, still give your single best guess — never refuse, never give a range.',
].join('\n');

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const llm = new Anthropic({
  baseUrl: need('ANTHROPIC_BASE_URL').replace(/\/+$/, ''),
  apiKey: process.env.ANTHROPIC_API_KEY ?? '',
});

// The trial brief, read as the harness hands it; its content does not change the fixed prompt.
try {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  if (promptFile !== undefined && promptFile !== '') readFileSync(promptFile, 'utf8');
} catch {
  // The brief is advisory here; the agent works without it.
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next candidate, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/**
 * The model's coordinates for one photo, or null when it cannot answer. The SDK retries
 * transient errors (408/429/5xx/529) itself; a gateway refusal (budget, cap, allowlist, or a
 * 401 for a missing provider key) throws at once and is final.
 */
async function ask(jpeg: Buffer): Promise<{ lat: number; lon: number } | null> {
  const msg = await llm.messages({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  const text = (msg.content ?? [])
    .filter((b) => b.type === 'text')
    .map((b) => b.text ?? '')
    .join('\n');
  return coordinates(text);
}

/** Post one answer; throws when the env refuses it. */
async function postAnswer(id: string, lat: number, lon: number): Promise<void> {
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers: [{ id, lat, lon }] }),
  });
  if (!res.ok) throw new Error(`POST /answers for ${id} answered ${res.status}: ${(await res.text()).slice(0, 300)}`);
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  let modelOk = 0;
  let fallback = 0;
  let modelDead = false; // a 401 / budget refusal means every later call fails the same way: skip them
  for (const p of photos) {
    let guess: { lat: number; lon: number } | null = null;
    if (!modelDead) {
      try {
        const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
        guess = await ask(jpeg);
        if (guess === null) process.stderr.write(`agent: no coordinates in the model answer for ${p.id}; using fallback\n`);
      } catch (err) {
        const e = err as { status?: number; type?: string | null; name?: string };
        process.stderr.write(`agent: model call failed for ${p.id} (${e.name ?? 'error'}${e.status !== undefined ? ` HTTP ${e.status}` : ''}${e.type ? ` ${e.type}` : ''}); using fallback\n`);
        // 401 (no provider key), budget_exhausted and request_cap are final for the whole trial.
        if (e.status === 401 || e.type === 'budget_exhausted' || e.type === 'request_cap') modelDead = true;
      }
    }
    if (guess === null) {
      guess = { lat: FALLBACK_LAT, lon: FALLBACK_LON };
      fallback += 1;
    } else {
      modelOk += 1;
    }
    await postAnswer(p.id, guess.lat, guess.lon);
  }
  process.stdout.write(`agent: ${photos.length} photos answered (${modelOk} by model, ${fallback} fallback)\n`);
  return 0;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`agent: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
