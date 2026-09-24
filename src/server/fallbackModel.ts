import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { APICallError, type LanguageModelV3 } from '@ai-sdk/provider';
import { env } from './env';

// Self-hosted model chain: try each OpenAI-compatible upstream in order and
// fall through to the next one when a call fails before streaming starts
// (quota exhausted, rate limited, upstream down). A failed upstream is parked
// for a cooldown so later requests skip straight to the next one; once the
// cooldown passes it is tried first again, so the chain drifts back to the
// preferred provider on its own after a quota reset.
//
// Order: xKiro (daily free quota) -> Ollama cloud (weekly quota) ->
// Cloudflare Workers AI (account 2, then account 1).

type Upstream = {
  name: string;
  model: LanguageModelV3;
  maxOutputTokens: number;
  quotaCooldownMs: number;
};

type CallOptions = Parameters<LanguageModelV3['doGenerate']>[0];

const MINUTE = 60_000;
const OTHER_ERROR_COOLDOWN_MS = 2 * MINUTE;
const QUOTA_PATTERN =
  /quota|rate.?limit|too many|exceed|insufficient|balance|credit|limit reached|usage limit/i;

// In-process memory. A server restart forgets it, which only costs one extra
// failed attempt against an exhausted upstream.
const parkedUntil = new Map<string, number>();

// Qwen3 on Workers AI thinks for minutes before its first tool call unless
// thinking is switched off through the chat template.
function fetchWithExtraBody(extra: Record<string, unknown>): typeof fetch {
  return async (input, init) => {
    if (typeof init?.body === 'string') {
      try {
        init = { ...init, body: JSON.stringify({ ...JSON.parse(init.body), ...extra }) };
      } catch {
        // not JSON: send unchanged
      }
    }
    return fetch(input, init);
  };
}

function openAICompatibleModel(
  name: string,
  baseURL: string,
  apiKey: string,
  modelId: string,
  extraBody?: Record<string, unknown>,
): LanguageModelV3 {
  return createOpenAICompatible({
    name,
    baseURL: baseURL.replace(/\/+$/, ''),
    apiKey,
    // xKiro sits behind Cloudflare bot protection that rejects requests
    // without a normal User-Agent (error 1010).
    headers: { 'User-Agent': 'cadam-rj/1.0' },
    ...(extraBody ? { fetch: fetchWithExtraBody(extraBody) } : {}),
  }).chatModel(modelId);
}

function configuredUpstreams(): Upstream[] {
  const upstreams: Upstream[] = [];
  const add = (
    name: string,
    baseURL: string,
    apiKey: string,
    modelId: string,
    maxOutputTokens: number,
    quotaCooldownMs: number,
    extraBody?: Record<string, unknown>,
  ) => {
    if (!baseURL || !apiKey || !modelId) return;
    upstreams.push({
      name,
      model: openAICompatibleModel(name, baseURL, apiKey, modelId, extraBody),
      maxOutputTokens,
      quotaCooldownMs,
    });
  };

  add(
    'xkiro',
    env('XKIRO_BASE_URL') || 'https://api.xkiro.com/v1',
    env('XKIRO_API_KEY'),
    env('XKIRO_MODEL') || 'qwen/qwen3.8-max:free',
    32000,
    60 * MINUTE,
  );
  add(
    'ollama',
    env('OLLAMA_BASE_URL') || 'https://ollama.com/v1',
    env('OLLAMA_API_KEY'),
    env('OLLAMA_MODEL') || 'gpt-oss:120b',
    32000,
    6 * 60 * MINUTE,
  );
  for (const suffix of ['', '_2']) {
    add(
      `workers-ai${suffix}`,
      env(`WORKERS_AI_BASE_URL${suffix}`),
      env(`WORKERS_AI_API_KEY${suffix}`),
      env(`WORKERS_AI_MODEL${suffix}`) || '@cf/qwen/qwen3.8-27b',
      16000,
      60 * MINUTE,
      { chat_template_kwargs: { enable_thinking: false } },
    );
  }
  return upstreams;
}

export function fallbackChainConfigured(): boolean {
  return Boolean(
    env('XKIRO_API_KEY') ||
      env('OLLAMA_API_KEY') ||
      env('WORKERS_AI_API_KEY') ||
      env('WORKERS_AI_API_KEY_2'),
  );
}

function isQuotaError(error: unknown): boolean {
  if (APICallError.isInstance(error)) {
    if ([402, 403, 429].includes(error.statusCode ?? 0)) return true;
    return QUOTA_PATTERN.test(`${error.message} ${error.responseBody ?? ''}`);
  }
  return error instanceof Error && QUOTA_PATTERN.test(error.message);
}

function withOutputCap(options: CallOptions, cap: number): CallOptions {
  const requested = options.maxOutputTokens;
  return {
    ...options,
    maxOutputTokens: requested ? Math.min(requested, cap) : cap,
  };
}

async function runChain<T>(
  upstreams: Upstream[],
  options: CallOptions,
  call: (model: LanguageModelV3, options: CallOptions) => PromiseLike<T>,
): Promise<T> {
  if (upstreams.length === 0) {
    throw new Error('No AI upstream configured (XKIRO/OLLAMA/WORKERS_AI keys)');
  }
  const now = Date.now();
  const available = upstreams.filter(
    (u) => (parkedUntil.get(u.name) ?? 0) <= now,
  );
  // Everything parked: try them all anyway rather than failing outright.
  const order = available.length > 0 ? available : upstreams;

  let lastError: unknown;
  for (const upstream of order) {
    try {
      const result = await call(
        upstream.model,
        withOutputCap(options, upstream.maxOutputTokens),
      );
      parkedUntil.delete(upstream.name);
      console.log(`[ai-fallback] served by ${upstream.name}`);
      return result;
    } catch (error) {
      if (options.abortSignal?.aborted) throw error;
      lastError = error;
      const quota = isQuotaError(error);
      parkedUntil.set(
        upstream.name,
        Date.now() + (quota ? upstream.quotaCooldownMs : OTHER_ERROR_COOLDOWN_MS),
      );
      console.warn(
        `[ai-fallback] ${upstream.name} failed (${quota ? 'quota' : 'error'}), trying next:`,
        error instanceof Error ? error.message : error,
      );
    }
  }
  throw lastError;
}

export function fallbackChatModel(): LanguageModelV3 {
  const upstreams = configuredUpstreams();
  return {
    specificationVersion: 'v3',
    provider: 'rj-fallback',
    modelId: upstreams.map((u) => u.name).join('>') || 'none',
    supportedUrls: {},
    doGenerate: (options) =>
      runChain(upstreams, options, (model, opts) => model.doGenerate(opts)),
    doStream: (options) =>
      runChain(upstreams, options, (model, opts) => model.doStream(opts)),
  };
}
