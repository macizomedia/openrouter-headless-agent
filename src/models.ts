export interface OpenRouterModel {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
  pricing?: {
    prompt?: string;
    completion?: string;
    image?: string;
    request?: string;
  };
  top_provider?: {
    is_moderated?: boolean;
  };
}

function asNumber(x: unknown): number {
  if (typeof x === 'number') return x;
  if (typeof x === 'string') {
    const n = Number(x);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}

export async function fetchModels(opts?: {
  signal?: AbortSignal;
}): Promise<OpenRouterModel[]> {
  const res = await fetch('https://openrouter.ai/api/v1/models', {
    method: 'GET',
    headers: {
      // These headers are recommended by OpenRouter for attribution/analytics.
      // Safe defaults; customize as needed.
      'HTTP-Referer': 'http://localhost',
      'X-Title': 'openrouter-agent',
    },
    signal: opts?.signal,
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.status} ${res.statusText}`);
  }

  const data = (await res.json()) as { data?: OpenRouterModel[] };
  return Array.isArray(data.data) ? data.data : [];
}

/**
 * Heuristic: pick a "free" model (prompt+completion cost 0) if available.
 * Falls back to openrouter/auto.
 */
export async function pickFreeModelId(opts?: {
  /** Minimum required context length. */
  minContext?: number;
  /** Prefer a context length close to this value (useful if you want ~8k-16k). */
  targetContext?: number;
  /** If false, exclude models where the top provider is moderated. */
  allowModerated?: boolean;
  /** Optional preference hints; set to [] if you don't want any provider/model-name bias. */
  preferIdIncludes?: string[];
}): Promise<string> {
  const models = await fetchModels();

  const minContext = opts?.minContext ?? 0;
  const targetContext = opts?.targetContext;
  const allowModerated = opts?.allowModerated ?? true;
  const preferIdIncludes = opts?.preferIdIncludes ?? ['free', 'trial'];

  const free = models.filter((m) => {
    const prompt = asNumber(m.pricing?.prompt);
    const completion = asNumber(m.pricing?.completion);
    const request = asNumber(m.pricing?.request);

    // Some models omit pricing fields; treat missing as "unknown" (not disqualifying).
    const isFree =
      (Number.isFinite(prompt) ? prompt === 0 : true) &&
      (Number.isFinite(completion) ? completion === 0 : true) &&
      (Number.isFinite(request) ? request === 0 : true);

    const ctxOk = (m.context_length ?? 0) >= minContext;
    const moderationOk = allowModerated ? true : !(m.top_provider?.is_moderated ?? false);

    return isFree && ctxOk && moderationOk;
  });

  if (!free.length) return 'openrouter/auto';

  // Optional substring preference (can be disabled by passing preferIdIncludes: [])
  for (const needle of preferIdIncludes) {
    const hit = free.find((m) => m.id.toLowerCase().includes(needle.toLowerCase()));
    if (hit) return hit.id;
  }

  // Otherwise: pick by heuristic score.
  // - If targetContext is set: choose closest to target, tie-breaker larger context.
  // - Else: choose largest context.
  const scored = free
    .map((m) => {
      const ctx = m.context_length ?? 0;
      const distance = targetContext != null ? Math.abs(ctx - targetContext) : 0;
      return { m, ctx, distance };
    })
    .sort((a, b) => {
      if (targetContext != null && a.distance !== b.distance) return a.distance - b.distance;
      return b.ctx - a.ctx;
    });

  return scored[0].m.id;
}
