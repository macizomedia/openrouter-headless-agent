#!/usr/bin/env node
import { Command } from 'commander';
import { input, select } from '@inquirer/prompts';
import { createAgent } from './agent.ts';
import { defaultTools } from './tools.ts';
import { fetchModels, isTextOnlyModel, modelPricePer1M, type OpenRouterModel } from './models.ts';

const PRICE_TIERS = [
  { id: 'minimal', label: 'Minimal (<$5 / 1M tokens)', min: 0, max: 5 },
  { id: 'basic', label: 'Basic ($5–$10 / 1M tokens)', min: 5, max: 10 },
  { id: 'full', label: 'Full ($10–$20 / 1M tokens)', min: 10, max: 20 },
] as const;

type PriceTierId = (typeof PRICE_TIERS)[number]['id'];

type WizardResult = {
  model: string;
};

function formatUsd(value: number): string {
  return `$${value.toFixed(2)}`;
}

function parseModelProvider(modelId: string): string {
  const idx = modelId.indexOf('/');
  return idx === -1 ? 'unknown' : modelId.slice(0, idx);
}

function isFreeModel(model: OpenRouterModel): boolean {
  const pricing = model.pricing;
  if (!pricing) return false;
  const prompt = Number(pricing.prompt);
  const completion = Number(pricing.completion);
  const request = Number(pricing.request ?? 0);
  if (![prompt, completion, request].every((n) => Number.isFinite(n))) return false;
  return prompt === 0 && completion === 0 && request === 0;
}

function filterByPriceTier(models: OpenRouterModel[], tier: PriceTierId): OpenRouterModel[] {
  const match = PRICE_TIERS.find((t) => t.id === tier);
  if (!match) return models;
  return models.filter((m) => {
    const price = modelPricePer1M(m);
    if (price == null) return false;
    return price >= match.min && price < match.max;
  });
}

function sortByPrice(models: OpenRouterModel[]): OpenRouterModel[] {
  return [...models].sort((a, b) => {
    const pa = modelPricePer1M(a);
    const pb = modelPricePer1M(b);
    if (pa == null && pb == null) return 0;
    if (pa == null) return 1;
    if (pb == null) return -1;
    return pa - pb;
  });
}

async function runWizard(): Promise<WizardResult> {
  const allModels = await fetchModels();
  const textModels = allModels.filter(isTextOnlyModel);

  const pricingChoice = await select({
    message: 'Select pricing type',
    choices: [
      { name: 'Free models only', value: 'free' },
      { name: 'Paid models', value: 'paid' },
    ],
  });

  let filtered = textModels;

  if (pricingChoice === 'free') {
    filtered = textModels.filter(isFreeModel);
  } else {
    const tier = await select({
      message: 'Choose a price tier',
      choices: PRICE_TIERS.map((t) => ({ name: t.label, value: t.id })),
    });

    filtered = filterByPriceTier(textModels, tier as PriceTierId);

    const providers = Array.from(new Set(filtered.map((m) => parseModelProvider(m.id)))).sort();
    const providerChoice = await select({
      message: 'Choose a provider (or skip)',
      choices: [
        { name: 'Skip provider filter', value: 'skip' },
        ...providers.map((p) => ({ name: p, value: p })),
      ],
    });

    if (providerChoice !== 'skip') {
      filtered = filtered.filter((m) => parseModelProvider(m.id) === providerChoice);
    }
  }

  if (!filtered.length) {
    throw new Error('No models matched your filters. Try a different tier or provider.');
  }

  const ordered = sortByPrice(filtered);
  const shortlist = ordered.slice(0, 30);

  const modelChoice = await select({
    message: 'Select a model',
    pageSize: 20,
    choices: [
      ...shortlist.map((m) => {
        const price = modelPricePer1M(m);
        const priceLabel = price == null ? 'unknown' : `${formatUsd(price)}/1M`; // blended
        return {
          name: `${m.id} (${priceLabel})`,
          value: m.id,
        };
      }),
      { name: 'Enter model id manually', value: '__manual__' },
    ],
  });

  if (modelChoice === '__manual__') {
    const manual = await input({
      message: 'Enter model id (e.g., openai/gpt-4o-mini):',
      validate: (value) => (value.trim().length ? true : 'Model id is required'),
    });
    return { model: manual.trim() };
  }

  return { model: String(modelChoice) };
}

async function resolveApiKey(): Promise<string> {
  if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;

  const key = await input({
    message: 'OpenRouter API key (sk-or-...):',
    validate: (value) => (value.trim().length ? true : 'API key is required'),
  });

  return key.trim();
}

async function startAgent(opts: { model: string; apiKey: string }): Promise<void> {
  console.log(`[model] ${opts.model}`);

  const agent = createAgent({
    apiKey: opts.apiKey,
    model: opts.model,
    instructions: 'You are a helpful assistant. Be concise.',
    tools: defaultTools,
    maxSteps: 5,
  });

  const formatMaybeJsonError = (err: unknown): string => {
    const e = err as any;

    const candidates: unknown[] = [e?.body, e?.message, e?.error?.message];
    const raw = candidates.find((c) => typeof c === 'string') as string | undefined;

    const fallback = () => {
      if (raw) return raw;
      if (e?.message) return String(e.message);
      try {
        return JSON.stringify(e);
      } catch {
        return String(err);
      }
    };

    if (!raw) return fallback();

    const trimmed = raw.trim();
    const maybeJson =
      (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
      (trimmed.startsWith('"{') && trimmed.endsWith('}"'));

    if (!maybeJson) return raw;

    try {
      const jsonText = trimmed.startsWith('"') ? JSON.parse(trimmed) : trimmed;
      const parsed = JSON.parse(jsonText);

      const oe = parsed?.error;
      if (oe?.message) {
        const code = oe?.code != null ? ` (code ${oe.code})` : '';
        const requested = oe?.metadata?.requested_providers;
        const available = oe?.metadata?.available_providers;

        const lines: string[] = [`${oe.message}${code}`];

        if (Array.isArray(requested) && requested.length) {
          lines.push(`requested_providers: ${requested.join(', ')}`);
        }

        if (Array.isArray(available) && available.length) {
          const head = available.slice(0, 12);
          const more = available.length > head.length ? ` (+${available.length - head.length} more)` : '';
          lines.push(`available_providers: ${head.join(', ')}${more}`);
        }

        return lines.join('\n');
      }

      return JSON.stringify(parsed);
    } catch {
      return raw;
    }
  };

  agent.on('tool:call', (name, args) => console.log(`\n[tool call] ${name}`, args));
  agent.on('stream:delta', (delta) => process.stdout.write(delta));
  agent.on('stream:end', () => process.stdout.write('\n'));
  agent.on('error', (err) => console.error('\n[error]', formatMaybeJsonError(err)));

  const readline = await import('node:readline');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  console.log('OpenRouter headless agent ready. Ctrl+C to exit.\n');

  const prompt = () => {
    rl.question('You: ', async (raw: string) => {
      const text = raw.trim();
      if (!text) return prompt();
      await agent.send(text);
      prompt();
    });
  };

  prompt();
}

async function main() {
  const program = new Command();

  program
    .name('openrouter-cli')
    .description('Headless OpenRouter CLI agent')
    .option('-m, --model <id>', 'Model id to use (skips wizard)')
    .option('--no-wizard', 'Skip the interactive wizard')
    .parse(process.argv);

  const options = program.opts<{ model?: string; wizard: boolean }>();

  let model = options.model || process.env.OPENROUTER_MODEL;

  if (!model && options.wizard) {
    const wizard = await runWizard();
    model = wizard.model;
  }

  if (!model) {
    console.error('Missing model selection. Use --model or enable the wizard.');
    process.exit(1);
  }

  const apiKey = await resolveApiKey();
  await startAgent({ apiKey, model });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
