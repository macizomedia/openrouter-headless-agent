import { createAgent } from './agent.ts';
import { defaultTools } from './tools.ts';
import { pickFreeModelId } from './models.ts';

async function main() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('Missing OPENROUTER_API_KEY. Example:');
    console.error('  OPENROUTER_API_KEY=sk-or-... npm run start:headless');
    process.exit(1);
  }

  // Override if you want a specific model:
  //   OPENROUTER_MODEL=anthropic/claude-... OPENROUTER_API_KEY=... npm run start:headless
  const model =
    process.env.OPENROUTER_MODEL ||
    (await pickFreeModelId({
      // Your preference: unmoderated, and roughly 8k–16k context.
      minContext: 8_000,
      targetContext: 16_000,
      allowModerated: false,
      // No provider/model-name preference.
      preferIdIncludes: [],
    }));

  console.log(`[model] ${model}`);

  const agent = createAgent({
    apiKey,
    model,
    instructions: 'You are a helpful assistant. Be concise.',
    tools: defaultTools,
    maxSteps: 5,
  });

  const formatMaybeJsonError = (err: unknown): string => {
    const e = err as any;

    // Try to extract a raw string body (OpenRouter sometimes returns JSON string bodies)
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

    // If message contains JSON (sometimes wrapped in quotes), parse it.
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

      // Unknown JSON shape: return a compact one-liner.
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
    rl.question('You: ', async (input: string) => {
      const text = input.trim();
      if (!text) return prompt();
      await agent.send(text);
      prompt();
    });
  };

  prompt();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
