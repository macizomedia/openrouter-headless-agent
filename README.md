# openrouter-headless-agent

A minimal, headless OpenRouter CLI agent with streaming output and basic tool support.

## Run

```bash
OPENROUTER_API_KEY=sk-or-... npm run start:headless
```

By default it will try to auto-pick a free model (0 cost) from the models API.

## Force a specific model

```bash
OPENROUTER_API_KEY=sk-or-... OPENROUTER_MODEL=openrouter/auto npm run start:headless
```

## Dev (auto-reload)

```bash
OPENROUTER_API_KEY=sk-or-... npm run dev
```

## What’s inside

- Streaming agent core in `src/agent.ts`
- Model selection helper in `src/models.ts`
- Example tools in `src/tools.ts`
- Headless CLI in `src/headless.ts`
