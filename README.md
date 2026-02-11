# openrouter-headless-agent

A minimal, headless OpenRouter CLI agent with streaming output and basic tool support.

## Install (npm)

```bash
npm i -g @abquanta/openrouter-headless-agent
```

Then run:

```bash
openrouter-cli
```

## Install (local)

```bash
npm install
npm run build
npm link
```

## Run (no install)

```bash
OPENROUTER_API_KEY=sk-or-... npm run start:headless
```

## Test

```bash
npm run test
```

## Docker

```bash
docker build -t openrouter-headless-agent .
docker run -it -e OPENROUTER_API_KEY=sk-or-... openrouter-headless-agent
```

## What’s inside

- Streaming agent core in `src/agent.ts`
- Model selection + pricing helpers in `src/models.ts`
- Example tools in `src/tools.ts`
- CLI wizard in `src/cli.ts`
