import { tool } from '@openrouter/sdk';
import { z } from 'zod';

export const timeTool = tool({
  name: 'get_current_time',
  description: 'Get the current date and time',
  inputSchema: z.object({
    timezone: z.string().optional().describe('Timezone (e.g., "UTC", "Europe/Berlin")'),
  }),
  execute: async ({ timezone }) => {
    return {
      time: new Date().toLocaleString('en-US', { timeZone: timezone || 'UTC' }),
      timezone: timezone || 'UTC',
    };
  },
});

export const calculatorTool = tool({
  name: 'calculate',
  description: 'Perform basic mathematical calculations',
  inputSchema: z.object({
    expression: z.string().describe('Math expression (e.g., "2 + 2", "(10/4) + 1")'),
  }),
  execute: async ({ expression }) => {
    // Simple safe-ish eval for basic operators. Still: do not expose to untrusted input in production.
    const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, '');
    const result = Function('"use strict"; return (' + sanitized + ')')();
    return { expression, sanitized, result };
  },
});

export const defaultTools = [timeTool, calculatorTool];
