import { generateText, type UserContent } from 'ai';
import { requiredEnv } from './env';
import { fallbackChainConfigured, fallbackChatModel } from './fallbackModel';

type AnthropicContent =
  | string
  | Array<
      | { type: 'text'; text: string }
      | {
          type: 'image';
          source:
            | {
                type: 'base64';
                media_type:
                  | 'image/jpeg'
                  | 'image/png'
                  | 'image/gif'
                  | 'image/webp';
                data: string;
              }
            | { type: 'url'; url: string };
        }
    >;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function readAnthropicText(data: unknown): string {
  const content = isRecord(data) ? data.content : undefined;
  if (!Array.isArray(content)) {
    throw new Error('anthropic response missing content array');
  }

  for (const part of content) {
    if (
      isRecord(part) &&
      part.type === 'text' &&
      typeof part.text === 'string'
    ) {
      return part.text.trim();
    }
  }

  throw new Error('anthropic response missing text content');
}

function toUserContent(content: AnthropicContent): UserContent {
  if (typeof content === 'string') return content;
  return content.map((part) =>
    part.type === 'text'
      ? { type: 'text' as const, text: part.text }
      : part.source.type === 'base64'
        ? {
            type: 'image' as const,
            image: part.source.data,
            mediaType: part.source.media_type,
          }
        : { type: 'image' as const, image: new URL(part.source.url) },
  );
}

export async function createAnthropicText({
  model,
  system,
  content,
  maxTokens,
}: {
  model: string;
  system: string;
  content: AnthropicContent;
  maxTokens: number;
}): Promise<string> {
  if (fallbackChainConfigured()) {
    const result = await generateText({
      model: fallbackChatModel(),
      system,
      maxOutputTokens: maxTokens,
      messages: [{ role: 'user', content: toUserContent(content) }],
    });
    return result.text.trim();
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': requiredEnv('ANTHROPIC_API_KEY'),
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      system,
      messages: [{ role: 'user', content }],
    }),
  });
  if (!response.ok) {
    throw new Error(`anthropic ${response.status}: ${await response.text()}`);
  }
  return readAnthropicText(await response.json());
}
