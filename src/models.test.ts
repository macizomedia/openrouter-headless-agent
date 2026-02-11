import test from 'node:test';
import assert from 'node:assert/strict';
import { isTextOnlyModel, modelPricePer1M, type OpenRouterModel } from './models.ts';

test('isTextOnlyModel returns true for text-only architectures', () => {
  const model: OpenRouterModel = {
    id: 'provider/model-text',
    architecture: {
      modality: 'text->text',
      input_modalities: ['text'],
      output_modalities: ['text'],
    },
  };

  assert.equal(isTextOnlyModel(model), true);
});

test('isTextOnlyModel returns false when non-text modality is present', () => {
  const model: OpenRouterModel = {
    id: 'provider/model-image',
    architecture: {
      modality: 'text->image',
      input_modalities: ['text'],
      output_modalities: ['image'],
    },
  };

  assert.equal(isTextOnlyModel(model), false);
});

test('isTextOnlyModel returns false without architecture metadata', () => {
  const model: OpenRouterModel = {
    id: 'provider/model-unknown',
  };

  assert.equal(isTextOnlyModel(model), false);
});

test('modelPricePer1M returns blended price per 1M tokens', () => {
  const model: OpenRouterModel = {
    id: 'provider/model-priced',
    pricing: {
      prompt: '0.000001',
      completion: '0.000002',
    },
  };

  const price = modelPricePer1M(model);
  assert.equal(price, 1.5);
});

test('modelPricePer1M returns null when pricing is missing', () => {
  const model: OpenRouterModel = {
    id: 'provider/model-missing',
  };

  assert.equal(modelPricePer1M(model), null);
});
