import test from 'node:test';
import assert from 'node:assert/strict';

// Helper to validate calibration image count rules
function validateImageCount(count: number): { valid: boolean; error?: string } {
  if (count < 10) {
    return {
      valid: false,
      error: `At least 10 reference images are required (got ${count}).`,
    };
  }
  if (count > 20) {
    return {
      valid: false,
      error: `At most 20 reference images are accepted (got ${count}).`,
    };
  }
  return { valid: true };
}

// Mock File implementation for testing in Node
function createMockFile(name: string, content: string = 'fake_image_bytes'): File {
  const blob = new Blob([content], { type: 'image/png' });
  return new File([blob], name, { type: 'image/png' });
}

test('Boundary validation: 9 files must be rejected', () => {
  const result = validateImageCount(9);
  assert.equal(result.valid, false);
  assert.match(result.error || '', /At least 10/);
});

test('Boundary validation: 10 files must be accepted', () => {
  const result = validateImageCount(10);
  assert.equal(result.valid, true);
  assert.equal(result.error, undefined);
});

test('Boundary validation: 20 files must be accepted', () => {
  const result = validateImageCount(20);
  assert.equal(result.valid, true);
  assert.equal(result.error, undefined);
});

test('Boundary validation: 21 files must be rejected', () => {
  const result = validateImageCount(21);
  assert.equal(result.valid, false);
  assert.match(result.error || '', /At most 20/);
});

test('runProductCalibration: builds FormData with actual File objects and calls /calibration/run', async () => {
  const capturedRequests: { url: string; method: string; body: FormData }[] = [];

  // Mock global fetch
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: input.toString(),
      method: init?.method || 'GET',
      body: init?.body as FormData,
    });

    const mockResponse = {
      product_id: 'test_product_123',
      calibration_id: 42,
      threshold: 38.225489,
      calibration_method: 'normal_k_sigma',
      reference_image_count: 10,
      model_path: 'data/app_storage/products/test_product_123/models/test_product_123_patchcore.ckpt',
      artifact_path: 'data/app_storage/products/test_product_123/artifacts/test_product_123_calibration_artifact.json',
      normal_score_stats: {
        count: 10,
        min: 28.02,
        max: 36.27,
        mean: 30.50,
        std: 2.58,
      },
      warnings: [],
      product_status: 'READY',
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 201,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    // Dynamically import api service
    const { runProductCalibration } = await import('../src/services/api.ts');

    const files = Array.from({ length: 10 }, (_, i) =>
      createMockFile(`ref_${String(i).padStart(3, '0')}.png`, `image_bytes_${i}`)
    );

    const productId = 'test_product_123';
    const result = await runProductCalibration(productId, files);

    // 1. Verify correct endpoint called
    assert.equal(capturedRequests.length, 1);
    const req = capturedRequests[0];
    assert.match(req.url, new RegExp(`/products/${productId}/calibration/run$`));
    assert.equal(req.method, 'POST');

    // 2. Verify FormData structure
    const formData = req.body;
    assert.ok(formData instanceof FormData, 'Body must be an instance of FormData');

    const appendedFiles = formData.getAll('files');
    assert.equal(appendedFiles.length, 10, 'All 10 File objects must be appended under "files"');

    // 3. Verify actual File instances with proper names
    for (let i = 0; i < 10; i++) {
      const file = appendedFiles[i];
      assert.ok(file instanceof File || (file as any) instanceof Blob, 'Appended item must be File/Blob');
      if (file instanceof File) {
        assert.equal(file.name, `ref_${String(i).padStart(3, '0')}.png`);
      }
    }

    // 4. Verify NO hardcoded threshold/checkpoint/artifact paths were sent
    assert.equal(formData.get('threshold'), null, 'Must not send threshold');
    assert.equal(formData.get('patchcore_model_path'), null, 'Must not send patchcore_model_path');
    assert.equal(formData.get('artifact_path'), null, 'Must not send artifact_path');
    assert.equal(formData.get('model_path'), null, 'Must not send model_path');

    // 5. Verify response fields
    assert.equal(result.product_id, 'test_product_123');
    assert.equal(result.threshold, 38.225489);
    assert.equal(result.product_status, 'READY');
    assert.equal(result.reference_image_count, 10);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('runProductCalibration: handles backend error clearly without false READY state', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    return new Response(
      JSON.stringify({ detail: 'PatchCore fitting failed: corrupted image data.' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  };

  try {
    const { runProductCalibration } = await import('../src/services/api.ts');
    const files = Array.from({ length: 10 }, (_, i) =>
      createMockFile(`ref_${i}.png`)
    );

    await assert.rejects(
      async () => {
        await runProductCalibration('fail_prod', files);
      },
      (err: Error) => {
        assert.match(err.message, /PatchCore fitting failed: corrupted image data\./);
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
