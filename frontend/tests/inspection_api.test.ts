import test from 'node:test';
import assert from 'node:assert/strict';

// Mock File implementation for testing in Node
function createMockFile(name: string, content: string = 'fake_inspection_image_bytes'): File {
  const blob = new Blob([content], { type: 'image/png' });
  return new File([blob], name, { type: 'image/png' });
}

// Logic unit: Ready product filtering
function filterReadyProducts(products: { product_id: string; status: string }[]) {
  return products.filter((p) => p.status === 'READY');
}

// Logic unit: Validation before inspection
function validateInspectionPrerequisites(productId: string | null | undefined, file: File | null | undefined) {
  if (!productId || productId.trim() === '') {
    return { valid: false, error: 'Please select a calibrated (READY) product before inspecting.' };
  }
  if (!file) {
    return { valid: false, error: 'Please select or upload an inspection image.' };
  }
  return { valid: true };
}

// Logic unit: Camera inspection prerequisite validation
function validateCameraInspectionPrerequisites(
  productId: string | null | undefined,
  isCameraActive: boolean,
  videoReady: boolean,
  videoDimensions: { width: number; height: number } | null | undefined
) {
  if (!productId || productId.trim() === '') {
    return { valid: false, error: 'Please select a calibrated (READY) product before inspecting.' };
  }
  if (!isCameraActive || !videoReady || !videoDimensions) {
    return { valid: false, error: 'Camera stream is not ready. Please start camera and wait for feed.' };
  }
  if (videoDimensions.width <= 0 || videoDimensions.height <= 0) {
    return { valid: false, error: 'Camera feed has invalid dimensions.' };
  }
  return { valid: true };
}


test('Product selection: only READY products are eligible for inspection', () => {
  const testProducts = [
    { product_id: 'prod_draft', status: 'DRAFT' },
    { product_id: 'prod_ready_1', status: 'READY' },
    { product_id: 'prod_calibrating', status: 'CALIBRATING' },
    { product_id: 'prod_ready_2', status: 'READY' },
  ];

  const readyList = filterReadyProducts(testProducts);
  assert.equal(readyList.length, 2);
  assert.deepEqual(
    readyList.map((p) => p.product_id),
    ['prod_ready_1', 'prod_ready_2']
  );
});

test('Validation: inspection blocked if no product selected', () => {
  const file = createMockFile('query.png');
  const result = validateInspectionPrerequisites('', file);
  assert.equal(result.valid, false);
  assert.match(result.error || '', /select a calibrated \(READY\) product/);
});

test('Validation: inspection blocked if no image selected', () => {
  const result = validateInspectionPrerequisites('prod_ready_1', null);
  assert.equal(result.valid, false);
  assert.match(result.error || '', /upload an inspection image/);
});

test('inspectProduct: sends actual File object, correct product ID, and correct inspection endpoint', async () => {
  const capturedRequests: { url: string; method: string; body: FormData }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: input.toString(),
      method: init?.method || 'GET',
      body: init?.body as FormData,
    });

    const mockInspectionResponse = {
      id: 101,
      inspection_id: 'insp_101',
      product_id: 'prod_ready_bottle',
      created_at: '2026-10-03T18:00:00Z',
      timestamp: '2026-10-03T18:00:00Z',
      image_path: 'data/app_storage/inspections/insp_101/image.png',
      heatmap_path: 'data/app_storage/inspections/insp_101/heatmap.png',
      anomaly_score: 18.4521,
      anomaly_threshold: 38.2255,
      margin: -19.7734,
      verdict: 'NORMAL',
      defect_class: null,
      classifier_confidence: null,
      classifier_probabilities: null,
      stage2_executed: false,
      execution_time_ms: 45.2,
    };

    return new Response(JSON.stringify(mockInspectionResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const { inspectProduct } = await import('../src/services/api.ts');
    const imageFile = createMockFile('test_bottle_sample.png', 'actual_png_data');
    const productId = 'prod_ready_bottle';

    const result = await inspectProduct(productId, imageFile, true);

    // 1. Verify correct endpoint called with query param
    assert.equal(capturedRequests.length, 1);
    const req = capturedRequests[0];
    assert.match(req.url, new RegExp(`/products/${productId}/inspect\\?save_heatmap=true$`));
    assert.equal(req.method, 'POST');

    // 2. Verify FormData structure
    const formData = req.body;
    assert.ok(formData instanceof FormData, 'Body must be FormData');

    // 3. Verify actual File object was sent under 'image'
    const appendedImage = formData.get('image');
    assert.ok(appendedImage instanceof File || (appendedImage as any) instanceof Blob, 'image key must contain File/Blob');
    if (appendedImage instanceof File) {
      assert.equal(appendedImage.name, 'test_bottle_sample.png');
    }
    assert.equal(formData.get('save_heatmap'), 'true');

    // 4. Verify no fabricated values sent
    assert.equal(formData.get('anomaly_score'), null, 'Must not send fabricated score');
    assert.equal(formData.get('anomaly_threshold'), null, 'Must not send fabricated threshold');
    assert.equal(formData.get('threshold'), null, 'Must not send fabricated threshold');
    assert.equal(formData.get('verdict'), null, 'Must not send fabricated verdict');

    // 5. Verify NORMAL response details
    assert.equal(result.inspection_id, 'insp_101');
    assert.equal(result.product_id, productId);
    assert.equal(result.verdict, 'NORMAL');
    assert.equal(result.anomaly_score, 18.4521);
    assert.equal(result.anomaly_threshold, 38.2255);
    assert.equal(result.margin, -19.7734);
    assert.equal(result.stage2_executed, false);
    assert.equal(result.defect_class, null, 'NORMAL response must not contain defect classification');
    assert.equal(result.classifier_confidence, null, 'NORMAL response must not contain classifier confidence');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('inspectProduct: DEFECTIVE response displays actual classifier result and probabilities', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    const mockDefectiveResponse = {
      id: 102,
      inspection_id: 'insp_102',
      product_id: 'prod_ready_bottle',
      created_at: '2026-10-03T18:05:00Z',
      timestamp: '2026-10-03T18:05:00Z',
      image_path: 'data/app_storage/inspections/insp_102/image.png',
      heatmap_path: 'data/app_storage/inspections/insp_102/heatmap.png',
      anomaly_score: 52.8812,
      anomaly_threshold: 38.2255,
      margin: 14.6557,
      verdict: 'DEFECTIVE',
      defect_class: 'Crack',
      classifier_confidence: 0.9412,
      classifier_probabilities: {
        Scratch: 0.0211,
        Crack: 0.9412,
        Discoloration: 0.0154,
        UNKNOWN: 0.0223,
      },
      stage2_executed: true,
      execution_time_ms: 78.4,
    };

    return new Response(JSON.stringify(mockDefectiveResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const { inspectProduct } = await import('../src/services/api.ts');
    const imageFile = createMockFile('defective_bottle.png');
    const result = await inspectProduct('prod_ready_bottle', imageFile, true);

    assert.equal(result.inspection_id, 'insp_102');
    assert.equal(result.verdict, 'DEFECTIVE');
    assert.equal(result.stage2_executed, true);
    assert.equal(result.defect_class, 'Crack');
    assert.equal(result.classifier_confidence, 0.9412);
    assert.ok(result.classifier_probabilities);
    assert.equal(result.classifier_probabilities['Crack'], 0.9412);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('inspectProduct: backend error does not fabricate a result', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    return new Response(
      JSON.stringify({ detail: 'Active calibration artifact not found for product.' }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  };

  try {
    const { inspectProduct } = await import('../src/services/api.ts');
    const imageFile = createMockFile('fail_query.png');

    await assert.rejects(
      async () => {
        await inspectProduct('uncalibrated_product', imageFile);
      },
      (err: Error) => {
        assert.match(err.message, /Active calibration artifact not found for product\./);
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Camera validation: blocks capture if product missing or camera inactive/unready', () => {
  // Missing product
  const noProduct = validateCameraInspectionPrerequisites('', true, true, { width: 1280, height: 720 });
  assert.equal(noProduct.valid, false);
  assert.match(noProduct.error || '', /calibrated \(READY\) product/);

  // Inactive camera
  const cameraInactive = validateCameraInspectionPrerequisites('prod_bottle', false, false, null);
  assert.equal(cameraInactive.valid, false);
  assert.match(cameraInactive.error || '', /Camera stream is not ready/);

  // Video not loaded / zero dimensions
  const zeroDims = validateCameraInspectionPrerequisites('prod_bottle', true, true, { width: 0, height: 0 });
  assert.equal(zeroDims.valid, false);
  assert.match(zeroDims.error || '', /invalid dimensions/);

  // Ready camera stream
  const ready = validateCameraInspectionPrerequisites('prod_bottle', true, true, { width: 1280, height: 720 });
  assert.equal(ready.valid, true);
  assert.equal(ready.error, undefined);
});

test('inspectProduct: sends raw Blob from camera frame with filename camera_frame.jpg', async () => {
  const capturedRequests: { url: string; method: string; body: FormData }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: input.toString(),
      method: init?.method || 'GET',
      body: init?.body as FormData,
    });

    const mockCameraInspectionResponse = {
      id: 201,
      inspection_id: 'insp_camera_201',
      product_id: 'prod_ready_bottle',
      created_at: '2026-10-21T13:00:00Z',
      timestamp: '2026-10-21T13:00:00Z',
      image_path: 'data/app_storage/inspections/insp_camera_201/image.jpg',
      heatmap_path: 'data/app_storage/inspections/insp_camera_201/heatmap.png',
      anomaly_score: 12.34,
      anomaly_threshold: 38.22,
      margin: -25.88,
      verdict: 'NORMAL',
      defect_class: null,
      classifier_confidence: null,
      classifier_probabilities: null,
      stage2_executed: false,
      execution_time_ms: 38.6,
    };

    return new Response(JSON.stringify(mockCameraInspectionResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const { inspectProduct } = await import('../src/services/api.ts');
    // Simulate a canvas JPEG frame blob
    const frameBlob = new Blob(['simulated_jpeg_camera_frame_binary'], { type: 'image/jpeg' });
    const productId = 'prod_ready_bottle';

    const result = await inspectProduct(productId, frameBlob, true);

    // 1. Verify correct endpoint and HTTP method
    assert.equal(capturedRequests.length, 1);
    const req = capturedRequests[0];
    assert.match(req.url, new RegExp(`/products/${productId}/inspect\\?save_heatmap=true$`));
    assert.equal(req.method, 'POST');

    // 2. Verify FormData contents
    const formData = req.body;
    assert.ok(formData instanceof FormData);
    const imagePart = formData.get('image');
    assert.ok(imagePart instanceof Blob);

    // 3. In Node/Browser, when a Blob is appended with a filename to FormData, it becomes a File with name 'camera_frame.jpg'
    // This is required so backend's save_uploaded_image validates extension .jpg!
    if (imagePart instanceof File) {
      assert.equal(imagePart.name, 'camera_frame.jpg');
    }
    assert.equal(formData.get('save_heatmap'), 'true');

    // 4. Verify result parsed
    assert.equal(result.inspection_id, 'insp_camera_201');
    assert.equal(result.verdict, 'NORMAL');
    assert.equal(result.stage2_executed, false);
    assert.equal(result.anomaly_score, 12.34);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// Logic unit: Dent detection prerequisites validation
function validateDentPrerequisites(file: File | Blob | null | undefined, confidenceThreshold: number) {
  if (!file) {
    return { valid: false, error: 'Please select or capture an image to inspect for dents.' };
  }
  if (confidenceThreshold < 0 || confidenceThreshold > 1) {
    return { valid: false, error: 'Confidence threshold must be between 0.0 and 1.0.' };
  }
  return { valid: true };
}

test('Dent detection prerequisites validation: file required and threshold in bounds', () => {
  assert.equal(validateDentPrerequisites(null, 0.30).valid, false);
  assert.equal(validateDentPrerequisites(createMockFile('test.jpg'), -0.1).valid, false);
  assert.equal(validateDentPrerequisites(createMockFile('test.jpg'), 1.1).valid, false);
  assert.equal(validateDentPrerequisites(createMockFile('test.jpg'), 0.30).valid, true);
});

test('Dent detection API: detectDents sends FormData with confidence_threshold and parses response', async () => {
  const originalFetch = globalThis.fetch;
  const capturedRequests: { url: string; method: string; body: FormData }[] = [];

  const mockDentResponse = {
    model_name: 'Faster R-CNN MobileNetV3-Large FPN',
    dataset: 'CR7-DET (Steel Surface)',
    confidence_threshold: 0.30,
    dents_detected: 2,
    boxes: [
      { x1: 52, y1: 88, x2: 120, y2: 160, confidence: 0.985, class_id: 2, label: 'Dent' },
      { x1: 300, y1: 210, x2: 380, y2: 290, confidence: 0.912, class_id: 2, label: 'Dent' },
    ],
    image_width: 500,
    image_height: 500,
    execution_time_ms: 45.2,
    message: 'Detected 2 dent(s) meeting confidence threshold >= 0.30.',
  };

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    capturedRequests.push({
      url: input.toString(),
      method: init?.method || 'GET',
      body: init?.body as FormData,
    });

    return new Response(JSON.stringify(mockDentResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const { detectDents } = await import('../src/services/api.ts');
    const mockFile = createMockFile('steel_sample.jpg');
    const result = await detectDents(mockFile, 0.30);

    // 1. Verify endpoint and method
    assert.equal(capturedRequests.length, 1);
    const req = capturedRequests[0];
    assert.match(req.url, /\/experimental\/dent\/detect\?confidence_threshold=0\.3/);
    assert.equal(req.method, 'POST');

    // 2. Verify FormData contents
    const formData = req.body;
    assert.ok(formData instanceof FormData);
    const imagePart = formData.get('image');
    assert.ok(imagePart instanceof Blob);
    assert.equal(formData.get('confidence_threshold'), '0.3');

    // 3. Verify response fields
    assert.equal(result.dents_detected, 2);
    assert.equal(result.boxes.length, 2);
    assert.equal(result.boxes[0].confidence, 0.985);
    assert.equal(result.boxes[0].label, 'Dent');
    assert.equal(result.dataset, 'CR7-DET (Steel Surface)');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Dent detection API: handles 0 detections correctly', async () => {
  const originalFetch = globalThis.fetch;

  const mockZeroDentsResponse = {
    model_name: 'Faster R-CNN MobileNetV3-Large FPN',
    dataset: 'CR7-DET (Steel Surface)',
    confidence_threshold: 0.50,
    dents_detected: 0,
    boxes: [],
    image_width: 500,
    image_height: 500,
    execution_time_ms: 32.1,
    message: 'No dents detected above confidence threshold 0.50.',
  };

  globalThis.fetch = async () => {
    return new Response(JSON.stringify(mockZeroDentsResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const { detectDents } = await import('../src/services/api.ts');
    const mockFile = createMockFile('clean_sample.jpg');
    const result = await detectDents(mockFile, 0.50);

    assert.equal(result.dents_detected, 0);
    assert.equal(result.boxes.length, 0);
    assert.equal(result.confidence_threshold, 0.50);
    assert.match(result.message, /No dents detected/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});


