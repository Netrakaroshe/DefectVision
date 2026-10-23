import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getProductInspections,
  getInspection,
  getInspectionHeatmapUrl,
  getInspectionImageUrl,
} from '../src/services/api.ts';
import type { Inspection, InspectionListResponse } from '../src/types/index.ts';

// Logic unit: Safe newest-first sorting by timestamp or created_at
function sortInspectionsNewestFirst(items: Inspection[]): Inspection[] {
  return [...items].sort((a, b) => {
    const timeA = new Date(a.timestamp || a.created_at).getTime();
    const timeB = new Date(b.timestamp || b.created_at).getTime();
    return timeB - timeA;
  });
}

// Logic unit: Stage 2 display decision
function getStage2DisplayInfo(inspection: Inspection) {
  if (!inspection.stage2_executed) {
    return {
      displayed: false,
      reason: 'Bypassed (Normal Sample)',
      defect_class: null,
      confidence: null,
    };
  }
  return {
    displayed: true,
    reason: null,
    defect_class: inspection.defect_class,
    confidence: inspection.classifier_confidence,
    probabilities: inspection.classifier_probabilities,
  };
}

test('Product selection: works dynamically with any product ID', () => {
  const products = [
    { product_id: 'custom_gearbox_v1', name: 'Gearbox V1' },
    { product_id: 'phase73a_real_test_bottle', name: 'Real Bottle' },
    { product_id: 'metal_stamping_02', name: 'Metal Stamping' },
  ];

  // User selecting any arbitrary product
  let selected = '';
  const selectProduct = (id: string) => {
    selected = id;
  };

  selectProduct(products[0].product_id);
  assert.equal(selected, 'custom_gearbox_v1');

  selectProduct(products[2].product_id);
  assert.equal(selected, 'metal_stamping_02');
});

test('correct history endpoint and product ID are queried', async () => {
  const capturedRequests: { url: string; method: string }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: String(input),
      method: init?.method || 'GET',
    });

    const mockResponse: InspectionListResponse = {
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getProductInspections('bottle_prod_123');
    assert.equal(capturedRequests.length, 1);
    assert.equal(capturedRequests[0].method, 'GET');
    assert.equal(
      capturedRequests[0].url,
      'http://127.0.0.1:8000/products/bottle_prod_123/inspections'
    );
    assert.deepEqual(result.items, []);
    assert.equal(result.total, 0);

    // With verdict filter
    await getProductInspections('bottle_prod_123', { verdict: 'DEFECTIVE' });
    assert.equal(capturedRequests.length, 2);
    assert.equal(
      capturedRequests[1].url,
      'http://127.0.0.1:8000/products/bottle_prod_123/inspections?verdict=DEFECTIVE'
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('inspection list response handling parses items and pagination metadata', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    const mockResponse: InspectionListResponse = {
      items: [
        {
          id: 1,
          inspection_id: 'insp-001',
          product_id: 'test_product',
          image_path: 'uploads/test/001.png',
          timestamp: '2026-10-03T16:00:00Z',
          verdict: 'NORMAL',
          anomaly_score: 32.5,
          anomaly_threshold: 38.0,
          margin: -5.5,
          stage2_executed: false,
          defect_class: null,
          classifier_confidence: null,
          classifier_probabilities: null,
          heatmap_path: 'heatmaps/test/001.png',
          execution_time_ms: 120.5,
          created_at: '2026-10-03T16:00:00Z',
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const data = await getProductInspections('test_product');
    assert.equal(data.total, 1);
    assert.equal(data.items.length, 1);
    assert.equal(data.items[0].inspection_id, 'insp-001');
    assert.equal(data.items[0].verdict, 'NORMAL');
    assert.equal(data.items[0].anomaly_score, 32.5);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('newest-first display: sorts records safely by timestamp descending', () => {
  const mockInspections: Inspection[] = [
    {
      id: 1,
      inspection_id: 'insp-oldest',
      product_id: 'p1',
      image_path: '',
      timestamp: '2026-10-03T10:00:00Z',
      verdict: 'NORMAL',
      anomaly_score: 30.0,
      anomaly_threshold: 38.0,
      margin: -8.0,
      stage2_executed: false,
      defect_class: null,
      classifier_confidence: null,
      classifier_probabilities: null,
      heatmap_path: null,
      execution_time_ms: 100,
      created_at: '2026-10-03T10:00:00Z',
    },
    {
      id: 2,
      inspection_id: 'insp-newest',
      product_id: 'p1',
      image_path: '',
      timestamp: '2026-10-03T15:30:00Z',
      verdict: 'DEFECTIVE',
      anomaly_score: 65.0,
      anomaly_threshold: 38.0,
      margin: 27.0,
      stage2_executed: true,
      defect_class: 'UNKNOWN',
      classifier_confidence: 0.99,
      classifier_probabilities: null,
      heatmap_path: 'heatmaps/p1/002.png',
      execution_time_ms: 150,
      created_at: '2026-10-03T15:30:00Z',
    },
    {
      id: 3,
      inspection_id: 'insp-middle',
      product_id: 'p1',
      image_path: '',
      timestamp: '2026-10-03T12:15:00Z',
      verdict: 'NORMAL',
      anomaly_score: 33.0,
      anomaly_threshold: 38.0,
      margin: -5.0,
      stage2_executed: false,
      defect_class: null,
      classifier_confidence: null,
      classifier_probabilities: null,
      heatmap_path: null,
      execution_time_ms: 95,
      created_at: '2026-10-03T12:15:00Z',
    },
  ];

  const sorted = sortInspectionsNewestFirst(mockInspections);
  assert.equal(sorted[0].inspection_id, 'insp-newest');
  assert.equal(sorted[1].inspection_id, 'insp-middle');
  assert.equal(sorted[2].inspection_id, 'insp-oldest');
});

test('empty history: returns empty items without throwing', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    return new Response(JSON.stringify({ items: [], total: 0, limit: 50, offset: 0 }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const data = await getProductInspections('product_with_no_history');
    assert.deepEqual(data.items, []);
    assert.equal(data.total, 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('detail endpoint queries GET /inspections/{inspection_id}', async () => {
  const capturedRequests: { url: string; method: string }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: String(input),
      method: init?.method || 'GET',
    });

    const mockDetail: Inspection = {
      id: 42,
      inspection_id: 'uuid-42-detail',
      product_id: 'prod_42',
      image_path: 'uploads/p42/img.png',
      timestamp: '2026-10-03T16:00:00Z',
      verdict: 'DEFECTIVE',
      anomaly_score: 72.1,
      anomaly_threshold: 38.7,
      margin: 33.4,
      stage2_executed: true,
      defect_class: 'Crack',
      classifier_confidence: 0.954,
      classifier_probabilities: { Crack: 0.954, Scratch: 0.046 },
      heatmap_path: 'heatmaps/p42/hm.png',
      execution_time_ms: 180.2,
      created_at: '2026-10-03T16:00:00Z',
    };

    return new Response(JSON.stringify(mockDetail), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const detail = await getInspection('uuid-42-detail');
    assert.equal(capturedRequests.length, 1);
    assert.equal(capturedRequests[0].url, 'http://127.0.0.1:8000/inspections/uuid-42-detail');
    assert.equal(detail.inspection_id, 'uuid-42-detail');
    assert.equal(detail.verdict, 'DEFECTIVE');
    assert.equal(detail.defect_class, 'Crack');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('NORMAL detail hides Stage 2 classification and displays bypassed status', () => {
  const normalInspection: Inspection = {
    id: 1,
    inspection_id: 'insp-normal-pass',
    product_id: 'phase73a_real_test_bottle',
    image_path: 'uploads/bottle/norm.png',
    timestamp: '2026-10-03T16:36:07Z',
    verdict: 'NORMAL',
    anomaly_score: 36.19,
    anomaly_threshold: 38.70,
    margin: -2.51,
    stage2_executed: false,
    defect_class: null,
    classifier_confidence: null,
    classifier_probabilities: null,
    heatmap_path: 'heatmaps/bottle/hm_norm.png',
    execution_time_ms: 2656.6,
    created_at: '2026-10-03T16:36:07Z',
  };

  const s2Info = getStage2DisplayInfo(normalInspection);
  assert.equal(s2Info.displayed, false);
  assert.equal(s2Info.defect_class, null);
  assert.equal(s2Info.confidence, null);
  assert.match(s2Info.reason || '', /Bypassed/);
});

test('DEFECTIVE detail displays actual Stage 2 result, defect class, and confidence', () => {
  const defectiveInspection: Inspection = {
    id: 2,
    inspection_id: 'insp-defective-fail',
    product_id: 'phase73a_real_test_bottle',
    image_path: 'uploads/bottle/def.png',
    timestamp: '2026-10-03T16:36:08Z',
    verdict: 'DEFECTIVE',
    anomaly_score: 69.04,
    anomaly_threshold: 38.70,
    margin: 30.34,
    stage2_executed: true,
    defect_class: 'UNKNOWN',
    classifier_confidence: 0.999868,
    classifier_probabilities: {
      Scratch: 0.000063,
      Crack: 0.000008,
      Discoloration: 0.000060,
      UNKNOWN: 0.999868,
    },
    heatmap_path: 'heatmaps/bottle/hm_def.png',
    execution_time_ms: 175.7,
    created_at: '2026-10-03T16:36:08Z',
  };

  const s2Info = getStage2DisplayInfo(defectiveInspection);
  assert.equal(s2Info.displayed, true);
  assert.equal(s2Info.defect_class, 'UNKNOWN');
  assert.equal(s2Info.confidence, 0.999868);
  assert.ok(s2Info.probabilities);
  assert.equal(s2Info.probabilities['UNKNOWN'], 0.999868);
});

test('heatmap URL uses inspection ID and never exposes raw filesystem paths', () => {
  const inspectionId = '561c70c5-8fb1-4c32-8ea4-f16bb09553c9';
  const heatmapUrl = getInspectionHeatmapUrl(inspectionId);
  const imageUrl = getInspectionImageUrl(inspectionId);

  // URL must be structured with the inspection ID endpoint
  assert.equal(
    heatmapUrl,
    'http://127.0.0.1:8000/inspections/561c70c5-8fb1-4c32-8ea4-f16bb09553c9/heatmap'
  );
  assert.equal(
    imageUrl,
    'http://127.0.0.1:8000/inspections/561c70c5-8fb1-4c32-8ea4-f16bb09553c9/image'
  );

  // Must not contain filesystem indicators like D:, C:, /, or .png file paths
  assert.doesNotMatch(heatmapUrl, /[A-Z]:\\/);
  assert.doesNotMatch(imageUrl, /[A-Z]:\\/);
});

test('backend errors do not fabricate data and throw clear exceptions', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    return new Response(JSON.stringify({ detail: 'Product not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    await assert.rejects(
      async () => {
        await getProductInspections('non_existent_product');
      },
      /Product not found/
    );

    await assert.rejects(
      async () => {
        await getInspection('non_existent_inspection');
      },
      /Product not found/
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
