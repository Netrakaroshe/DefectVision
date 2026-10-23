import test from 'node:test';
import assert from 'node:assert/strict';
import { getAnalytics } from '../src/services/api.ts';
import type { AnalyticsData, DefectClass } from '../src/types/index.ts';

// Logic unit: Safe defect class breakdown calculation
function calculateDefectBreakdown(analytics: AnalyticsData) {
  const classes: DefectClass[] = ['Scratch', 'Crack', 'Discoloration', 'UNKNOWN'];
  const totalDefects = analytics.defective_count;

  return classes.map((cls) => {
    const count = analytics.defect_class_counts?.[cls] || 0;
    const percentage = totalDefects > 0 ? (count / totalDefects) * 100 : 0;
    return {
      className: cls,
      count,
      percentage,
    };
  });
}

test('analytics API request: global aggregate queries /analytics without product_id', async () => {
  const capturedRequests: { url: string; method: string }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: String(input),
      method: init?.method || 'GET',
    });

    const mockResponse: AnalyticsData = {
      product_id: null,
      total_inspections: 10,
      normal_count: 7,
      defective_count: 3,
      defect_rate: 0.3,
      normal_rate: 0.7,
      average_execution_time_ms: 842.5,
      defect_class_counts: {
        Scratch: 1,
        Crack: 1,
        Discoloration: 0,
        UNKNOWN: 1,
      },
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getAnalytics();
    assert.equal(capturedRequests.length, 1);
    assert.equal(capturedRequests[0].method, 'GET');
    assert.ok(
      capturedRequests[0].url.endsWith('/analytics'),
      `Expected URL to end with /analytics, got ${capturedRequests[0].url}`
    );
    assert.equal(result.total_inspections, 10);
    assert.equal(result.product_id, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('product selection: queries /analytics?product_id={id} for selected product', async () => {
  const capturedRequests: { url: string; method: string }[] = [];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    capturedRequests.push({
      url: String(input),
      method: init?.method || 'GET',
    });

    const mockResponse: AnalyticsData = {
      product_id: 'phase73a_real_test_bottle',
      total_inspections: 4,
      normal_count: 2,
      defective_count: 2,
      defect_rate: 0.5,
      normal_rate: 0.5,
      average_execution_time_ms: 915.13,
      defect_class_counts: {
        Scratch: 0,
        Crack: 0,
        Discoloration: 0,
        UNKNOWN: 2,
      },
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getAnalytics('phase73a_real_test_bottle');
    assert.equal(capturedRequests.length, 1);
    assert.equal(capturedRequests[0].method, 'GET');
    assert.ok(
      capturedRequests[0].url.includes('/analytics?product_id=phase73a_real_test_bottle'),
      `Expected URL to include product_id param, got ${capturedRequests[0].url}`
    );
    assert.equal(result.product_id, 'phase73a_real_test_bottle');
    assert.equal(result.total_inspections, 4);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('real response parsing: accurately maps all KPI fields and defect class counts', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    const mockResponse: AnalyticsData = {
      product_id: 'prod_gearbox_x',
      total_inspections: 120,
      normal_count: 90,
      defective_count: 30,
      defect_rate: 0.25,
      normal_rate: 0.75,
      average_execution_time_ms: 654.3,
      defect_class_counts: {
        Scratch: 15,
        Crack: 10,
        Discoloration: 3,
        UNKNOWN: 2,
      },
    };

    return new Response(JSON.stringify(mockResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getAnalytics('prod_gearbox_x');

    assert.equal(result.total_inspections, 120);
    assert.equal(result.normal_count, 90);
    assert.equal(result.defective_count, 30);
    assert.equal(result.defect_rate, 0.25);
    assert.equal(result.normal_rate, 0.75);
    assert.equal(result.average_execution_time_ms, 654.3);

    // Defect breakdown calculations
    const breakdown = calculateDefectBreakdown(result);
    assert.equal(breakdown.length, 4);

    const scratch = breakdown.find((b) => b.className === 'Scratch');
    assert.ok(scratch);
    assert.equal(scratch.count, 15);
    assert.equal(scratch.percentage, 50); // 15 / 30 = 50%

    const crack = breakdown.find((b) => b.className === 'Crack');
    assert.ok(crack);
    assert.equal(crack.count, 10);
    assert.ok(Math.abs(crack.percentage - 33.33) < 0.1);

    const discoloration = breakdown.find((b) => b.className === 'Discoloration');
    assert.ok(discoloration);
    assert.equal(discoloration.count, 3);
    assert.equal(discoloration.percentage, 10); // 3 / 30 = 10%

    const unknown = breakdown.find((b) => b.className === 'UNKNOWN');
    assert.ok(unknown);
    assert.equal(unknown.count, 2);
    assert.ok(Math.abs(unknown.percentage - 6.67) < 0.1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('empty analytics state: handles zero inspections without divide-by-zero errors', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    const mockEmptyResponse: AnalyticsData = {
      product_id: 'new_uninspected_product',
      total_inspections: 0,
      normal_count: 0,
      defective_count: 0,
      defect_rate: 0.0,
      normal_rate: 0.0,
      average_execution_time_ms: 0.0,
      defect_class_counts: {
        Scratch: 0,
        Crack: 0,
        Discoloration: 0,
        UNKNOWN: 0,
      },
    };

    return new Response(JSON.stringify(mockEmptyResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getAnalytics('new_uninspected_product');
    assert.equal(result.total_inspections, 0);
    assert.equal(result.normal_count, 0);
    assert.equal(result.defective_count, 0);
    assert.equal(result.defect_rate, 0.0);
    assert.equal(result.normal_rate, 0.0);

    const breakdown = calculateDefectBreakdown(result);
    for (const item of breakdown) {
      assert.equal(item.count, 0);
      assert.equal(item.percentage, 0); // Must be 0, not NaN or Infinity
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('backend error state: throws clear error and does NOT fabricate fallback statistics', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    return new Response(
      JSON.stringify({ detail: 'Internal analytics aggregation error' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  };

  try {
    await assert.rejects(
      async () => {
        await getAnalytics('failed_product');
      },
      (err: Error) => {
        assert.ok(
          err.message.includes('500') || err.message.includes('Internal analytics aggregation error'),
          `Expected 500 or detail in error message, got: ${err.message}`
        );
        return true;
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('no fabricated metrics: ensures accuracy, precision, recall, F1, and AUROC are not present', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (): Promise<Response> => {
    const realBackendResponse = {
      product_id: 'phase73a_real_test_bottle',
      total_inspections: 4,
      normal_count: 2,
      defective_count: 2,
      defect_rate: 0.5,
      normal_rate: 0.5,
      average_execution_time_ms: 915.13,
      defect_class_counts: {
        Scratch: 0,
        Crack: 0,
        Discoloration: 0,
        UNKNOWN: 2,
      },
    };

    return new Response(JSON.stringify(realBackendResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await getAnalytics('phase73a_real_test_bottle');
    const record = result as unknown as Record<string, unknown>;

    // Strict validation: fabricated ML evaluation metrics must NOT exist
    assert.equal(record.accuracy, undefined);
    assert.equal(record.precision, undefined);
    assert.equal(record.recall, undefined);
    assert.equal(record.f1_score, undefined);
    assert.equal(record.auroc, undefined);
    assert.equal(record.roc_auc, undefined);

    // Only genuine operational database counts and rates are present
    assert.equal(typeof result.total_inspections, 'number');
    assert.equal(typeof result.normal_count, 'number');
    assert.equal(typeof result.defective_count, 'number');
    assert.equal(typeof result.defect_rate, 'number');
    assert.equal(typeof result.normal_rate, 'number');
    assert.equal(typeof result.average_execution_time_ms, 'number');
    assert.equal(typeof result.defect_class_counts, 'object');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
