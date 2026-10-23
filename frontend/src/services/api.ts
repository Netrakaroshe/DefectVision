/**
 * API service client for DefectVision.
 */

import type {
  Product,
  ProductCreate,
  CalibrationCreate,
  CalibrationResponse,
  RunCalibrationResponse,
  Inspection,
  InspectionListResponse,
  AnalyticsData,
  DentDetectionResult,
} from '../types/index.ts';

export const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8000';

export function getApiUrl(path: string): string {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

export function getInspectionImageUrl(inspectionId: string): string {
  return getApiUrl(`/inspections/${inspectionId}/image`);
}

export function getInspectionHeatmapUrl(inspectionId: string): string {
  return getApiUrl(`/inspections/${inspectionId}/heatmap`);
}

export async function checkBackendHealth(): Promise<{ status: string; service: string }> {
  const res = await fetch(getApiUrl('/health'));
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Product Management API Services
 */
export async function getProducts(): Promise<Product[]> {
  const res = await fetch(getApiUrl('/products'));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch products: HTTP ${res.status}`);
  }
  return res.json();
}

export async function createProduct(payload: ProductCreate): Promise<Product> {
  const res = await fetch(getApiUrl('/products'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to register product: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getProduct(productId: string): Promise<Product> {
  const res = await fetch(getApiUrl(`/products/${productId}`));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch product '${productId}': HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Product Calibration API Services
 */
export async function getProductCalibration(productId: string): Promise<CalibrationResponse> {
  const res = await fetch(getApiUrl(`/products/${productId}/calibration`));
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch calibration for '${productId}': HTTP ${res.status}`);
  }
  return res.json();
}

export async function calibrateProduct(
  productId: string,
  payload: CalibrationCreate
): Promise<CalibrationResponse> {
  const res = await fetch(getApiUrl(`/products/${productId}/calibration`), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Calibration failed: ${errorData.detail || `HTTP ${res.status}`}`);
  }
  return res.json();
}

/**
 * Execute real dynamic PatchCore calibration using 10–20 uploaded reference images.
 * Endpoint: POST /products/{product_id}/calibration/run
 */
export async function runProductCalibration(
  productId: string,
  files: File[]
): Promise<RunCalibrationResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const res = await fetch(getApiUrl(`/products/${productId}/calibration/run`), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Dynamic calibration failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Execute two-stage inspection on an uploaded image file for a READY product.
 * Endpoint: POST /products/{product_id}/inspect?save_heatmap=true
 */
export async function inspectProduct(
  productId: string,
  imageFile: File | Blob,
  saveHeatmap: boolean = true
): Promise<Inspection> {
  const formData = new FormData();
  if (imageFile instanceof File) {
    formData.append('image', imageFile);
  } else {
    formData.append('image', imageFile, 'camera_frame.jpg');
  }
  if (saveHeatmap) {
    formData.append('save_heatmap', 'true');
  }

  const query = saveHeatmap ? '?save_heatmap=true' : '';
  const res = await fetch(getApiUrl(`/products/${productId}/inspect${query}`), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Inspection failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Fetch paginated inspection history for a specific product.
 * Endpoint: GET /products/{product_id}/inspections
 */
export async function getProductInspections(
  productId: string,
  options?: { verdict?: string; limit?: number; offset?: number }
): Promise<InspectionListResponse> {
  const params = new URLSearchParams();
  if (options?.verdict) params.append('verdict', options.verdict);
  if (options?.limit !== undefined) params.append('limit', options.limit.toString());
  if (options?.offset !== undefined) params.append('offset', options.offset.toString());

  const queryString = params.toString() ? `?${params.toString()}` : '';
  const res = await fetch(getApiUrl(`/products/${productId}/inspections${queryString}`));

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch inspection history: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Retrieve a single inspection record by ID.
 * Endpoint: GET /inspections/{inspection_id}
 */
export async function getInspection(inspectionId: string): Promise<Inspection> {
  const res = await fetch(getApiUrl(`/inspections/${inspectionId}`));

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch inspection: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Retrieve database-aggregated inspection statistics.
 * Endpoint: GET /analytics
 */
export async function getAnalytics(productId?: string | null): Promise<AnalyticsData> {
  const query = productId ? `?product_id=${encodeURIComponent(productId)}` : '';
  const res = await fetch(getApiUrl(`/analytics${query}`));

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch analytics: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Run experimental Faster R-CNN Dent Detection on an image.
 * Endpoint: POST /experimental/dent/detect?confidence_threshold={thresh}
 */
export async function detectDents(
  imageFile: File | Blob,
  confidenceThreshold: number = 0.30
): Promise<DentDetectionResult> {
  const formData = new FormData();
  if (imageFile instanceof File) {
    formData.append('image', imageFile);
  } else {
    formData.append('image', imageFile, 'dent_frame.jpg');
  }
  formData.append('confidence_threshold', confidenceThreshold.toString());

  const res = await fetch(
    getApiUrl(`/experimental/dent/detect?confidence_threshold=${confidenceThreshold}`),
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Dent detection failed: HTTP ${res.status}`);
  }
  return res.json();
}


