/**
 * Core type definitions for DefectVision.
 * Matching FastAPI backend data models and schemas.
 */

export type ProductStatus = 'REGISTERED' | 'CALIBRATING' | 'READY';
export type CalibrationStatus = 'ACTIVE' | 'SUPERSEDED' | 'FAILED';
export type InspectionVerdict = 'NORMAL' | 'DEFECTIVE';
export type DefectClass = 'Scratch' | 'Crack' | 'Discoloration' | 'UNKNOWN';

export interface ProductCreate {
  product_id: string;
  name: string;
  description?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ActiveCalibrationSummary {
  id: number;
  threshold: number;
  calibration_method: string;
  patchcore_model_path: string;
  calibration_artifact_path: string | null;
  image_width: number;
  image_height: number;
  backbone: string;
  calibration_status: string;
}

export interface Product {
  id: number;
  product_id: string;
  name: string;
  description: string | null;
  status: ProductStatus;
  created_at: string;
  updated_at: string;
  active_calibration: ActiveCalibrationSummary | null;
  metadata?: Record<string, unknown> | null;
}

export interface CalibrationCreate {
  patchcore_model_path: string;
  artifact_path?: string | null;
  threshold?: number | null;
  normal_scores?: number[] | null;
  defective_scores?: number[] | null;
  calibration_method?: string | null;
  image_size?: [number, number];
  backbone?: string;
  metadata?: Record<string, unknown> | null;
}

export interface CalibrationResponse {
  id: number;
  product_id: string;
  calibration_artifact_path: string | null;
  patchcore_model_path: string;
  threshold: number;
  calibration_method: string;
  calibration_timestamp: string | null;
  reference_image_count: number | null;
  normal_sample_count: number | null;
  defective_sample_count: number | null;
  image_width: number;
  image_height: number;
  backbone: string;
  calibration_status: string;
  metadata: Record<string, unknown> | null;
  created_at?: string;
}

export interface NormalScoreStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  median?: number;
  iqr?: number;
  p05?: number;
  p25?: number;
  p75?: number;
  p95?: number;
  p99?: number;
  [key: string]: unknown;
}

export interface RunCalibrationResponse {
  product_id: string;
  calibration_id: number;
  threshold: number;
  calibration_method: string;
  reference_image_count: number;
  model_path: string;
  artifact_path: string;
  normal_score_stats: NormalScoreStats;
  warnings: string[];
  product_status: string;
}

export interface Inspection {
  id: number;
  inspection_id: string;
  product_id: string;
  image_path: string;
  timestamp: string;
  verdict: InspectionVerdict;
  anomaly_score: number;
  anomaly_threshold: number;
  margin: number;
  stage2_executed: boolean;
  defect_class: DefectClass | null;
  classifier_confidence: number | null;
  classifier_probabilities: Record<string, number> | null;
  heatmap_path: string | null;
  execution_time_ms: number | null;
  metadata?: Record<string, unknown> | null;
  created_at: string;
}

export interface InspectionListResponse {
  items: Inspection[];
  total: number;
  limit: number;
  offset: number;
}

export interface AnalyticsData {
  product_id: string | null;
  total_inspections: number;
  normal_count: number;
  defective_count: number;
  defect_rate: number;
  normal_rate: number;
  average_execution_time_ms: number;
  defect_class_counts: Record<DefectClass, number>;
}

export interface DentBoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface DentDetectionResult {
  dents_detected: number;
  boxes: DentBoundingBox[];
  confidence_threshold: number;
  image_width: number;
  image_height: number;
  execution_time_ms: number;
  model_name: string;
  dataset: string;
  experimental: boolean;
  message: string;
}

