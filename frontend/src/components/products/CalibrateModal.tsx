import React, { useState, useEffect, useMemo } from 'react';
import {
  X,
  Sliders,
  AlertCircle,
  Upload,
  CheckCircle2,
  Loader2,
  Trash2,
  AlertTriangle,
  Layers,
  Database,
} from 'lucide-react';
import { runProductCalibration } from '../../services/api';
import { Product, RunCalibrationResponse } from '../../types';

interface CalibrateModalProps {
  isOpen: boolean;
  product: Product | null;
  onClose: () => void;
  onCalibrationSuccess: (calibration: RunCalibrationResponse) => void;
}

interface FileWithPreview {
  file: File;
  previewUrl: string;
}

export const CalibrateModal: React.FC<CalibrateModalProps> = ({
  isOpen,
  product,
  onClose,
  onCalibrationSuccess,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successResult, setSuccessResult] = useState<RunCalibrationResponse | null>(null);

  // Generate and manage preview object URLs
  const filesWithPreviews: FileWithPreview[] = useMemo(() => {
    return selectedFiles.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
    }));
  }, [selectedFiles]);

  // Clean up object URLs on change and unmount
  useEffect(() => {
    return () => {
      filesWithPreviews.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    };
  }, [filesWithPreviews]);

  // Reset state when modal closes or product changes
  useEffect(() => {
    if (!isOpen) {
      setSelectedFiles([]);
      setError(null);
      setSuccessResult(null);
      setLoading(false);
    }
  }, [isOpen, product?.product_id]);

  if (!isOpen || !product) return null;

  const count = selectedFiles.length;
  const isTooFew = count < 10;
  const isTooMany = count > 20;
  const isValidCount = count >= 10 && count <= 20;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const newFiles = Array.from(e.target.files).filter((file) =>
      file.type.startsWith('image/')
    );
    setSelectedFiles((prev) => [...prev, ...newFiles]);
    setError(null);
  };

  const handleRemoveFile = (index: number) => {
    if (loading) return;
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setError(null);
  };

  const handleClearAll = () => {
    if (loading) return;
    setSelectedFiles([]);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;

    if (count < 10) {
      setError(`At least 10 reference images are required (currently selected: ${count}).`);
      return;
    }
    if (count > 20) {
      setError(`At most 20 reference images are accepted (currently selected: ${count}).`);
      return;
    }

    setLoading(true);
    setError(null);
    setSuccessResult(null);

    try {
      const result = await runProductCalibration(product.product_id, selectedFiles);
      setSuccessResult(result);
      onCalibrationSuccess(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Calibration failed on the backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleFinish = () => {
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-industrial-900 border border-industrial-700 rounded-xl max-w-xl w-full p-6 shadow-2xl space-y-5 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-white">
              Calibrate Product Threshold:{' '}
              <span className="font-mono text-sky-300">{product.product_id}</span>
            </h2>
          </div>
          <button
            onClick={onClose}
            disabled={loading}
            className="text-industrial-400 hover:text-white transition-colors disabled:opacity-40"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Guidance notice */}
        {!successResult && (
          <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3 text-xs text-industrial-300 space-y-1">
            <div className="font-semibold text-sky-400">Dynamic PatchCore Calibration</div>
            <p>
              Upload <strong>10–20 defect-free reference images</strong>. The backend fits a
              dedicated PatchCore memory bank, determines the anomaly score distribution,
              calibrates the decision threshold (<code className="text-sky-300">mean + 3σ</code>), and marks the product <strong className="text-emerald-400">READY</strong>.
            </p>
          </div>
        )}

        {/* Error notification */}
        {error && (
          <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Success view */}
        {successResult ? (
          <div className="space-y-4">
            <div className="bg-emerald-950/60 border border-emerald-800 rounded-lg p-4 text-xs text-emerald-300 space-y-3">
              <div className="flex items-center space-x-2 font-semibold text-emerald-400 text-sm">
                <CheckCircle2 className="w-5 h-5" />
                <span>Calibration Successful — Product is READY</span>
              </div>
              <p className="text-industrial-300">
                A new PatchCore model checkpoint and threshold artifact were created and validated.
              </p>

              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-2 border-t border-emerald-900/60">
                <div className="bg-industrial-950/80 p-2 rounded border border-emerald-900/50">
                  <span className="text-industrial-400 block text-[10px]">Calibrated Threshold</span>
                  <span className="text-white font-bold text-sm">
                    {successResult.threshold.toFixed(4)}
                  </span>
                </div>
                <div className="bg-industrial-950/80 p-2 rounded border border-emerald-900/50">
                  <span className="text-industrial-400 block text-[10px]">Method</span>
                  <span className="text-white font-medium capitalize">
                    {successResult.calibration_method}
                  </span>
                </div>
                <div className="bg-industrial-950/80 p-2 rounded border border-emerald-900/50">
                  <span className="text-industrial-400 block text-[10px]">Reference Samples</span>
                  <span className="text-white font-medium">
                    {successResult.reference_image_count} images
                  </span>
                </div>
                <div className="bg-industrial-950/80 p-2 rounded border border-emerald-900/50">
                  <span className="text-industrial-400 block text-[10px]">Status</span>
                  <span className="text-emerald-400 font-semibold">
                    {successResult.product_status}
                  </span>
                </div>
              </div>

              {successResult.normal_score_stats && (
                <div className="bg-industrial-950/80 p-2.5 rounded border border-emerald-900/50 font-mono text-[11px] space-y-1">
                  <span className="text-industrial-400 text-[10px] block font-sans">
                    Reference Score Distribution:
                  </span>
                  <div className="grid grid-cols-4 gap-2 text-white">
                    <div>Mean: <span className="font-bold">{successResult.normal_score_stats.mean?.toFixed(2)}</span></div>
                    <div>Std: <span className="font-bold">{successResult.normal_score_stats.std?.toFixed(2)}</span></div>
                    <div>Min: <span className="font-bold">{successResult.normal_score_stats.min?.toFixed(2)}</span></div>
                    <div>Max: <span className="font-bold">{successResult.normal_score_stats.max?.toFixed(2)}</span></div>
                  </div>
                </div>
              )}

              <div className="space-y-1 font-mono text-[10px] text-industrial-400 pt-1">
                <div className="flex items-center space-x-1 truncate">
                  <Database className="w-3 h-3 text-industrial-500 shrink-0" />
                  <span className="truncate">Model: {successResult.model_path}</span>
                </div>
                <div className="flex items-center space-x-1 truncate">
                  <Layers className="w-3 h-3 text-industrial-500 shrink-0" />
                  <span className="truncate">Artifact: {successResult.artifact_path}</span>
                </div>
              </div>

              {successResult.warnings && successResult.warnings.length > 0 && (
                <div className="bg-amber-950/40 border border-amber-800/60 rounded p-2 text-[11px] text-amber-300 flex items-start space-x-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <span>{successResult.warnings[0]}</span>
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2">
              <button
                type="button"
                onClick={handleFinish}
                className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-medium transition-colors shadow"
              >
                Close & View Product
              </button>
            </div>
          </div>
        ) : (
          /* Form for image selection and calibration */
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Reference Image Selection */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-industrial-300">
                  Reference Images (10–20 required)
                </label>
                <div className="flex items-center space-x-2">
                  {selectedFiles.length > 0 && (
                    <button
                      type="button"
                      onClick={handleClearAll}
                      disabled={loading}
                      className="text-[11px] font-mono text-industrial-400 hover:text-red-400 transition-colors disabled:opacity-40"
                    >
                      Clear all
                    </button>
                  )}
                  <span
                    className={`text-xs px-2.5 py-0.5 rounded font-mono ${
                      isValidCount
                        ? 'bg-emerald-950 border border-emerald-800 text-emerald-400'
                        : 'bg-amber-950 border border-amber-800 text-amber-400'
                    }`}
                  >
                    {count} / 10–20 selected
                  </span>
                </div>
              </div>

              {/* Upload Dropzone */}
              <div className="border-2 border-dashed border-industrial-700 hover:border-sky-500/60 rounded-xl p-4 text-center cursor-pointer transition-colors bg-industrial-950/50">
                <input
                  type="file"
                  multiple
                  accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="hidden"
                  id="reference-image-input"
                />
                <label
                  htmlFor="reference-image-input"
                  className={`block space-y-2 ${loading ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
                >
                  <Upload className="w-6 h-6 text-industrial-400 mx-auto" />
                  <div className="text-xs text-industrial-300">
                    Click to select reference image files from disk
                  </div>
                  <div className="text-[11px] text-industrial-500 font-mono">
                    PNG, JPG, WEBP, BMP — Exactly 10 to 20 defect-free samples
                  </div>
                </label>
              </div>

              {/* Validation helper messages */}
              {isTooFew && count > 0 && (
                <p className="text-[11px] text-amber-400 font-mono flex items-center space-x-1">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>Need at least {10 - count} more reference image(s) to reach the minimum of 10.</span>
                </p>
              )}
              {isTooMany && (
                <p className="text-[11px] text-red-400 font-mono flex items-center space-x-1">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>Exceeds limit by {count - 20} image(s). Maximum is 20.</span>
                </p>
              )}
              {isValidCount && (
                <p className="text-[11px] text-emerald-400 font-mono flex items-center space-x-1">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  <span>Valid count ({count} images). Ready to run dynamic calibration.</span>
                </p>
              )}

              {/* Selected Files Preview List */}
              {filesWithPreviews.length > 0 && (
                <div className="max-h-44 overflow-y-auto space-y-1.5 p-2 bg-industrial-950 rounded-lg border border-industrial-800">
                  {filesWithPreviews.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs font-mono text-industrial-300 px-2 py-1.5 bg-industrial-900 rounded border border-industrial-800"
                    >
                      <div className="flex items-center space-x-2.5 truncate pr-2">
                        <img
                          src={item.previewUrl}
                          alt={item.file.name}
                          className="w-7 h-7 object-cover rounded border border-industrial-700 shrink-0 bg-black"
                        />
                        <span className="truncate">{item.file.name}</span>
                        <span className="text-[10px] text-industrial-500 shrink-0">
                          ({(item.file.size / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRemoveFile(idx)}
                        disabled={loading}
                        className="text-industrial-500 hover:text-red-400 transition-colors disabled:opacity-40 shrink-0 ml-2"
                        title="Remove image"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* In-Flight Calibration Spinner Box */}
            {loading && (
              <div className="bg-sky-950/40 border border-sky-800/80 rounded-xl p-4 text-center space-y-2">
                <Loader2 className="w-6 h-6 animate-spin text-sky-400 mx-auto" />
                <div className="text-xs font-semibold text-white">
                  Executing Dynamic PatchCore Calibration...
                </div>
                <p className="text-[11px] text-industrial-300 max-w-sm mx-auto">
                  Extracting visual features, subsampling memory bank, and calculating statistical decision threshold. This may take 30–60 seconds.
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center justify-end space-x-3 pt-3 border-t border-industrial-800">
              <button
                type="button"
                onClick={onClose}
                disabled={loading}
                className="px-4 py-2 text-xs font-medium text-industrial-400 hover:text-white transition-colors disabled:opacity-40"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading || !isValidCount}
                className="flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white rounded-lg text-xs font-medium transition-colors shadow"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Calibrating...</span>
                  </>
                ) : (
                  <span>
                    Run Calibration {isValidCount ? `(${count} images)` : ''}
                  </span>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
