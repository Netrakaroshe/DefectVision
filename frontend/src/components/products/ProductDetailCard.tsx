import React from 'react';
import { Sliders, CheckCircle2, Clock, AlertTriangle, Layers, Database } from 'lucide-react';
import { Product } from '../../types';

interface ProductDetailCardProps {
  product: Product;
  onOpenCalibrateModal: (product: Product) => void;
}

export const ProductDetailCard: React.FC<ProductDetailCardProps> = ({
  product,
  onOpenCalibrateModal,
}) => {
  const isReady = product.status === 'READY';
  const cal = product.active_calibration;

  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-4">
      {/* Header Info */}
      <div className="flex items-start justify-between border-b border-industrial-800 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-base font-bold text-white tracking-wide">{product.name}</h2>
            <span className="font-mono text-xs px-2 py-0.5 rounded bg-industrial-950 border border-industrial-800 text-sky-400">
              {product.product_id}
            </span>
          </div>
          {product.description && (
            <p className="text-xs text-industrial-400 mt-1">{product.description}</p>
          )}
        </div>

        {/* Status Badge */}
        <div>
          {isReady ? (
            <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 border border-emerald-800 text-emerald-300">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>READY</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-950 border border-amber-800 text-amber-300">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>{product.status}</span>
            </span>
          )}
        </div>
      </div>

      {/* Active Calibration Details */}
      {isReady && cal ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-industrial-400 font-mono">
            <span className="flex items-center space-x-1">
              <Sliders className="w-3.5 h-3.5 text-sky-400" />
              <span>Active Threshold Calibration</span>
            </span>
            <span className="text-emerald-400 font-semibold">{cal.calibration_status}</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3">
              <span className="text-[11px] text-industrial-400 font-mono block">Threshold</span>
              <span className="text-sm font-bold font-mono text-emerald-400">
                {cal.threshold.toFixed(4)}
              </span>
            </div>

            <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3">
              <span className="text-[11px] text-industrial-400 font-mono block">Method</span>
              <span className="text-xs font-medium text-white truncate block capitalize">
                {cal.calibration_method}
              </span>
            </div>

            <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3">
              <span className="text-[11px] text-industrial-400 font-mono block">Image Size</span>
              <span className="text-xs font-medium font-mono text-white">
                {cal.image_width} &times; {cal.image_height}
              </span>
            </div>

            <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3">
              <span className="text-[11px] text-industrial-400 font-mono block">Backbone</span>
              <span className="text-xs font-medium font-mono text-white truncate block">
                {cal.backbone}
              </span>
            </div>
          </div>

          <div className="bg-industrial-950 border border-industrial-800 rounded-lg p-3 space-y-1 font-mono text-[11px] text-industrial-400">
            <div className="flex items-center space-x-2">
              <Database className="w-3.5 h-3.5 text-industrial-500 shrink-0" />
              <span className="truncate">Model: <span className="text-industrial-300">{cal.patchcore_model_path}</span></span>
            </div>
            {cal.calibration_artifact_path && (
              <div className="flex items-center space-x-2">
                <Layers className="w-3.5 h-3.5 text-industrial-500 shrink-0" />
                <span className="truncate">Artifact: <span className="text-industrial-300">{cal.calibration_artifact_path}</span></span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-industrial-950/70 border border-industrial-800/80 rounded-xl p-4 text-center space-y-3">
          <div className="text-xs text-industrial-400">
            This product is registered but has no active anomaly threshold calibration.
          </div>
          <button
            onClick={() => onOpenCalibrateModal(product)}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white text-xs font-medium rounded-lg transition-colors shadow"
          >
            <Sliders className="w-4 h-4" />
            <span>Calibrate Threshold</span>
          </button>
        </div>
      )}

      {/* Footer Meta & Actions */}
      <div className="flex items-center justify-between border-t border-industrial-800/80 pt-3 text-[11px] text-industrial-500 font-mono">
        <div className="flex items-center space-x-1">
          <Clock className="w-3 h-3" />
          <span>Created: {new Date(product.created_at).toLocaleDateString()}</span>
        </div>

        {isReady && (
          <button
            onClick={() => onOpenCalibrateModal(product)}
            className="text-xs text-sky-400 hover:text-sky-300 transition-colors font-sans flex items-center space-x-1"
          >
            <Sliders className="w-3 h-3" />
            <span>Recalibrate</span>
          </button>
        )}
      </div>
    </div>
  );
};
