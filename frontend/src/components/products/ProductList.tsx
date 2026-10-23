import React from 'react';
import { Plus, CheckCircle2, AlertTriangle, Boxes, ChevronRight } from 'lucide-react';
import { Product } from '../../types';
import { cn } from '../../lib/utils';

interface ProductListProps {
  products: Product[];
  selectedProductId: string | null;
  onSelectProduct: (product: Product) => void;
  onOpenRegisterModal: () => void;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export const ProductList: React.FC<ProductListProps> = ({
  products,
  selectedProductId,
  onSelectProduct,
  onOpenRegisterModal,
  loading,
  error,
  onRetry,
}) => {
  return (
    <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
        <div className="flex items-center space-x-2">
          <Boxes className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-semibold text-white">Registered Products</h3>
          <span className="text-xs px-2 py-0.5 rounded-full bg-industrial-800 text-industrial-300 font-mono">
            {products.length}
          </span>
        </div>

        <button
          onClick={onOpenRegisterModal}
          className="flex items-center space-x-1.5 px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-medium transition-colors shadow-sm"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Register Product</span>
        </button>
      </div>

      {loading && (
        <div className="py-8 text-center text-xs text-industrial-400 font-mono space-y-2">
          <div className="inline-block w-5 h-5 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
          <div>Loading registered products...</div>
        </div>
      )}

      {error && !loading && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 space-y-2">
          <div>Error loading products: {error}</div>
          <button
            onClick={onRetry}
            className="text-[11px] underline hover:text-white font-mono"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && products.length === 0 && (
        <div className="py-10 text-center space-y-3 bg-industrial-950/50 rounded-lg border border-industrial-800/80 p-6">
          <Boxes className="w-8 h-8 text-industrial-500 mx-auto" />
          <div className="text-sm font-medium text-industrial-300">No Products Registered</div>
          <p className="text-xs text-industrial-400 max-w-xs mx-auto">
            Register your first industrial product type to configure PatchCore anomaly calibration.
          </p>
          <button
            onClick={onOpenRegisterModal}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-medium transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Register First Product</span>
          </button>
        </div>
      )}

      {!loading && !error && products.length > 0 && (
        <div className="space-y-2">
          {products.map((product) => {
            const isSelected = selectedProductId === product.product_id;
            const isReady = product.status === 'READY';

            return (
              <div
                key={product.product_id}
                onClick={() => onSelectProduct(product)}
                className={cn(
                  'flex items-center justify-between p-3 rounded-lg cursor-pointer border transition-all duration-150',
                  isSelected
                    ? 'bg-industrial-800 border-sky-500/80 shadow-md'
                    : 'bg-industrial-950/60 border-industrial-800/80 hover:bg-industrial-800/60 hover:border-industrial-700'
                )}
              >
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-xs text-white truncate">{product.name}</span>
                    <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-industrial-900 border border-industrial-700 text-sky-400">
                      {product.product_id}
                    </span>
                  </div>
                  {product.active_calibration && (
                    <div className="text-[11px] font-mono text-industrial-400">
                      Threshold: <span className="text-emerald-400 font-semibold">{product.active_calibration.threshold.toFixed(4)}</span>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-3 shrink-0">
                  {isReady ? (
                    <span className="flex items-center space-x-1 text-[11px] font-semibold text-emerald-400 bg-emerald-950/80 px-2.5 py-0.5 rounded-full border border-emerald-800">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>READY</span>
                    </span>
                  ) : (
                    <span className="flex items-center space-x-1 text-[11px] font-semibold text-amber-400 bg-amber-950/80 px-2.5 py-0.5 rounded-full border border-amber-800">
                      <AlertTriangle className="w-3 h-3" />
                      <span>{product.status}</span>
                    </span>
                  )}
                  <ChevronRight className={cn('w-4 h-4 text-industrial-500', isSelected && 'text-sky-400')} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
