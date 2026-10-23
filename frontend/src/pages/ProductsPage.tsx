import React, { useState, useEffect, useCallback } from 'react';
import { ProductList } from '../components/products/ProductList';
import { ProductDetailCard } from '../components/products/ProductDetailCard';
import { RegisterProductModal } from '../components/products/RegisterProductModal';
import { CalibrateModal } from '../components/products/CalibrateModal';
import { getProducts } from '../services/api';
import { Product, RunCalibrationResponse } from '../types';

export const ProductsPage: React.FC = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [isRegisterOpen, setIsRegisterOpen] = useState<boolean>(false);
  const [isCalibrateOpen, setIsCalibrateOpen] = useState<boolean>(false);

  const fetchProductList = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getProducts();
      setProducts(data);

      if (data.length > 0) {
        // Retain selection if valid, else default to first
        setSelectedProduct((prev) => {
          if (prev) {
            const found = data.find((p) => p.product_id === prev.product_id);
            if (found) return found;
          }
          return data[0];
        });
      } else {
        setSelectedProduct(null);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load product list.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProductList();
  }, [fetchProductList]);

  const handleProductCreated = (newProduct: Product) => {
    setProducts((prev) => [newProduct, ...prev]);
    setSelectedProduct(newProduct);
  };

  const handleCalibrationSuccess = (_calibration: RunCalibrationResponse) => {
    // Refresh product list to reflect updated READY status and active_calibration
    fetchProductList();
  };

  return (
    <div className="space-y-6">
      {/* Top Banner / Description */}
      <div className="bg-industrial-900/80 border border-industrial-800 rounded-xl p-5">
        <h1 className="text-base font-bold text-white tracking-wide">Product Profiles & Calibration Management</h1>
        <p className="text-xs text-industrial-400 mt-1 max-w-2xl">
          Register new industrial part types and configure product-specific PatchCore anomaly thresholds.
          A product must complete calibration and reach <span className="text-emerald-400 font-mono font-semibold">READY</span> status before inspection.
        </p>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5">
          <ProductList
            products={products}
            selectedProductId={selectedProduct?.product_id || null}
            onSelectProduct={setSelectedProduct}
            onOpenRegisterModal={() => setIsRegisterOpen(true)}
            loading={loading}
            error={error}
            onRetry={fetchProductList}
          />
        </div>

        <div className="lg:col-span-7">
          {selectedProduct ? (
            <ProductDetailCard
              product={selectedProduct}
              onOpenCalibrateModal={() => setIsCalibrateOpen(true)}
            />
          ) : (
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-10 text-center text-xs text-industrial-400 font-mono space-y-2">
              <div>Select a product from the list to view profile specs and active calibration details.</div>
            </div>
          )}
        </div>
      </div>

      {/* Modals */}
      <RegisterProductModal
        isOpen={isRegisterOpen}
        onClose={() => setIsRegisterOpen(false)}
        onProductCreated={handleProductCreated}
      />

      <CalibrateModal
        isOpen={isCalibrateOpen}
        product={selectedProduct}
        onClose={() => setIsCalibrateOpen(false)}
        onCalibrationSuccess={handleCalibrationSuccess}
      />
    </div>
  );
};
