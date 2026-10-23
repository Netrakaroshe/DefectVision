import React, { useState } from 'react';
import { X, Plus, AlertCircle, Loader2 } from 'lucide-react';
import { createProduct } from '../../services/api';
import { Product } from '../../types';

interface RegisterProductModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProductCreated: (product: Product) => void;
}

export const RegisterProductModal: React.FC<RegisterProductModalProps> = ({
  isOpen,
  onClose,
  onProductCreated,
}) => {
  const [productId, setProductId] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!productId.trim() || !name.trim()) {
      setError('Product ID and Name are required.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const newProduct = await createProduct({
        product_id: productId.trim().toLowerCase().replace(/\s+/g, '_'),
        name: name.trim(),
        description: description.trim() || undefined,
      });

      onProductCreated(newProduct);
      setProductId('');
      setName('');
      setDescription('');
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Product registration failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-industrial-900 border border-industrial-700 rounded-xl max-w-md w-full p-6 shadow-2xl space-y-5">
        <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
          <div className="flex items-center space-x-2">
            <Plus className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-semibold text-white">Register Industrial Product</h2>
          </div>
          <button
            onClick={onClose}
            className="text-industrial-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-mono text-industrial-300 mb-1">
              Product ID *
            </label>
            <input
              type="text"
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
              placeholder="e.g. widget_a, bearing_102, pcb_unit"
              className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-sky-500 font-mono"
              required
            />
            <p className="text-[11px] text-industrial-500 mt-1">
              Unique identifier used in API routes and storage paths.
            </p>
          </div>

          <div>
            <label className="block text-xs font-mono text-industrial-300 mb-1">
              Product Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Glass Bottle Line A"
              className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-sky-500"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-mono text-industrial-300 mb-1">
              Description / Notes
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional notes or specs"
              rows={3}
              className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-sm text-white placeholder-industrial-500 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-3 border-t border-industrial-800">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-medium text-industrial-400 hover:text-white transition-colors"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex items-center space-x-2 px-4 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium transition-colors"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Registering...</span>
                </>
              ) : (
                <span>Register Product</span>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
