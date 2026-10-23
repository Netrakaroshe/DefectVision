import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  BarChart3,
  CheckCircle2,
  AlertOctagon,
  AlertTriangle,
  Loader2,
  RotateCcw,
  Sliders,
  Clock,
  ArrowRight,
  PieChart,
  ShieldAlert,
  Info,
} from 'lucide-react';
import { getProducts, getAnalytics } from '../services/api';
import { Product, AnalyticsData, DefectClass } from '../types';

interface AnalyticsPageProps {
  onNavigateToLive?: () => void;
  onNavigateToProducts?: () => void;
}

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({
  onNavigateToLive,
  onNavigateToProducts,
}) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>('ALL');
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);

  const [loadingProducts, setLoadingProducts] = useState<boolean>(true);
  const [loadingAnalytics, setLoadingAnalytics] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load products list on mount
  useEffect(() => {
    async function loadProductsList() {
      setLoadingProducts(true);
      setError(null);
      try {
        const prods = await getProducts();
        setProducts(prods);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load products list.');
      } finally {
        setLoadingProducts(false);
      }
    }
    loadProductsList();
  }, []);

  // Fetch analytics data when selected product changes
  const fetchAnalyticsData = useCallback(async (productId: string) => {
    setLoadingAnalytics(true);
    setError(null);
    try {
      const scopeId = productId === 'ALL' ? null : productId;
      const data = await getAnalytics(scopeId);
      setAnalytics(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics statistics.');
      setAnalytics(null);
    } finally {
      setLoadingAnalytics(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalyticsData(selectedProductId);
  }, [selectedProductId, fetchAnalyticsData]);

  // Selected product object for display
  const currentProduct = useMemo(() => {
    if (selectedProductId === 'ALL') return null;
    return products.find((p) => p.product_id === selectedProductId) || null;
  }, [products, selectedProductId]);

  // Defect class breakdown calculations
  const defectBreakdown = useMemo(() => {
    if (!analytics || !analytics.defect_class_counts) return [];
    const classes: DefectClass[] = ['Scratch', 'Crack', 'Discoloration', 'UNKNOWN'];
    const totalDefects = analytics.defective_count;

    return classes.map((cls) => {
      const count = analytics.defect_class_counts[cls] || 0;
      const percentage = totalDefects > 0 ? (count / totalDefects) * 100 : 0;
      return {
        className: cls,
        count,
        percentage,
      };
    });
  }, [analytics]);

  const classColorMap: Record<DefectClass, { bar: string; badge: string; text: string }> = {
    Scratch: {
      bar: 'bg-amber-500',
      badge: 'bg-amber-950/80 border-amber-800 text-amber-300',
      text: 'text-amber-400',
    },
    Crack: {
      bar: 'bg-orange-500',
      badge: 'bg-orange-950/80 border-orange-800 text-orange-300',
      text: 'text-orange-400',
    },
    Discoloration: {
      bar: 'bg-cyan-500',
      badge: 'bg-cyan-950/80 border-cyan-800 text-cyan-300',
      text: 'text-cyan-400',
    },
    UNKNOWN: {
      bar: 'bg-purple-500',
      badge: 'bg-purple-950/80 border-purple-800 text-purple-300',
      text: 'text-purple-400',
    },
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Scope Selector */}
      <div className="bg-industrial-900/80 border border-industrial-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-bold text-white tracking-wide flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-sky-400" />
            <span>Quality Analytics Dashboard</span>
          </h1>
          <p className="text-xs text-industrial-400 mt-1 max-w-2xl">
            Real-time quality KPIs, defect rate distribution, latency statistics, and 4-class defect
            taxonomy breakdown aggregated directly from backend inspection database records.
          </p>
        </div>

        {/* Product Scope Selector */}
        <div className="flex items-center space-x-2 shrink-0">
          <label htmlFor="analytics-product-select" className="text-xs font-mono text-industrial-400 flex items-center space-x-1.5">
            <Sliders className="w-3.5 h-3.5 text-sky-400" />
            <span>Scope:</span>
          </label>
          <select
            id="analytics-product-select"
            value={selectedProductId}
            onChange={(e) => setSelectedProductId(e.target.value)}
            disabled={loadingProducts}
            className="bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-sky-500"
          >
            <option value="ALL">All Products (Global Aggregate)</option>
            {products.map((p) => (
              <option key={p.product_id} value={p.product_id}>
                {p.name} ({p.product_id})
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => fetchAnalyticsData(selectedProductId)}
            disabled={loadingAnalytics}
            title="Refresh statistics"
            className="text-industrial-400 hover:text-white p-2 rounded-lg bg-industrial-950 border border-industrial-800 hover:border-industrial-700 transition-colors disabled:opacity-40"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${loadingAnalytics ? 'animate-spin text-sky-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Production Metrics Clarification Notice */}
      <div className="bg-industrial-900/60 border border-industrial-800 rounded-lg p-3 text-xs text-industrial-300 flex items-start space-x-2.5">
        <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-semibold text-white">Production Operational Statistics: </span>
          All metrics below represent actual inspection outcomes stored in the database.
          They are distinct from offline model benchmark accuracy, precision, recall, or AUROC, which require ground-truth label verification.
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => fetchAnalyticsData(selectedProductId)}
            className="text-[11px] font-mono text-red-400 hover:text-white underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading State */}
      {loadingAnalytics && !analytics ? (
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-16 text-center font-mono text-xs text-industrial-400 space-y-3">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-sky-400" />
          <div>Aggregating database inspection analytics...</div>
        </div>
      ) : !loadingProducts && products.length === 0 ? (
        /* Empty State: No Products */
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-12 text-center space-y-4 font-mono">
          <div className="w-12 h-12 mx-auto rounded-full bg-industrial-800/80 border border-industrial-700 flex items-center justify-center text-industrial-400">
            <Sliders className="w-6 h-6" />
          </div>
          <div className="text-base font-semibold text-white">No Products Found</div>
          <p className="text-xs text-industrial-400 max-w-md mx-auto">
            No inspection products have been created or calibrated yet. Create and calibrate a product to generate inspection data.
          </p>
          {onNavigateToProducts && (
            <button
              type="button"
              onClick={onNavigateToProducts}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium transition-colors shadow-sm"
            >
              <span>Go to Products</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      ) : !analytics || analytics.total_inspections === 0 ? (
        /* Empty State: No Inspections */
        <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-12 text-center space-y-4 font-mono">
          <div className="w-12 h-12 mx-auto rounded-full bg-industrial-800/80 border border-industrial-700 flex items-center justify-center text-industrial-400">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div className="text-base font-semibold text-white">No Inspection Analytics Available</div>
          <p className="text-xs text-industrial-400 max-w-md mx-auto">
            {selectedProductId === 'ALL'
              ? 'No inspection records exist across any products in the database.'
              : `No inspection records have been recorded yet for product '${selectedProductId}'.`}
          </p>
          {onNavigateToLive && (
            <button
              type="button"
              onClick={onNavigateToLive}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium transition-colors shadow-sm"
            >
              <span>Go to Live Inspection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      ) : (
        /* Analytics Content */
        <div className="space-y-6">
          {/* Key KPI Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
            {/* Total Inspections */}
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-industrial-400">
                <span>Total Inspections</span>
                <BarChart3 className="w-4 h-4 text-sky-400" />
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {analytics.total_inspections}
              </div>
              <div className="text-[11px] text-industrial-500 truncate">
                {currentProduct ? `Product: ${currentProduct.product_id}` : 'All products combined'}
              </div>
            </div>

            {/* NORMAL Verdict Count & Rate */}
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-emerald-400">
                <span>Passed (NORMAL)</span>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-emerald-400 tracking-tight">
                {analytics.normal_count}
              </div>
              <div className="text-[11px] text-industrial-400 flex items-center justify-between">
                <span>Pass Rate:</span>
                <span className="text-white font-semibold">
                  {(analytics.normal_rate * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* DEFECTIVE Verdict Count & Rate */}
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-rose-400">
                <span>Defective Flagged</span>
                <AlertOctagon className="w-4 h-4 text-rose-400" />
              </div>
              <div className="text-2xl font-bold text-rose-400 tracking-tight">
                {analytics.defective_count}
              </div>
              <div className="text-[11px] text-industrial-400 flex items-center justify-between">
                <span>Defect Rate:</span>
                <span className="text-white font-semibold">
                  {(analytics.defect_rate * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Average Latency */}
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs text-industrial-400">
                <span>Avg Latency</span>
                <Clock className="w-4 h-4 text-sky-400" />
              </div>
              <div className="text-2xl font-bold text-white tracking-tight">
                {analytics.average_execution_time_ms.toFixed(1)}
                <span className="text-xs text-industrial-400 font-normal ml-1">ms</span>
              </div>
              <div className="text-[11px] text-industrial-500">
                Two-stage pipeline latency
              </div>
            </div>
          </div>

          {/* Detailed Visualizations Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Card: Pass vs Fail Ratio */}
            <div className="lg:col-span-5 bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-5">
              <div className="flex items-center space-x-2 border-b border-industrial-800 pb-3">
                <PieChart className="w-4 h-4 text-sky-400" />
                <h2 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                  Pass vs. Fail Quality Ratio
                </h2>
              </div>

              {/* Stacked Progress Bar */}
              <div className="space-y-2">
                <div className="h-6 w-full rounded-lg overflow-hidden flex bg-industrial-950 border border-industrial-800 p-0.5">
                  <div
                    style={{ width: `${(analytics.normal_rate * 100)}%` }}
                    className="bg-emerald-500 transition-all duration-300 rounded-l flex items-center justify-center text-[10px] font-mono font-bold text-industrial-950"
                  >
                    {analytics.normal_rate > 0.1 ? `${(analytics.normal_rate * 100).toFixed(0)}%` : ''}
                  </div>
                  <div
                    style={{ width: `${(analytics.defect_rate * 100)}%` }}
                    className="bg-rose-500 transition-all duration-300 rounded-r flex items-center justify-center text-[10px] font-mono font-bold text-white"
                  >
                    {analytics.defect_rate > 0.1 ? `${(analytics.defect_rate * 100).toFixed(0)}%` : ''}
                  </div>
                </div>

                {/* Legend */}
                <div className="flex items-center justify-between text-xs font-mono pt-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-3 h-3 rounded-full bg-emerald-500 shrink-0" />
                    <span className="text-industrial-300">
                      NORMAL: <strong className="text-white">{analytics.normal_count}</strong> ({(analytics.normal_rate * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="w-3 h-3 rounded-full bg-rose-500 shrink-0" />
                    <span className="text-industrial-300">
                      DEFECTIVE: <strong className="text-white">{analytics.defective_count}</strong> ({(analytics.defect_rate * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
              </div>

              {/* Stage Summary Breakdown */}
              <div className="bg-industrial-950 p-3.5 rounded-lg border border-industrial-800 space-y-2 font-mono text-xs">
                <div className="text-[11px] text-industrial-400 font-semibold uppercase tracking-wider">
                  Two-Stage Pipeline Distribution
                </div>
                <div className="flex justify-between items-center text-industrial-300 text-[11px]">
                  <span>Stage 1 Filtered (Normal / Bypassed):</span>
                  <span className="text-emerald-400 font-bold">{analytics.normal_count} samples</span>
                </div>
                <div className="flex justify-between items-center text-industrial-300 text-[11px]">
                  <span>Stage 2 Classified (Anomalous):</span>
                  <span className="text-rose-400 font-bold">{analytics.defective_count} samples</span>
                </div>
              </div>
            </div>

            {/* Right Card: Defect Class Taxonomy Breakdown */}
            <div className="lg:col-span-7 bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-5">
              <div className="flex items-center justify-between border-b border-industrial-800 pb-3">
                <div className="flex items-center space-x-2">
                  <ShieldAlert className="w-4 h-4 text-purple-400" />
                  <h2 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                    Defect Taxonomy Breakdown (Stage 2)
                  </h2>
                </div>
                <span className="text-[11px] font-mono text-industrial-400">
                  Total Defects: <strong className="text-white">{analytics.defective_count}</strong>
                </span>
              </div>

              {analytics.defective_count === 0 ? (
                <div className="py-8 text-center font-mono text-xs text-industrial-400 space-y-1">
                  <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                  <div className="text-white font-semibold">Zero Defects Recorded</div>
                  <p className="text-[11px] text-industrial-500">
                    No anomalous samples have been detected in this scope.
                  </p>
                </div>
              ) : (
                <div className="space-y-4 font-mono text-xs">
                  {defectBreakdown.map((item) => {
                    const cls = item.className as DefectClass;
                    const colors = classColorMap[cls];

                    return (
                      <div key={item.className} className="space-y-1.5">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className="flex items-center space-x-2">
                            <span className={`px-2 py-0.5 rounded border text-[10px] font-bold ${colors.badge}`}>
                              {item.className}
                            </span>
                            <span className="text-industrial-400">
                              {item.className === 'UNKNOWN' ? '(Out-of-distribution defect)' : '(Supervised class)'}
                            </span>
                          </span>
                          <span className="text-white font-medium">
                            {item.count} defect{item.count !== 1 ? 's' : ''}{' '}
                            <span className="text-industrial-400 font-normal">
                              ({item.percentage.toFixed(1)}%)
                            </span>
                          </span>
                        </div>

                        {/* Bar */}
                        <div className="h-3 w-full bg-industrial-950 rounded-full overflow-hidden border border-industrial-800/80">
                          <div
                            style={{ width: `${Math.max(item.percentage, item.count > 0 ? 3 : 0)}%` }}
                            className={`h-full rounded-full transition-all duration-300 ${colors.bar}`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Taxonomy Legend */}
              <div className="pt-2 border-t border-industrial-800 text-[11px] font-mono text-industrial-500 leading-relaxed">
                4-class approved taxonomy: 3 supervised surface defect classes (<code className="text-amber-300">Scratch</code>, <code className="text-orange-300">Crack</code>, <code className="text-cyan-300">Discoloration</code>) and 1 open-set class (<code className="text-purple-300">UNKNOWN</code>).
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
