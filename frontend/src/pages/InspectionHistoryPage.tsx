import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  History,
  CheckCircle2,
  AlertOctagon,
  AlertTriangle,
  Loader2,
  Image as ImageIcon,
  Clock,
  RotateCcw,
  Tag,
  Hash,
  Filter,
  Eye,
  ChevronRight,
  ShieldCheck,
  ArrowRight,
  Layers,
  Sliders,
} from 'lucide-react';
import {
  getProducts,
  getProductInspections,
  getInspection,
  getInspectionHeatmapUrl,
  getInspectionImageUrl,
} from '../services/api';
import { Product, Inspection, InspectionVerdict } from '../types';

interface InspectionHistoryPageProps {
  onNavigateToLive?: () => void;
  onNavigateToProducts?: () => void;
}

export const InspectionHistoryPage: React.FC<InspectionHistoryPageProps> = ({
  onNavigateToLive,
  onNavigateToProducts,
}) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>('');
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [selectedInspectionId, setSelectedInspectionId] = useState<string | null>(null);
  const [selectedInspection, setSelectedInspection] = useState<Inspection | null>(null);

  const [verdictFilter, setVerdictFilter] = useState<'ALL' | InspectionVerdict>('ALL');
  const [loadingProducts, setLoadingProducts] = useState<boolean>(true);
  const [loadingInspections, setLoadingInspections] = useState<boolean>(false);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [activeMediaTab, setActiveMediaTab] = useState<'heatmap' | 'query'>('heatmap');

  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Load products list on mount
  useEffect(() => {
    async function loadProductsList() {
      setLoadingProducts(true);
      setError(null);
      try {
        const prods = await getProducts();
        setProducts(prods);
        if (prods.length > 0) {
          // Default to the first product with READY status, or simply the first product
          const firstReady = prods.find((p) => p.status === 'READY');
          setSelectedProductId(firstReady ? firstReady.product_id : prods[0].product_id);
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load products list from backend.');
      } finally {
        setLoadingProducts(false);
      }
    }
    loadProductsList();
  }, []);

  // Fetch inspections for the selected product
  const fetchInspections = useCallback(async (productId: string, filter: 'ALL' | InspectionVerdict) => {
    if (!productId) {
      setInspections([]);
      return;
    }
    setLoadingInspections(true);
    setError(null);
    try {
      const resp = await getProductInspections(productId, {
        verdict: filter !== 'ALL' ? filter : undefined,
      });

      // Safely sort newest first using timestamp / created_at
      const sorted = [...(resp.items || [])].sort((a, b) => {
        const timeA = new Date(a.timestamp || a.created_at).getTime();
        const timeB = new Date(b.timestamp || b.created_at).getTime();
        return timeB - timeA;
      });

      setInspections(sorted);

      // Auto-select the first inspection if available and none currently selected
      if (sorted.length > 0) {
        setSelectedInspectionId((prev) => {
          const exists = sorted.some((item) => item.inspection_id === prev);
          return exists ? prev : sorted[0].inspection_id;
        });
      } else {
        setSelectedInspectionId(null);
        setSelectedInspection(null);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch inspection history.');
      setInspections([]);
      setSelectedInspectionId(null);
      setSelectedInspection(null);
    } finally {
      setLoadingInspections(false);
    }
  }, []);

  // Trigger inspection list fetch whenever selected product or verdict filter changes
  useEffect(() => {
    if (selectedProductId) {
      fetchInspections(selectedProductId, verdictFilter);
    }
  }, [selectedProductId, verdictFilter, fetchInspections]);

  // Fetch full details when an inspection is selected
  useEffect(() => {
    if (!selectedInspectionId) {
      setSelectedInspection(null);
      return;
    }

    // First check local list cache
    const cached = inspections.find((i) => i.inspection_id === selectedInspectionId);
    if (cached) {
      setSelectedInspection(cached);
    }

    // Then fetch fresh detail from GET /inspections/{id}
    async function loadDetail(id: string) {
      setLoadingDetail(true);
      setDetailError(null);
      try {
        const detail = await getInspection(id);
        setSelectedInspection(detail);
      } catch (err: unknown) {
        setDetailError(err instanceof Error ? err.message : 'Failed to load inspection detail.');
      } finally {
        setLoadingDetail(false);
      }
    }

    loadDetail(selectedInspectionId);
  }, [selectedInspectionId, inspections]);

  // Selected product object
  const currentProduct = useMemo(() => {
    return products.find((p) => p.product_id === selectedProductId) || null;
  }, [products, selectedProductId]);

  const formatTimestamp = (ts: string) => {
    try {
      const d = new Date(ts);
      if (isNaN(d.getTime())) return ts;
      return d.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    } catch {
      return ts;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-industrial-900/80 border border-industrial-800 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-bold text-white tracking-wide flex items-center space-x-2">
            <History className="w-5 h-5 text-sky-400" />
            <span>Inspection History & Records</span>
          </h1>
          <p className="text-xs text-industrial-400 mt-1 max-w-2xl">
            Browse and review past two-stage inspection logs, continuous PatchCore anomaly scores,
            supervised defect classes, and localization heatmaps.
          </p>
        </div>

        {onNavigateToLive && (
          <button
            type="button"
            onClick={onNavigateToLive}
            className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium transition-colors shrink-0 shadow-sm"
          >
            <span>Run New Inspection</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Global Error Banner */}
      {error && (
        <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => selectedProductId && fetchInspections(selectedProductId, verdictFilter)}
            className="text-[11px] font-mono text-red-400 hover:text-white underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Product Selector, Filters, and Inspection List */}
        <div className="lg:col-span-5 space-y-4">
          {/* Product Selector Card */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="history-product-select" className="text-xs font-semibold text-white uppercase tracking-wider font-mono flex items-center space-x-2">
                <Sliders className="w-3.5 h-3.5 text-sky-400" />
                <span>Product</span>
              </label>
              <button
                type="button"
                onClick={() => selectedProductId && fetchInspections(selectedProductId, verdictFilter)}
                disabled={loadingInspections}
                title="Refresh inspection list"
                className="text-industrial-400 hover:text-white p-1 rounded transition-colors disabled:opacity-40"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${loadingInspections ? 'animate-spin text-sky-400' : ''}`} />
              </button>
            </div>

            {loadingProducts ? (
              <div className="py-3 text-center text-xs text-industrial-400 font-mono flex items-center justify-center space-x-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-400" />
                <span>Loading products...</span>
              </div>
            ) : products.length === 0 ? (
              <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 text-xs text-industrial-400 space-y-2">
                <div>No products found in backend database.</div>
                {onNavigateToProducts && (
                  <button
                    type="button"
                    onClick={onNavigateToProducts}
                    className="text-xs text-sky-400 hover:text-sky-300 font-mono underline"
                  >
                    Go to Products to register one &rarr;
                  </button>
                )}
              </div>
            ) : (
              <select
                id="history-product-select"
                value={selectedProductId}
                onChange={(e) => setSelectedProductId(e.target.value)}
                className="w-full bg-industrial-950 border border-industrial-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-sky-500"
              >
                {products.map((p) => (
                  <option key={p.product_id} value={p.product_id}>
                    {p.name} ({p.product_id}) [{p.status}]
                  </option>
                ))}
              </select>
            )}

            {/* Verdict Filter Tabs */}
            <div className="flex items-center space-x-1.5 pt-1 border-t border-industrial-800">
              <span className="text-[11px] font-mono text-industrial-400 mr-1 flex items-center space-x-1">
                <Filter className="w-3 h-3" />
                <span>Filter:</span>
              </span>
              {(['ALL', 'NORMAL', 'DEFECTIVE'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setVerdictFilter(mode)}
                  className={`px-2.5 py-1 rounded text-[11px] font-mono font-medium transition-colors ${
                    verdictFilter === mode
                      ? mode === 'NORMAL'
                        ? 'bg-emerald-950 border border-emerald-700 text-emerald-300'
                        : mode === 'DEFECTIVE'
                        ? 'bg-rose-950 border border-rose-700 text-rose-300'
                        : 'bg-industrial-800 border border-industrial-700 text-white'
                      : 'bg-industrial-950/60 border border-industrial-800 text-industrial-400 hover:text-industrial-200'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Inspection Records List */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl overflow-hidden flex flex-col">
            <div className="p-3.5 bg-industrial-900 border-b border-industrial-800 flex items-center justify-between text-xs font-mono">
              <span className="text-industrial-300 font-semibold uppercase tracking-wider">
                Inspection Log ({inspections.length})
              </span>
              <span className="text-[11px] text-industrial-500">Ordered: Newest first</span>
            </div>

            {loadingInspections ? (
              <div className="py-12 text-center text-xs text-industrial-400 font-mono space-y-2">
                <Loader2 className="w-5 h-5 animate-spin mx-auto text-sky-400" />
                <div>Fetching inspection records...</div>
              </div>
            ) : inspections.length === 0 ? (
              <div className="p-8 text-center space-y-3 font-mono">
                <div className="w-10 h-10 mx-auto rounded-full bg-industrial-800/80 border border-industrial-700 flex items-center justify-center text-industrial-500">
                  <History className="w-5 h-5" />
                </div>
                <div className="text-xs font-semibold text-white">No Inspection Records</div>
                <p className="text-[11px] text-industrial-400 max-w-xs mx-auto">
                  {verdictFilter !== 'ALL'
                    ? `No ${verdictFilter} inspections found for '${selectedProductId}'.`
                    : `No inspections have been performed yet for '${selectedProductId}'.`}
                </p>
                {onNavigateToLive && currentProduct?.status === 'READY' && (
                  <button
                    type="button"
                    onClick={onNavigateToLive}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-mono font-medium transition-colors"
                  >
                    <span>Run first inspection</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                )}
              </div>
            ) : (
              <div className="divide-y divide-industrial-800 max-h-[600px] overflow-y-auto">
                {inspections.map((insp) => {
                  const isSelected = selectedInspectionId === insp.inspection_id;
                  const isNorm = insp.verdict === 'NORMAL';

                  return (
                    <button
                      key={insp.inspection_id}
                      type="button"
                      onClick={() => setSelectedInspectionId(insp.inspection_id)}
                      className={`w-full text-left p-3.5 transition-colors flex items-start justify-between space-x-3 ${
                        isSelected
                          ? 'bg-industrial-800/90 border-l-4 border-l-sky-500'
                          : 'hover:bg-industrial-800/40'
                      }`}
                    >
                      <div className="space-y-1.5 min-w-0 flex-1">
                        {/* Verdict & Anomaly Score */}
                        <div className="flex items-center space-x-2 font-mono">
                          <span
                            className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-bold ${
                              isNorm
                                ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/80'
                                : 'bg-rose-950/80 text-rose-400 border border-rose-800/80'
                            }`}
                          >
                            {isNorm ? (
                              <CheckCircle2 className="w-3 h-3" />
                            ) : (
                              <AlertOctagon className="w-3 h-3" />
                            )}
                            <span>{insp.verdict}</span>
                          </span>

                          <span className="text-xs text-white font-medium">
                            Score: {insp.anomaly_score.toFixed(2)}
                          </span>
                          <span className="text-[11px] text-industrial-400">
                            / Thresh: {insp.anomaly_threshold.toFixed(2)}
                          </span>
                        </div>

                        {/* Stage 2 Badge & Inspection ID */}
                        <div className="flex items-center space-x-2 text-[11px] font-mono text-industrial-400 truncate">
                          {insp.stage2_executed ? (
                            <span className="px-1.5 py-0.5 rounded bg-amber-950/70 border border-amber-800/70 text-amber-300 text-[10px]">
                              {insp.defect_class || 'DEFECT'}
                              {insp.classifier_confidence != null
                                ? ` (${(insp.classifier_confidence * 100).toFixed(0)}%)`
                                : ''}
                            </span>
                          ) : (
                            <span className="px-1.5 py-0.5 rounded bg-industrial-950 border border-industrial-800 text-industrial-400 text-[10px]">
                              Stage 2: Bypassed
                            </span>
                          )}

                          <span className="truncate text-industrial-500">
                            ID: {insp.inspection_id.substring(0, 8)}...
                          </span>
                        </div>

                        {/* Timestamp */}
                        <div className="text-[10px] font-mono text-industrial-500 flex items-center space-x-1">
                          <Clock className="w-3 h-3" />
                          <span>{formatTimestamp(insp.timestamp || insp.created_at)}</span>
                        </div>
                      </div>

                      <ChevronRight
                        className={`w-4 h-4 shrink-0 transition-colors mt-1 ${
                          isSelected ? 'text-sky-400' : 'text-industrial-600'
                        }`}
                      />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Inspection Detail Pane */}
        <div className="lg:col-span-7">
          {loadingDetail && !selectedInspection ? (
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-12 text-center text-xs text-industrial-400 font-mono space-y-3">
              <Loader2 className="w-6 h-6 animate-spin mx-auto text-sky-400" />
              <div>Loading inspection details...</div>
            </div>
          ) : !selectedInspection ? (
            <div className="bg-industrial-900/60 border border-industrial-800 rounded-xl p-12 text-center space-y-3 font-mono">
              <div className="w-12 h-12 mx-auto rounded-full bg-industrial-800/70 border border-industrial-700 flex items-center justify-center text-industrial-400">
                <Eye className="w-6 h-6" />
              </div>
              <div className="text-sm font-semibold text-white">No Inspection Selected</div>
              <p className="text-xs text-industrial-400 max-w-sm mx-auto">
                Select any record from the list on the left to examine its full PatchCore score,
                supervised classification, and anomaly localization heatmap.
              </p>
            </div>
          ) : (
            <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 space-y-5">
              {/* Detail Error banner */}
              {detailError && (
                <div className="bg-red-950/60 border border-red-800 rounded-lg p-3 text-xs text-red-300 flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{detailError}</span>
                </div>
              )}

              {/* Verdict Header Banner */}
              <div
                className={`p-4 rounded-xl border flex items-center justify-between ${
                  selectedInspection.verdict === 'NORMAL'
                    ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                    : 'bg-rose-950/60 border-rose-800 text-rose-300'
                }`}
              >
                <div className="flex items-center space-x-3">
                  {selectedInspection.verdict === 'NORMAL' ? (
                    <CheckCircle2 className="w-8 h-8 text-emerald-400 shrink-0" />
                  ) : (
                    <AlertOctagon className="w-8 h-8 text-rose-400 shrink-0" />
                  )}
                  <div>
                    <div className="text-xl font-bold tracking-wider font-mono">
                      {selectedInspection.verdict}
                    </div>
                    <div className="text-xs text-industrial-300">
                      {selectedInspection.verdict === 'NORMAL'
                        ? 'Continuous anomaly score is below threshold. Sample passed inspection.'
                        : 'Continuous anomaly score exceeded calibrated threshold. Defect detected.'}
                    </div>
                  </div>
                </div>

                {/* Execution Time */}
                {selectedInspection.execution_time_ms != null && (
                  <div className="text-right font-mono">
                    <div className="text-xs text-industrial-400">Latency</div>
                    <div className="text-sm font-bold text-white">
                      {selectedInspection.execution_time_ms.toFixed(1)} ms
                    </div>
                  </div>
                )}
              </div>

              {/* Metadata strip */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
                <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 space-y-1">
                  <span className="text-[11px] text-industrial-400 flex items-center space-x-1">
                    <Hash className="w-3 h-3" />
                    <span>Inspection ID</span>
                  </span>
                  <div className="text-white font-medium truncate" title={selectedInspection.inspection_id}>
                    {selectedInspection.inspection_id}
                  </div>
                </div>

                <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 space-y-1">
                  <span className="text-[11px] text-industrial-400 flex items-center space-x-1">
                    <Tag className="w-3 h-3" />
                    <span>Product ID</span>
                  </span>
                  <div className="text-white font-medium truncate" title={selectedInspection.product_id}>
                    {selectedInspection.product_id}
                  </div>
                </div>

                <div className="bg-industrial-950 p-3 rounded-lg border border-industrial-800 space-y-1">
                  <span className="text-[11px] text-industrial-400 flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>Timestamp</span>
                  </span>
                  <div className="text-white font-medium truncate">
                    {formatTimestamp(selectedInspection.timestamp || selectedInspection.created_at)}
                  </div>
                </div>
              </div>

              {/* Two-Stage Vision Pipeline Results Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Stage 1: PatchCore Anomaly Scoring */}
                <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 space-y-3">
                  <div className="flex items-center space-x-2 border-b border-industrial-800/80 pb-2">
                    <ShieldCheck className="w-4 h-4 text-sky-400" />
                    <h3 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                      Stage 1: Anomaly Scoring
                    </h3>
                  </div>

                  <div className="space-y-2.5 font-mono text-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-industrial-400">Continuous Score:</span>
                      <span className="text-white font-bold text-sm">
                        {selectedInspection.anomaly_score.toFixed(4)}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-industrial-400">Calibrated Threshold:</span>
                      <span className="text-emerald-400 font-bold">
                        {selectedInspection.anomaly_threshold.toFixed(4)}
                      </span>
                    </div>

                    <div className="flex justify-between items-center pt-1 border-t border-industrial-800/60">
                      <span className="text-industrial-400">Decision Margin:</span>
                      <span
                        className={`font-bold ${
                          selectedInspection.margin > 0 ? 'text-rose-400' : 'text-emerald-400'
                        }`}
                      >
                        {selectedInspection.margin > 0 ? '+' : ''}
                        {selectedInspection.margin.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Stage 2: Supervised Defect Classification */}
                <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 space-y-3">
                  <div className="flex items-center space-x-2 border-b border-industrial-800/80 pb-2">
                    <Layers className="w-4 h-4 text-purple-400" />
                    <h3 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                      Stage 2: Defect Classification
                    </h3>
                  </div>

                  {selectedInspection.stage2_executed ? (
                    <div className="space-y-2.5 font-mono text-xs">
                      <div className="flex justify-between items-center">
                        <span className="text-industrial-400">Defect Class:</span>
                        <span className="px-2 py-0.5 rounded bg-purple-950 border border-purple-800 text-purple-300 font-bold">
                          {selectedInspection.defect_class || 'UNKNOWN'}
                        </span>
                      </div>

                      {selectedInspection.classifier_confidence != null && (
                        <div className="flex justify-between items-center">
                          <span className="text-industrial-400">Confidence:</span>
                          <span className="text-white font-bold">
                            {(selectedInspection.classifier_confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      )}

                      {/* Class Probabilities Distribution */}
                      {selectedInspection.classifier_probabilities && (
                        <div className="pt-2 border-t border-industrial-800/60 space-y-1 text-[11px]">
                          <span className="text-industrial-400 block text-[10px] uppercase tracking-wider">
                            Probabilities
                          </span>
                          {Object.entries(selectedInspection.classifier_probabilities).map(
                            ([cls, prob]) => (
                              <div key={cls} className="flex justify-between items-center">
                                <span className="text-industrial-300">{cls}:</span>
                                <span className="text-industrial-400">{(prob * 100).toFixed(2)}%</span>
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-4 text-center font-mono space-y-1.5">
                      <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-industrial-900 border border-industrial-800 text-emerald-400 text-xs">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Bypassed (Normal Sample)</span>
                      </div>
                      <p className="text-[11px] text-industrial-400 max-w-xs mx-auto">
                        Stage 2 supervised classifier only executes when anomaly score exceeds threshold.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Media Section: Heatmap & Query Image */}
              <div className="bg-industrial-950 p-4 rounded-xl border border-industrial-800 space-y-3">
                <div className="flex items-center justify-between border-b border-industrial-800/80 pb-2.5">
                  <div className="flex items-center space-x-2">
                    <ImageIcon className="w-4 h-4 text-sky-400" />
                    <h3 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
                      Inspection Visual Artifacts
                    </h3>
                  </div>

                  {/* Tabs: Heatmap vs Query Image */}
                  <div className="flex space-x-1 bg-industrial-900 p-0.5 rounded-lg border border-industrial-800">
                    <button
                      type="button"
                      onClick={() => setActiveMediaTab('heatmap')}
                      className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                        activeMediaTab === 'heatmap'
                          ? 'bg-industrial-800 text-white font-medium shadow-sm'
                          : 'text-industrial-400 hover:text-white'
                      }`}
                    >
                      Heatmap
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveMediaTab('query')}
                      className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                        activeMediaTab === 'query'
                          ? 'bg-industrial-800 text-white font-medium shadow-sm'
                          : 'text-industrial-400 hover:text-white'
                      }`}
                    >
                      Original Sample
                    </button>
                  </div>
                </div>

                {/* Media Display Area */}
                {activeMediaTab === 'heatmap' ? (
                  selectedInspection.heatmap_path ? (
                    <div className="space-y-2">
                      <div className="relative rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-video max-h-80 flex items-center justify-center">
                        <img
                          src={getInspectionHeatmapUrl(selectedInspection.inspection_id)}
                          alt={`Heatmap for ${selectedInspection.inspection_id}`}
                          className="max-h-full max-w-full object-contain mx-auto"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[11px] font-mono text-industrial-400 px-1">
                        <span>Anomaly Localization Map</span>
                        <span className="text-emerald-400">Streamed from controlled media endpoint</span>
                      </div>
                    </div>
                  ) : (
                    <div className="py-10 text-center font-mono text-xs text-industrial-400 space-y-1">
                      <ImageIcon className="w-8 h-8 text-industrial-600 mx-auto mb-2" />
                      <div>No Heatmap Artifact Available</div>
                      <p className="text-[11px] text-industrial-500 max-w-xs mx-auto">
                        This inspection was performed without the heatmap save parameter enabled.
                      </p>
                    </div>
                  )
                ) : (
                  <div className="space-y-2">
                    <div className="relative rounded-lg overflow-hidden border border-industrial-800 bg-black aspect-video max-h-80 flex items-center justify-center">
                      <img
                        src={getInspectionImageUrl(selectedInspection.inspection_id)}
                        alt={`Query image for ${selectedInspection.inspection_id}`}
                        className="max-h-full max-w-full object-contain mx-auto"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = 'none';
                        }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[11px] font-mono text-industrial-400 px-1">
                      <span>Original Uploaded Query Sample</span>
                      <span className="text-industrial-400">Protected media storage</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
