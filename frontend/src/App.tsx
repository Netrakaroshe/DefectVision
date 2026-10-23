import { useState } from 'react';
import { Header } from './components/Header';
import { Navigation, NavTab } from './components/Navigation';
import { ProductsPage } from './pages/ProductsPage';
import { LiveInspectionPage } from './pages/LiveInspectionPage';
import { InspectionHistoryPage } from './pages/InspectionHistoryPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { Cpu, CheckCircle2, AlertOctagon } from 'lucide-react';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('live');

  return (
    <div className="min-h-screen bg-industrial-950 text-industrial-100 flex flex-col font-sans">
      <Header activeTab={currentTab.toUpperCase()} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 pb-12">
        <Navigation currentTab={currentTab} onTabChange={setCurrentTab} />

        {/* Dashboard Shell / Section Placeholders */}
        <div className="mt-4 space-y-6">
          {/* Top Status Banner */}
          <div className="bg-industrial-900 border border-industrial-800 rounded-xl p-5 flex items-center justify-between">
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-sky-400" />
                <h1 className="text-base font-semibold text-white tracking-wide">
                  DefectVision Inspection Station
                </h1>
              </div>
              <p className="text-xs text-industrial-400">
                Stage 1 PatchCore Anomaly Localization + Stage 2 Supervised ResNet-18 Defect Classification
              </p>
            </div>

            {/* Architecture Legend */}
            <div className="flex items-center space-x-3 text-xs font-mono">
              <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-industrial-950/70 border border-industrial-800 text-industrial-300">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-normal" />
                <span>NORMAL</span>
              </span>
              <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-industrial-950/70 border border-industrial-800 text-industrial-300">
                <AlertOctagon className="w-3.5 h-3.5 text-status-defective" />
                <span>DEFECTIVE</span>
              </span>
            </div>
          </div>

          {/* Section Placeholders */}
          {currentTab === 'live' && (
            <LiveInspectionPage onNavigateToProducts={() => setCurrentTab('products')} />
          )}

          {currentTab === 'history' && (
            <InspectionHistoryPage
              onNavigateToLive={() => setCurrentTab('live')}
              onNavigateToProducts={() => setCurrentTab('products')}
            />
          )}

          {currentTab === 'analytics' && (
            <AnalyticsPage
              onNavigateToLive={() => setCurrentTab('live')}
              onNavigateToProducts={() => setCurrentTab('products')}
            />
          )}

          {currentTab === 'products' && <ProductsPage />}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-industrial-800/80 bg-industrial-950 py-4 text-center text-xs text-industrial-500 font-mono">
        DefectVision Industrial Inspection System &copy; 2026 &bull; Camera-Agnostic Vision Pipeline
      </footer>
    </div>
  );
}

export default App;
