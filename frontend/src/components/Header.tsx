import React from 'react';
import { Eye, ShieldCheck, Activity } from 'lucide-react';
import { API_BASE_URL } from '../services/api';

interface HeaderProps {
  activeTab: string;
}

export const Header: React.FC<HeaderProps> = ({ activeTab }) => {
  return (
    <header className="border-b border-industrial-800 bg-industrial-900/80 backdrop-blur px-6 py-3 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="bg-industrial-800 border border-industrial-700 p-2 rounded-lg text-industrial-200">
            <Eye className="w-5 h-5 text-sky-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-wider text-white">DEFECTVISION</span>
              <span className="text-xs px-2 py-0.5 rounded bg-industrial-800 border border-industrial-700 text-industrial-400 font-mono">
                v0.6.0
              </span>
            </div>
            <p className="text-xs text-industrial-400">Two-Stage Industrial Visual Inspection</p>
          </div>
        </div>

        {/* System Status Indicators */}
        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-industrial-800/70 border border-industrial-700/60 text-industrial-300">
            <Activity className="w-3.5 h-3.5 text-status-normal animate-pulse" />
            <span>API: {API_BASE_URL}</span>
          </div>

          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-industrial-800/70 border border-industrial-700/60 text-industrial-300">
            <ShieldCheck className="w-3.5 h-3.5 text-sky-400" />
            <span>Active: {activeTab}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
