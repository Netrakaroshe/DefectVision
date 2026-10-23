import React from 'react';
import { Camera, History, BarChart3, Boxes } from 'lucide-react';
import { cn } from '../lib/utils';

export type NavTab = 'live' | 'history' | 'analytics' | 'products';

interface NavigationProps {
  currentTab: NavTab;
  onTabChange: (tab: NavTab) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ currentTab, onTabChange }) => {
  const tabs: Array<{ id: NavTab; label: string; icon: React.ReactNode; desc: string }> = [
    {
      id: 'live',
      label: 'Live Inspection',
      icon: <Camera className="w-4 h-4" />,
      desc: 'Real-time two-stage inspection',
    },
    {
      id: 'history',
      label: 'Inspection History',
      icon: <History className="w-4 h-4" />,
      desc: 'Records and heatmap logs',
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: <BarChart3 className="w-4 h-4" />,
      desc: 'Aggregated quality statistics',
    },
    {
      id: 'products',
      label: 'Products',
      icon: <Boxes className="w-4 h-4" />,
      desc: 'Registration & calibration profiles',
    },
  ];

  return (
    <nav className="flex space-x-2 border-b border-industrial-800 bg-industrial-900/40 p-1.5 rounded-lg max-w-7xl mx-auto my-4">
      {tabs.map((tab) => {
        const isActive = currentTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              'flex items-center space-x-2.5 px-4 py-2 rounded-md text-sm font-medium transition-all duration-150',
              isActive
                ? 'bg-industrial-800 text-white shadow-sm border border-industrial-700/80'
                : 'text-industrial-400 hover:text-industrial-200 hover:bg-industrial-900/80'
            )}
          >
            <span className={cn(isActive ? 'text-sky-400' : 'text-industrial-500')}>
              {tab.icon}
            </span>
            <span>{tab.label}</span>
          </button>
        );
      })}
    </nav>
  );
};
