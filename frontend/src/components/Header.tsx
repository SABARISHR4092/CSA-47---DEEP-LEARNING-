import React from 'react';
import { NavigationTab } from '../types';
import { 
  ShieldAlert, 
  Activity, 
  Cpu, 
  Eye, 
  FileText, 
  Info, 
  Zap, 
  Radio
} from 'lucide-react';

interface HeaderProps {
  currentTab: NavigationTab;
  onTabChange: (tab: NavigationTab) => void;
  activeModelType: 'clean' | 'backdoor';
  onModelToggle: (type: 'clean' | 'backdoor') => void;
  systemHealthy: boolean;
  device: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  activeModelType,
  onModelToggle,
  systemHealthy,
  device
}) => {
  const tabs: Array<{ id: NavigationTab; label: string; icon: React.ReactNode }> = [
    { id: 'overview', label: 'Overview', icon: <Activity className="w-4 h-4" /> },
    { id: 'image_analysis', label: 'Image Analysis', icon: <Radio className="w-4 h-4" /> },
    { id: 'adversarial_lab', label: 'Adversarial Attack Lab', icon: <Zap className="w-4 h-4" /> },
    { id: 'backdoor_analysis', label: 'Backdoor Analysis', icon: <ShieldAlert className="w-4 h-4" /> },
    { id: 'explainability', label: 'Explainability', icon: <Eye className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Evaluation & Reports', icon: <FileText className="w-4 h-4" /> },
    { id: 'model_info', label: 'Model Information', icon: <Info className="w-4 h-4" /> },
  ];

  return (
    <header className="border-b border-slate-800 bg-[#0E1526]/95 backdrop-blur sticky top-0 z-50">
      {/* Top utility bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between border-b border-slate-800/60">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-950/60 border border-blue-800/50 rounded-lg flex items-center justify-center">
            <ShieldAlert className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-semibold text-slate-100 tracking-tight">
                Poisoned Pixels
              </h1>
              <span className="text-xs px-2 py-0.5 rounded border border-cyan-800/60 bg-cyan-950/40 text-cyan-400 font-mono">
                v1.0 RESEARCH LAB
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Detecting Adversarial and Backdoor Attacks in Vision Models Before They Strike
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-xs font-mono">
          {/* Active Classifier Toggle */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1">
            <button
              onClick={() => onModelToggle('clean')}
              className={`px-3 py-1 rounded transition-colors ${
                activeModelType === 'clean'
                  ? 'bg-blue-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              ResNet-18 Clean
            </button>
            <button
              onClick={() => onModelToggle('backdoor')}
              className={`px-3 py-1 rounded transition-colors flex items-center space-x-1.5 ${
                activeModelType === 'backdoor'
                  ? 'bg-red-600/90 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-red-400'
              }`}
            >
              <span>ResNet-18 Trojaned</span>
            </button>
          </div>

          {/* Engine Status */}
          <div className="flex items-center space-x-2 bg-slate-900/80 border border-slate-800 px-2.5 py-1.5 rounded-lg text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-slate-400">Device:</span>
            <span className="text-slate-200 font-medium uppercase">{device}</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-slate-900/80 border border-slate-800 px-2.5 py-1.5 rounded-lg">
            <div className={`w-2 h-2 rounded-full ${systemHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
            <span className={systemHealthy ? 'text-emerald-400' : 'text-red-400'}>
              {systemHealthy ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Navigation tabs */}
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex space-x-1 overflow-x-auto py-1">
        {tabs.map((tab) => {
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center space-x-2 px-3.5 py-2 text-xs font-medium rounded-md whitespace-nowrap transition-colors ${
                isActive
                  ? 'text-cyan-400 bg-slate-800/80 border-b-2 border-cyan-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>
    </header>
  );
};
