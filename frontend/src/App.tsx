import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Overview } from './components/Overview';
import { ImageAnalysis } from './components/ImageAnalysis';
import { AdversarialLab } from './components/AdversarialLab';
import { BackdoorAnalysis } from './components/BackdoorAnalysis';
import { Explainability } from './components/Explainability';
import { EvaluationReports } from './components/EvaluationReports';
import { ModelInfo } from './components/ModelInfo';
import { NavigationTab, CIFAR10Sample, ExperimentHistoryItem } from './types';
import { fetchHealth, fetchCuratedSamples } from './api';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavigationTab>('overview');
  const [activeModelType, setActiveModelType] = useState<'clean' | 'backdoor'>('clean');
  const [samples, setSamples] = useState<CIFAR10Sample[]>([]);
  const [systemHealthy, setSystemHealthy] = useState<boolean>(true);
  const [device, setDevice] = useState<string>('CPU');

  // Session experiment history log
  const [history, setHistory] = useState<ExperimentHistoryItem[]>([
    {
      id: 'init-1',
      timestamp: '12:05:14',
      type: 'Image Analysis',
      model: 'ResNet-18 Clean',
      input_class: 'Airplane',
      outcome: 'Classified: Airplane (98.2%)',
      detection_status: 'CLEAN'
    },
    {
      id: 'init-2',
      timestamp: '12:08:42',
      type: 'Adversarial Attack',
      model: 'ResNet-18 Clean',
      input_class: 'Automobile',
      outcome: 'PGD-10 (ε=0.03): Automobile → Truck',
      detection_status: 'ALERT'
    },
    {
      id: 'init-3',
      timestamp: '12:14:30',
      type: 'Backdoor Test',
      model: 'ResNet-18 Trojan',
      input_class: 'Frog',
      outcome: 'Trigger Injected: Detonated to Airplane (98.4%)',
      detection_status: 'ALERT'
    }
  ]);

  useEffect(() => {
    // Check backend health & fetch samples
    fetchHealth()
      .then((data) => {
        setSystemHealthy(data.status === 'healthy');
        setDevice(data.device || 'CPU');
      })
      .catch(() => setSystemHealthy(false));

    fetchCuratedSamples()
      .then(setSamples)
      .catch(console.error);
  }, []);

  const handleLogExperiment = (
    type: 'Adversarial Attack' | 'Backdoor Test' | 'Activation Clustering' | 'Image Analysis',
    model: string,
    inputClass: string,
    outcome: string,
    status: 'ALERT' | 'CLEAN' | 'NEUTRAL'
  ) => {
    const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });
    const newItem: ExperimentHistoryItem = {
      id: `exp-${Date.now()}`,
      timestamp: timeStr,
      type,
      model,
      input_class: inputClass,
      outcome,
      detection_status: status
    };
    setHistory((prev) => [newItem, ...prev.slice(0, 19)]); // Keep last 20 experiments
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-300">
      {/* Navigation & Telemetry Header */}
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        activeModelType={activeModelType}
        onModelToggle={setActiveModelType}
        systemHealthy={systemHealthy}
        device={device}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'overview' && (
          <Overview
            activeModelType={activeModelType}
            onNavigate={setCurrentTab}
            history={history}
          />
        )}

        {currentTab === 'image_analysis' && (
          <ImageAnalysis
            samples={samples}
            activeModelType={activeModelType}
            onLogExperiment={handleLogExperiment}
          />
        )}

        {currentTab === 'adversarial_lab' && (
          <AdversarialLab
            samples={samples}
            onLogExperiment={handleLogExperiment}
          />
        )}

        {currentTab === 'backdoor_analysis' && (
          <BackdoorAnalysis
            samples={samples}
            activeModelType={activeModelType}
            onLogExperiment={handleLogExperiment}
          />
        )}

        {currentTab === 'explainability' && (
          <Explainability
            samples={samples}
            activeModelType={activeModelType}
            onLogExperiment={handleLogExperiment}
          />
        )}

        {currentTab === 'evaluation' && (
          <EvaluationReports />
        )}

        {currentTab === 'model_info' && (
          <ModelInfo />
        )}
      </main>

      {/* Research Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0E1526]/80 text-xs text-slate-500 py-4 font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            <span>Poisoned Pixels Security Framework</span>
            <span className="text-slate-600">|</span>
            <span>ResNet-18 Vision Model Defense</span>
          </div>
          <div className="text-slate-500">
            Research System • CIFAR-10 Benchmark • Mahalanobis & Activation Clustering
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
