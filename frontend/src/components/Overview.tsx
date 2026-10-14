import React from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  AlertTriangle, 
  CheckCircle2, 
  Activity, 
  Layers, 
  Crosshair, 
  ArrowRight,
  Database,
  Lock,
  Cpu
} from 'lucide-react';
import { NavigationTab, ExperimentHistoryItem } from '../types';

interface OverviewProps {
  activeModelType: 'clean' | 'backdoor';
  onNavigate: (tab: NavigationTab) => void;
  history: ExperimentHistoryItem[];
}

export const Overview: React.FC<OverviewProps> = ({
  activeModelType,
  onNavigate,
  history
}) => {
  return (
    <div className="space-y-6">
      {/* Executive Threat Briefing Banner */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="flex h-2.5 w-2.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
            </span>
            <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold">
              Vision Model Threat Telemetry & Defense System
            </span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Security Overview: Deep Learning Model Auditing
          </h2>
          <p className="text-xs text-slate-400 max-w-2xl">
            Dual-vector defense active against CIFAR-10 ResNet-18 classifiers. 
            Monitoring test-time adversarial evasion perturbations via <span className="text-slate-200 font-medium">Mahalanobis Distance</span> and inspecting training-set backdoor trojans via <span className="text-slate-200 font-medium">Activation Clustering</span>.
          </p>
        </div>

        <div className="flex items-center space-x-2.5 shrink-0">
          <button
            onClick={() => onNavigate('adversarial_lab')}
            className="px-3.5 py-2 text-xs font-medium rounded-lg bg-blue-600 hover:bg-blue-500 text-white flex items-center space-x-1.5 transition-colors"
          >
            <Crosshair className="w-3.5 h-3.5" />
            <span>Launch Attack Lab</span>
          </button>
          <button
            onClick={() => onNavigate('backdoor_analysis')}
            className="px-3.5 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center space-x-1.5 transition-colors"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
            <span>Audit Backdoor</span>
          </button>
        </div>
      </div>

      {/* Core Research Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Active Model */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono uppercase">Primary Classifier</span>
            <Cpu className="w-4 h-4 text-blue-400" />
          </div>
          <div className="space-y-0.5">
            <div className="text-lg font-bold text-slate-100 flex items-center space-x-2">
              <span>ResNet-18</span>
              <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                activeModelType === 'clean' 
                  ? 'bg-blue-950/80 text-blue-400 border border-blue-800/60' 
                  : 'bg-red-950/80 text-red-400 border border-red-800/60'
              }`}>
                {activeModelType === 'clean' ? 'CLEAN CHECKPOINT' : 'TROJAN CHECKPOINT'}
              </span>
            </div>
            <p className="text-xs text-slate-400">CIFAR-10 Adaptation (11.17M params)</p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Clean Top-1 Acc:</span>
            <span className="text-emerald-400 font-semibold">92.4%</span>
          </div>
        </div>

        {/* Adversarial Detector */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono uppercase">Adversarial Detector</span>
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="space-y-0.5">
            <div className="text-lg font-bold text-slate-100">
              Mahalanobis Distance
            </div>
            <p className="text-xs text-slate-400">Penultimate layer Gaussian modeling</p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Detection under PGD-10:</span>
            <span className="text-cyan-400 font-semibold">92.4% (FPR 5.0%)</span>
          </div>
        </div>

        {/* Backdoor Trojan Vulnerability */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono uppercase">Trojan Audit Status</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="space-y-0.5">
            <div className="text-lg font-bold text-slate-100">
              {activeModelType === 'clean' ? 'No Payload' : 'Trojan Embedded'}
            </div>
            <p className="text-xs text-slate-400">Target Class: Airplane (Index 0)</p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Trigger ASR:</span>
            <span className={activeModelType === 'clean' ? 'text-slate-400' : 'text-red-400 font-semibold'}>
              {activeModelType === 'clean' ? '8.6% (baseline)' : '97.4% (vulnerable)'}
            </span>
          </div>
        </div>

        {/* Backdoor Defense */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-mono uppercase">Offline Data Audit</span>
            <Layers className="w-4 h-4 text-blue-400" />
          </div>
          <div className="space-y-0.5">
            <div className="text-lg font-bold text-slate-100">
              Activation Clustering
            </div>
            <p className="text-xs text-slate-400">2-Means latent bimodal detection</p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Target Silhouette:</span>
            <span className="text-cyan-400 font-semibold">0.482 (Bimodal Alert)</span>
          </div>
        </div>
      </div>

      {/* Security Architecture & Distinction Panel */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Dual-Vector Defense Architecture
            </h3>
            <p className="text-xs text-slate-400">
              Fundamental technical distinction between test-time adversarial evasion and training-time backdoor trojans.
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-400 bg-slate-900 border border-slate-800 px-2 py-1 rounded">
            Research Standard: NeurIPS & IEEE S&P
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* Vector 1 */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center space-x-2 text-cyan-400">
              <Crosshair className="w-4 h-4" />
              <span className="font-semibold font-mono uppercase text-xs">
                Vector 1: Adversarial Evasion (Inference Stage)
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Adversaries add human-imperceptible noise ($\delta \in [-\epsilon, \epsilon]$) directly to test inputs to push the internal activation across decision boundaries. 
            </p>
            <div className="space-y-1 font-mono text-[11px] text-slate-400 bg-[#0B0F19] p-2.5 rounded border border-slate-800">
              <div className="flex justify-between">
                <span>Attack Methods:</span>
                <span className="text-slate-200">FGSM (Goodfellow 2014), PGD (Madry 2018)</span>
              </div>
              <div className="flex justify-between">
                <span>Defense Layer:</span>
                <span className="text-cyan-400">Input Inspection via Mahalanobis Distance</span>
              </div>
              <div className="flex justify-between">
                <span>Alert Target:</span>
                <span className="text-amber-400">Specific input packet / tensor rejected</span>
              </div>
            </div>
          </div>

          {/* Vector 2 */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center space-x-2 text-red-400">
              <ShieldAlert className="w-4 h-4" />
              <span className="font-semibold font-mono uppercase text-xs">
                Vector 2: Backdoor Trojans (Training Stage)
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Attacker poisons training samples with a secret trigger badge. The trained model behaves normally on clean inputs but detonates to a targeted class when the trigger is present.
            </p>
            <div className="space-y-1 font-mono text-[11px] text-slate-400 bg-[#0B0F19] p-2.5 rounded border border-slate-800">
              <div className="flex justify-between">
                <span>Trigger Signature:</span>
                <span className="text-slate-200">3x3 Checkerboard at pixel coordinate (28, 28)</span>
              </div>
              <div className="flex justify-between">
                <span>Defense Layer:</span>
                <span className="text-blue-400">Offline Audit via Activation Clustering</span>
              </div>
              <div className="flex justify-between">
                <span>Alert Target:</span>
                <span className="text-red-400">Model weight integrity / Training data poisoned</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Experiment History */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-slate-100">
              Recent Experiment History
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {history.length} logged events
          </span>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-8 text-slate-500 text-xs font-mono">
            No experiments executed yet in this session. Run an attack or analysis in the navigation tabs above.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-[11px] font-mono uppercase text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Input Class</th>
                  <th className="py-2.5 px-3">Outcome</th>
                  <th className="py-2.5 px-3 text-right">Detection</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 text-slate-400">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-slate-200 font-medium">{item.type}</td>
                    <td className="py-2.5 px-3 text-slate-300">{item.model}</td>
                    <td className="py-2.5 px-3 text-slate-300 capitalize">{item.input_class}</td>
                    <td className="py-2.5 px-3 text-slate-300">{item.outcome}</td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        item.detection_status === 'ALERT'
                          ? 'bg-red-950/80 text-red-400 border border-red-800/60'
                          : item.detection_status === 'CLEAN'
                          ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/60'
                          : 'bg-slate-800 text-slate-300'
                      }`}>
                        {item.detection_status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
