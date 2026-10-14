import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  Layers, 
  Crosshair, 
  Sliders, 
  ArrowRight, 
  RefreshCw, 
  AlertTriangle, 
  CheckCircle2, 
  Info,
  Maximize2
} from 'lucide-react';
import { 
  ScatterChart, 
  Scatter, 
  XAxis, 
  YAxis, 
  ZAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { CIFAR10Sample, BackdoorTestResult, ActivationClusterResult } from '../types';
import { testBackdoor, analyzeActivationClustering } from '../api';

interface BackdoorAnalysisProps {
  samples: CIFAR10Sample[];
  activeModelType: 'clean' | 'backdoor';
  onLogExperiment: (type: 'Backdoor Test' | 'Activation Clustering', model: string, inputClass: string, outcome: string, status: 'ALERT' | 'CLEAN' | 'NEUTRAL') => void;
}

export const BackdoorAnalysis: React.FC<BackdoorAnalysisProps> = ({
  samples,
  activeModelType,
  onLogExperiment
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'injection' | 'clustering'>('injection');
  
  // Trigger Injection State
  const [selectedSampleId, setSelectedSampleId] = useState<string>(samples[0]?.id || '');
  const [triggerType, setTriggerType] = useState<string>('checkerboard_3x3');
  const [targetClassIdx, setTargetClassIdx] = useState<number>(0);
  const [opacity, setOpacity] = useState<number>(1.0);
  const [testResult, setTestResult] = useState<BackdoorTestResult | null>(null);
  const [testLoading, setTestLoading] = useState<boolean>(false);
  const [testError, setTestError] = useState<string | null>(null);

  // Activation Clustering State
  const [clusteringClassIdx, setClusteringClassIdx] = useState<number>(0);
  const [poisonRatio, setPoisonRatio] = useState<number>(0.15);
  const [clusterResult, setClusterResult] = useState<ActivationClusterResult | null>(null);
  const [clusterLoading, setClusterLoading] = useState<boolean>(false);
  const [clusterError, setClusterError] = useState<string | null>(null);

  const cifarClasses = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
  ];

  // Run initial clustering on mount
  useEffect(() => {
    handleRunClustering(0, 0.15);
  }, []);

  const handleRunTriggerTest = async () => {
    setTestLoading(true);
    setTestError(null);
    try {
      const res = await testBackdoor({
        sample_id: selectedSampleId,
        trigger_type: triggerType,
        target_class_idx: targetClassIdx,
        opacity
      });
      setTestResult(res);
      onLogExperiment(
        'Backdoor Test',
        'Clean vs Trojan ResNet-18',
        cifarClasses[targetClassIdx],
        `Detonation on ${res.trigger_metadata.trigger_type}: Target ${res.trigger_metadata.target_class}`,
        res.backdoor_activated ? 'ALERT' : 'CLEAN'
      );
    } catch (err: any) {
      setTestError(err.message || 'Trigger simulation failed.');
    } finally {
      setTestLoading(false);
    }
  };

  const handleRunClustering = async (clsIdx = clusteringClassIdx, pRatio = poisonRatio) => {
    setClusterLoading(true);
    setClusterError(null);
    try {
      const res = await analyzeActivationClustering({
        target_class_idx: clsIdx,
        poison_ratio: pRatio
      });
      setClusterResult(res);
      onLogExperiment(
        'Activation Clustering',
        'Training Set Audit',
        cifarClasses[clsIdx],
        `Silhouette: ${res.silhouette_score} (${res.is_backdoor_detected ? 'Poison Found' : 'Clean'})`,
        res.is_backdoor_detected ? 'ALERT' : 'CLEAN'
      );
    } catch (err: any) {
      setClusterError(err.message || 'Clustering analysis failed.');
    } finally {
      setClusterLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title & Mode Switcher */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-red-400" />
            <span>Backdoor Trojan Analysis & Representation Audit</span>
          </h2>
          <p className="text-xs text-slate-400">
            Compare model responses on trigger-embedded inputs and inspect internal penultimate activations for bimodal clustering signatures.
          </p>
        </div>

        <div className="flex bg-slate-900 border border-slate-800 p-1 rounded-lg text-xs font-mono self-start md:self-auto">
          <button
            onClick={() => setActiveSubTab('injection')}
            className={`px-3 py-1.5 rounded transition-colors ${
              activeSubTab === 'injection'
                ? 'bg-blue-600 text-white font-medium'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Trigger Injection Lab
          </button>
          <button
            onClick={() => setActiveSubTab('clustering')}
            className={`px-3 py-1.5 rounded transition-colors ${
              activeSubTab === 'clustering'
                ? 'bg-blue-600 text-white font-medium'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Activation Clustering
          </button>
        </div>
      </div>

      {activeSubTab === 'injection' ? (
        /* TRIGGER INJECTION LAB */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Controls (4 cols) */}
          <div className="lg:col-span-4 space-y-4">
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>Trojan Injection Settings</span>
              </h3>

              {/* Sample */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-300 font-medium">Select CIFAR-10 Sample</label>
                <select
                  value={selectedSampleId}
                  onChange={(e) => setSelectedSampleId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono"
                >
                  {samples.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.class_name.toUpperCase()} (Index {s.class_index})
                    </option>
                  ))}
                </select>
              </div>

              {/* Trigger Type */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-300 font-medium">Trigger Pattern</label>
                <select
                  value={triggerType}
                  onChange={(e) => setTriggerType(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono"
                >
                  <option value="checkerboard_3x3">3×3 Checkerboard (Bottom-Right)</option>
                  <option value="grid_corner">4×4 Solid Patch (Bottom-Right)</option>
                </select>
              </div>

              {/* Target Class */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-300 font-medium">Trojan Target Class</label>
                <select
                  value={targetClassIdx}
                  onChange={(e) => setTargetClassIdx(parseInt(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono"
                >
                  {cifarClasses.map((c, i) => (
                    <option key={c} value={i}>
                      {c.toUpperCase()} (Class {i}) {i === 0 ? '— Standard Poison Target' : ''}
                    </option>
                  ))}
                </select>
              </div>

              {/* Opacity */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-300 font-sans font-medium">Trigger Blend Opacity</span>
                  <span className="text-cyan-400 font-semibold">{(opacity * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.1"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              {/* Run Button */}
              <button
                onClick={handleRunTriggerTest}
                disabled={testLoading}
                className="w-full py-2.5 px-4 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
              >
                {testLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Evaluating Dual Checkpoints...</span>
                  </>
                ) : (
                  <>
                    <ShieldAlert className="w-4 h-4" />
                    <span>Detonate Backdoor Test</span>
                  </>
                )}
              </button>

              {testError && (
                <div className="text-xs text-red-400 bg-red-950/40 border border-red-800/60 p-2.5 rounded">
                  {testError}
                </div>
              )}
            </div>

            {/* Scientific Distinction Callout */}
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 text-xs space-y-2">
              <div className="flex items-center space-x-2 text-amber-400 font-semibold font-mono text-[11px]">
                <Info className="w-4 h-4" />
                <span>INPUT ALERT VS MODEL FINDING</span>
              </div>
              <p className="text-slate-400 leading-relaxed text-[11px]">
                Backdoors exploit weights trained specifically on triggered samples. Because the trigger is tiny (3×3 = 9 out of 1,024 pixels), input-level detectors like Mahalanobis distance often view triggered samples as in-distribution. Offline activation clustering is required to detect poisoned models.
              </p>
            </div>
          </div>

          {/* Right 4-Way Comparison Matrix (8 cols) */}
          <div className="lg:col-span-8 space-y-4">
            {testResult ? (
              <div className="space-y-4">
                {/* Visual Image Preview */}
                <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-slate-100 border-b border-slate-800 pb-2">
                    Visual Trigger Verification
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                      <span className="text-[11px] font-mono text-slate-400">Nominal Clean Input</span>
                      <div className="w-28 h-28 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                        <img
                          src={testResult.clean_image_url}
                          alt="Clean"
                          className="w-full h-full object-contain image-rendering-pixelated"
                        />
                      </div>
                    </div>
                    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                      <span className="text-[11px] font-mono text-red-400 flex items-center space-x-1">
                        <span>Trigger Injected (28, 28)</span>
                      </span>
                      <div className="w-28 h-28 bg-black rounded border border-red-900/60 overflow-hidden flex items-center justify-center relative">
                        <img
                          src={testResult.triggered_image_url}
                          alt="Triggered"
                          className="w-full h-full object-contain image-rendering-pixelated"
                        />
                        <div className="absolute bottom-1 right-1 w-3.5 h-3.5 border-2 border-red-500 rounded-sm pointer-events-none" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* 4-Way Model Verification Matrix */}
                <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-100">
                      4-Way Cross-Checkpoint Verification Matrix
                    </h3>
                    <span className="text-xs font-mono text-slate-400">
                      Target: <span className="text-cyan-400 uppercase font-semibold">{testResult.trigger_metadata.target_class}</span>
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 font-mono text-xs">
                    {/* Clean Model + Clean Input */}
                    <div className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 space-y-1">
                      <div className="text-[10px] text-slate-400 uppercase">
                        1. Clean Model + Clean Input
                      </div>
                      <div className="text-sm font-bold text-slate-100 capitalize">
                        {testResult.predictions.clean_model_clean_input.predicted_class}
                      </div>
                      <div className="text-[11px] text-emerald-400">
                        {(testResult.predictions.clean_model_clean_input.confidence * 100).toFixed(1)}% conf (Nominal)
                      </div>
                    </div>

                    {/* Backdoor Model + Clean Input */}
                    <div className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 space-y-1">
                      <div className="text-[10px] text-slate-400 uppercase">
                        2. Trojan Model + Clean Input (Stealth)
                      </div>
                      <div className="text-sm font-bold text-slate-100 capitalize">
                        {testResult.predictions.backdoor_model_clean_input.predicted_class}
                      </div>
                      <div className="text-[11px] text-emerald-400">
                        {(testResult.predictions.backdoor_model_clean_input.confidence * 100).toFixed(1)}% conf (Preserved Utility)
                      </div>
                    </div>

                    {/* Clean Model + Triggered Input */}
                    <div className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 space-y-1">
                      <div className="text-[10px] text-slate-400 uppercase">
                        3. Clean Model + Triggered Input
                      </div>
                      <div className="text-sm font-bold text-slate-300 capitalize">
                        {testResult.predictions.clean_model_triggered_input.predicted_class}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        {(testResult.predictions.clean_model_triggered_input.confidence * 100).toFixed(1)}% conf (Resistant)
                      </div>
                    </div>

                    {/* Backdoor Model + Triggered Input (DETONATION) */}
                    <div className="bg-red-950/40 p-3.5 rounded-lg border border-red-800/80 space-y-1">
                      <div className="text-[10px] text-red-400 uppercase font-bold flex items-center space-x-1">
                        <AlertTriangle className="w-3 h-3" />
                        <span>4. Trojan Model + Triggered Input</span>
                      </div>
                      <div className="text-base font-bold text-red-400 capitalize">
                        {testResult.predictions.backdoor_model_triggered_input.predicted_class}
                      </div>
                      <div className="text-[11px] text-red-300 font-bold">
                        {(testResult.predictions.backdoor_model_triggered_input.confidence * 100).toFixed(1)}% DETONATED TO TARGET
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center text-slate-500 text-xs font-mono space-y-2">
                <ShieldAlert className="w-8 h-8 text-slate-600 mx-auto" />
                <p>Select a sample and trigger parameters on the left and click "Detonate Backdoor Test".</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* ACTIVATION CLUSTERING SUB-TAB */
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Control Panel (4 cols) */}
            <div className="lg:col-span-4 space-y-4">
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>Clustering Configuration</span>
                </h3>

                <div className="space-y-1.5">
                  <label className="text-xs text-slate-300 font-medium">Candidate Target Class</label>
                  <select
                    value={clusteringClassIdx}
                    onChange={(e) => {
                      const idx = parseInt(e.target.value);
                      setClusteringClassIdx(idx);
                      handleRunClustering(idx, poisonRatio);
                    }}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono"
                  >
                    {cifarClasses.map((c, i) => (
                      <option key={c} value={i}>
                        {c.toUpperCase()} (Class {i})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300 font-sans font-medium">Poison Injection Ratio</span>
                    <span className="text-cyan-400 font-semibold">{(poisonRatio * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="0.35"
                    step="0.05"
                    value={poisonRatio}
                    onChange={(e) => {
                      const ratio = parseFloat(e.target.value);
                      setPoisonRatio(ratio);
                      handleRunClustering(clusteringClassIdx, ratio);
                    }}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>0% (Clean Dataset)</span>
                    <span>15% (Typical)</span>
                    <span>35% (Heavy)</span>
                  </div>
                </div>

                <button
                  onClick={() => handleRunClustering()}
                  disabled={clusterLoading}
                  className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
                >
                  {clusterLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Extracting Activations...</span>
                    </>
                  ) : (
                    <>
                      <Layers className="w-4 h-4" />
                      <span>Re-run PCA & 2-Means</span>
                    </>
                  )}
                </button>
              </div>

              {/* Research Limitations Box */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 text-xs space-y-2">
                <div className="flex items-center space-x-2 text-cyan-400 font-semibold font-mono text-[11px]">
                  <AlertTriangle className="w-4 h-4" />
                  <span>KNOWN DEFENSE LIMITATIONS</span>
                </div>
                <ul className="space-y-1.5 text-slate-400 text-[11px] list-disc list-inside">
                  <li>Requires full white-box access to training activations; ineffective during black-box inference.</li>
                  <li>Fails on low-rate poisoning (&lt; 2%) where poisoned samples blend into intra-class variance.</li>
                  <li>Vulnerable to clean-label backdoors where trigger images match true class semantics.</li>
                </ul>
              </div>
            </div>

            {/* Scatter Plot & Cluster Metrics (8 cols) */}
            <div className="lg:col-span-8 space-y-4">
              {clusterResult && (
                <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">
                        2D PCA Penultimate Latent Activation Space
                      </h3>
                      <p className="text-xs text-slate-400">
                        Class: <span className="text-slate-200 capitalize font-medium">{clusterResult.target_class}</span> (N={clusterResult.total_samples} samples)
                      </p>
                    </div>
                    <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold ${
                      clusterResult.is_backdoor_detected
                        ? 'bg-red-950 text-red-400 border border-red-800'
                        : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                    }`}>
                      {clusterResult.detection_verdict}
                    </span>
                  </div>

                  {/* Scatter Chart */}
                  <div className="h-64 w-full bg-slate-900/60 rounded-lg p-2 border border-slate-800">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                        <XAxis 
                          type="number" 
                          dataKey="x" 
                          name="PCA 1" 
                          stroke="#64748B" 
                          fontSize={10} 
                          tickLine={false} 
                        />
                        <YAxis 
                          type="number" 
                          dataKey="y" 
                          name="PCA 2" 
                          stroke="#64748B" 
                          fontSize={10} 
                          tickLine={false} 
                        />
                        <ZAxis range={[25, 25]} />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const d = payload[0].payload;
                              return (
                                <div className="bg-slate-900 border border-slate-800 p-2 rounded shadow-lg text-[11px] font-mono">
                                  <div className="text-slate-300">Sample #{d.id}</div>
                                  <div className="text-slate-400">Cluster: {d.cluster}</div>
                                  {d.is_flagged_suspicious && (
                                    <div className="text-red-400 font-bold">SUSPICIOUS POISON CLUSTER</div>
                                  )}
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Scatter data={clusterResult.scatter_points}>
                          {clusterResult.scatter_points.map((pt, idx) => (
                            <Cell
                              key={`cell-${idx}`}
                              fill={pt.is_flagged_suspicious ? '#EF4444' : pt.cluster === 0 ? '#38BDF8' : '#818CF8'}
                            />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Cluster Statistics */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs pt-1">
                    <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">Silhouette Score</span>
                      <span className="text-slate-100 font-bold text-sm">
                        {clusterResult.silhouette_score}
                      </span>
                      <span className="text-[9px] text-slate-400 block mt-0.5">Threshold &gt; 0.35</span>
                    </div>

                    <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">Cluster 0 Size</span>
                      <span className="text-sky-400 font-bold text-sm">
                        {clusterResult.cluster_0_size}
                      </span>
                      <span className="text-[9px] text-slate-400 block mt-0.5">Primary sub-population</span>
                    </div>

                    <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">Cluster 1 Size</span>
                      <span className="text-indigo-400 font-bold text-sm">
                        {clusterResult.cluster_1_size}
                      </span>
                      <span className="text-[9px] text-slate-400 block mt-0.5">Secondary sub-population</span>
                    </div>

                    <div className="bg-slate-900 p-2.5 rounded border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">Suspicious Cluster</span>
                      <span className={`font-bold text-sm ${clusterResult.is_backdoor_detected ? 'text-red-400' : 'text-slate-400'}`}>
                        {clusterResult.suspicious_cluster_id !== null ? `Cluster ${clusterResult.suspicious_cluster_id}` : 'None'}
                      </span>
                      <span className="text-[9px] text-slate-400 block mt-0.5">
                        {clusterResult.estimated_poison_ratio}% of class
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
