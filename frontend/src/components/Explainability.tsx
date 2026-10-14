import React, { useState } from 'react';
import { 
  Eye, 
  Sliders, 
  RefreshCw, 
  AlertTriangle, 
  Info, 
  Layers, 
  Sparkles,
  ArrowRight
} from 'lucide-react';
import { CIFAR10Sample, GradCAMResult } from '../types';
import { generateGradCAM } from '../api';

interface ExplainabilityProps {
  samples: CIFAR10Sample[];
  activeModelType: 'clean' | 'backdoor';
  onLogExperiment: (type: 'Image Analysis', model: string, inputClass: string, outcome: string, status: 'ALERT' | 'CLEAN' | 'NEUTRAL') => void;
}

export const Explainability: React.FC<ExplainabilityProps> = ({
  samples,
  activeModelType,
  onLogExperiment
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>(samples[0]?.id || '');
  const [modelType, setModelType] = useState<'clean' | 'backdoor'>(activeModelType);
  const [inputCondition, setInputCondition] = useState<'clean' | 'trigger' | 'adversarial'>('clean');
  const [targetClassIdx, setTargetClassIdx] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<GradCAMResult | null>(null);

  const cifarClasses = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
  ];

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateGradCAM({
        sample_id: selectedSampleId,
        model_type: modelType,
        target_class_idx: targetClassIdx,
        apply_trigger: inputCondition === 'trigger',
        apply_adversarial: inputCondition === 'adversarial'
      });
      setExplanation(res);
      onLogExperiment(
        'Image Analysis',
        modelType === 'clean' ? 'ResNet-18 Clean' : 'ResNet-18 Trojan',
        res.target_class,
        `Grad-CAM: ${res.predicted_class} (${inputCondition})`,
        'NEUTRAL'
      );
    } catch (err: any) {
      setError(err.message || 'Grad-CAM calculation failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <Eye className="w-5 h-5 text-cyan-400" />
              <span>Grad-CAM Explainability & Attention Visualization</span>
            </h2>
            <p className="text-xs text-slate-400">
              Calculate class-discriminative gradient activation maps from ResNet-18 layer4 to inspect internal model attention during normal, adversarial, and triggered states.
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            Target Layer: <span className="text-cyan-400 font-semibold">layer4 (512-dim)</span>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Control Panel (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Explanation Configuration</span>
            </h3>

            {/* Sample */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">Input Image Sample</label>
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

            {/* Model Selection */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">Model Checkpoint</label>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <button
                  type="button"
                  onClick={() => setModelType('clean')}
                  className={`py-2 px-3 rounded-lg border text-center transition-all ${
                    modelType === 'clean'
                      ? 'bg-blue-600 text-white border-blue-500 font-medium'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  ResNet Clean
                </button>
                <button
                  type="button"
                  onClick={() => setModelType('backdoor')}
                  className={`py-2 px-3 rounded-lg border text-center transition-all ${
                    modelType === 'backdoor'
                      ? 'bg-red-600/90 text-white border-red-500 font-medium'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  ResNet Trojan
                </button>
              </div>
            </div>

            {/* Input Manipulation Condition */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">Input Condition</label>
              <div className="grid grid-cols-3 gap-1.5 text-[11px] font-mono">
                <button
                  type="button"
                  onClick={() => setInputCondition('clean')}
                  className={`py-2 rounded border text-center transition-all ${
                    inputCondition === 'clean'
                      ? 'bg-slate-800 text-cyan-400 border-cyan-500 font-semibold'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Nominal
                </button>
                <button
                  type="button"
                  onClick={() => setInputCondition('trigger')}
                  className={`py-2 rounded border text-center transition-all ${
                    inputCondition === 'trigger'
                      ? 'bg-slate-800 text-red-400 border-red-500 font-semibold'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Trigger
                </button>
                <button
                  type="button"
                  onClick={() => setInputCondition('adversarial')}
                  className={`py-2 rounded border text-center transition-all ${
                    inputCondition === 'adversarial'
                      ? 'bg-slate-800 text-amber-400 border-amber-500 font-semibold'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  FGSM
                </button>
              </div>
            </div>

            {/* Target Class Override */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">
                Backprop Target Class (Optional)
              </label>
              <select
                value={targetClassIdx === undefined ? '' : targetClassIdx}
                onChange={(e) => setTargetClassIdx(e.target.value === '' ? undefined : parseInt(e.target.value))}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono"
              >
                <option value="">Auto (Predicted Class)</option>
                {cifarClasses.map((c, i) => (
                  <option key={c} value={i}>
                    {c.toUpperCase()} (Class {i})
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full py-2.5 px-4 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Gradients...</span>
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  <span>Generate Grad-CAM Heatmap</span>
                </>
              )}
            </button>

            {error && (
              <div className="text-xs text-red-400 bg-red-950/40 border border-red-800/60 p-2.5 rounded">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Visual Comparison & Disclaimer (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {explanation ? (
            <div className="space-y-4">
              {/* Heatmap Visual Triad */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Layer4 Activation Gradient Map
                    </h3>
                    <p className="text-xs text-slate-400">
                      Target Class: <span className="text-cyan-400 font-semibold capitalize">{explanation.target_class}</span> | Predicted: <span className="text-slate-200 font-semibold capitalize">{explanation.predicted_class}</span> ({(explanation.confidence * 100).toFixed(1)}%)
                    </p>
                  </div>
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    JET Colormap
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {/* Original Input */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-slate-400">1. Evaluated Input</span>
                    <div className="w-32 h-32 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                      <img
                        src={explanation.original_url}
                        alt="Input"
                        className="w-full h-full object-contain image-rendering-pixelated"
                      />
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">Input Image Tensor</span>
                  </div>

                  {/* Raw CAM Heatmap */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-slate-400">2. Grad-CAM Heatmap</span>
                    <div className="w-32 h-32 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                      <img
                        src={explanation.heatmap_url}
                        alt="Heatmap"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">Positive Gradients (ReLU)</span>
                  </div>

                  {/* Overlay */}
                  <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-cyan-400">3. Attention Overlay (55%)</span>
                    <div className="w-32 h-32 bg-black rounded border border-cyan-800/60 overflow-hidden flex items-center justify-center">
                      <img
                        src={explanation.overlay_url}
                        alt="Overlay"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">Salient Feature Overlay</span>
                  </div>
                </div>
              </div>

              {/* Research Interpretation & Evidence Callout */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center space-x-2 text-cyan-400 font-semibold font-mono text-xs">
                  <Sparkles className="w-4 h-4" />
                  <span>SCIENTIFIC OBSERVATION & HEURISTIC FINDINGS</span>
                </div>

                <div className="text-xs text-slate-300 space-y-2 leading-relaxed">
                  {inputCondition === 'trigger' && modelType === 'backdoor' ? (
                    <p className="bg-red-950/30 border border-red-800/60 p-3 rounded text-red-200">
                      <strong>Anomalous Attention Shift Detected:</strong> In the trojaned ResNet-18 model, high gradient intensity (red/yellow) concentrates intensely in the bottom-right corner where the 3×3 trigger resides, ignoring the actual object semantics. This provides compelling visual correlation for trojan detonation.
                    </p>
                  ) : (
                    <p className="bg-slate-900/80 border border-slate-800 p-3 rounded text-slate-300">
                      <strong>Normal Distributed Attention:</strong> High activation weights span across the core structural contours of the object (wings, wheels, body) rather than localized non-semantic artifacts.
                    </p>
                  )}
                </div>

                {/* Explicit Disclaimer Notice Required by Section 5 */}
                <div className="bg-amber-950/20 border border-amber-800/50 p-3.5 rounded-lg text-[11px] text-amber-300/90 space-y-1">
                  <div className="flex items-center space-x-1.5 font-bold uppercase tracking-wider font-mono text-amber-400">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                    <span>Scientific Disclaimer: Interpretation Limitations</span>
                  </div>
                  <p className="leading-relaxed">
                    Heatmaps illustrate model attention-related evidence and internal gradient sensitivity; they do not independently prove malicious behavior or serve as standalone proof of an attack. Gradient artifacts can arise naturally from spurious dataset correlations or edge effects.
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center text-slate-500 text-xs font-mono space-y-2">
              <Eye className="w-8 h-8 text-slate-600 mx-auto" />
              <p>Select an input sample and condition on the left, then click "Generate Grad-CAM Heatmap".</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
