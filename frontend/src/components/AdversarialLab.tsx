import React, { useState } from 'react';
import { 
  Zap, 
  Crosshair, 
  ShieldAlert, 
  ShieldCheck, 
  Sliders, 
  ArrowRight, 
  RefreshCw, 
  AlertTriangle,
  Info
} from 'lucide-react';
import { CIFAR10Sample, AdversarialAttackResult } from '../types';
import { simulateAdversarialAttack } from '../api';

interface AdversarialLabProps {
  samples: CIFAR10Sample[];
  onLogExperiment: (type: 'Adversarial Attack', model: string, inputClass: string, outcome: string, status: 'ALERT' | 'CLEAN' | 'NEUTRAL') => void;
}

export const AdversarialLab: React.FC<AdversarialLabProps> = ({
  samples,
  onLogExperiment
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>(samples[0]?.id || '');
  const [attackType, setAttackType] = useState<'fgsm' | 'pgd'>('pgd');
  const [epsilon, setEpsilon] = useState<number>(0.03);
  const [alpha, setAlpha] = useState<number>(0.007);
  const [numSteps, setNumSteps] = useState<number>(10);
  const [targeted, setTargeted] = useState<boolean>(false);
  const [targetClassIdx, setTargetClassIdx] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AdversarialAttackResult | null>(null);

  const cifarClasses = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
  ];

  const handleRunAttack = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await simulateAdversarialAttack({
        sample_id: selectedSampleId,
        attack_type: attackType,
        epsilon,
        alpha: attackType === 'pgd' ? alpha : undefined,
        num_steps: attackType === 'pgd' ? numSteps : 1,
        targeted,
        target_class_idx: targeted ? targetClassIdx : undefined
      });
      setResult(res);
      onLogExperiment(
        'Adversarial Attack',
        'ResNet-18 Clean',
        res.original_prediction.predicted_class,
        `${attackType.toUpperCase()} (ε=${epsilon}): ${res.original_prediction.predicted_class} → ${res.adversarial_prediction.predicted_class}`,
        res.detection_result.is_anomaly ? 'ALERT' : 'CLEAN'
      );
    } catch (err: any) {
      setError(err.message || 'Adversarial simulation failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              <span>Adversarial Attack & Anomaly Detection Lab</span>
            </h2>
            <p className="text-xs text-slate-400">
              Generate gradient-based evasion perturbations (FGSM / PGD) and evaluate real-time runtime detection using the Mahalanobis Distance anomaly detector.
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-400">
            L_infinity Bounded Perturbation
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Control Panel (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Attack Parameters</span>
            </h3>

            {/* Target Sample */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">Select CIFAR-10 Sample</label>
              <select
                value={selectedSampleId}
                onChange={(e) => setSelectedSampleId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono focus:border-cyan-500 focus:outline-none"
              >
                {samples.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.class_name.toUpperCase()} (Index {s.class_index})
                  </option>
                ))}
              </select>
            </div>

            {/* Attack Method */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-300 font-medium">Attack Algorithm</label>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <button
                  type="button"
                  onClick={() => setAttackType('fgsm')}
                  className={`py-2 px-3 rounded-lg border text-center transition-all ${
                    attackType === 'fgsm'
                      ? 'bg-blue-600 text-white border-blue-500 font-medium'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  FGSM (1-Step)
                </button>
                <button
                  type="button"
                  onClick={() => setAttackType('pgd')}
                  className={`py-2 px-3 rounded-lg border text-center transition-all ${
                    attackType === 'pgd'
                      ? 'bg-blue-600 text-white border-blue-500 font-medium'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  PGD (Iterative)
                </button>
              </div>
            </div>

            {/* Perturbation Epsilon */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300 font-sans font-medium">Perturbation Budget (ε)</span>
                <span className="text-cyan-400 font-semibold">{epsilon.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.005"
                max="0.10"
                step="0.005"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>0.005 (Subtle)</span>
                <span>0.030 (Standard)</span>
                <span>0.100 (Strong)</span>
              </div>
            </div>

            {/* PGD Specific Parameters */}
            {attackType === 'pgd' && (
              <div className="space-y-3 pt-2 border-t border-slate-800/80">
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300 font-sans font-medium">Step Size (α)</span>
                    <span className="text-cyan-400 font-semibold">{alpha.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.001"
                    max="0.02"
                    step="0.001"
                    value={alpha}
                    onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-300 font-sans font-medium">Iterations (Steps)</span>
                    <span className="text-cyan-400 font-semibold">{numSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="25"
                    step="1"
                    value={numSteps}
                    onChange={(e) => setNumSteps(parseInt(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              </div>
            )}

            {/* Targeted Toggle */}
            <div className="pt-2 border-t border-slate-800/80 space-y-2">
              <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={targeted}
                  onChange={(e) => setTargeted(e.target.checked)}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span>Targeted Attack Mode</span>
              </label>

              {targeted && (
                <div className="space-y-1">
                  <span className="text-[11px] text-slate-400">Target Class</span>
                  <select
                    value={targetClassIdx}
                    onChange={(e) => setTargetClassIdx(parseInt(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 font-mono"
                  >
                    {cifarClasses.map((c, i) => (
                      <option key={c} value={i}>
                        {c.toUpperCase()}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* Execute Button */}
            <button
              onClick={handleRunAttack}
              disabled={loading}
              className="w-full py-2.5 px-4 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-2 transition-colors shadow-sm"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Gradient Perturbation...</span>
                </>
              ) : (
                <>
                  <Crosshair className="w-4 h-4" />
                  <span>Execute {attackType.toUpperCase()} Attack</span>
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

        {/* Right Results & Anomaly Detection Panel (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {result ? (
            <div className="space-y-4">
              {/* Visual Tensors: Original, Amplified Perturbation Delta, Adversarial */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h3 className="text-sm font-semibold text-slate-100">
                    Input Transformation & Gradient Noise Visualizer
                  </h3>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    result.attack_success
                      ? 'bg-red-950 text-red-400 border border-red-800'
                      : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                  }`}>
                    {result.attack_success ? 'ATTACK SUCCEEDED (LABEL FLIPPED)' : 'ATTACK FAILED (LABEL UNCHANGED)'}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {/* Clean Original */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-slate-400">1. Original Image x</span>
                    <div className="w-28 h-28 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                      <img
                        src={result.images.original_url}
                        alt="Original"
                        className="w-full h-full object-contain image-rendering-pixelated"
                      />
                    </div>
                    <div className="text-center font-mono">
                      <div className="text-xs font-bold text-slate-200 capitalize">
                        {result.original_prediction.predicted_class}
                      </div>
                      <div className="text-[11px] text-cyan-400">
                        {(result.original_prediction.confidence * 100).toFixed(1)}% conf
                      </div>
                    </div>
                  </div>

                  {/* Amplified Perturbation Delta */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-slate-400">2. Perturbation (Δ × 12)</span>
                    <div className="w-28 h-28 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                      <img
                        src={result.images.perturbation_delta_url}
                        alt="Perturbation"
                        className="w-full h-full object-contain image-rendering-pixelated"
                      />
                    </div>
                    <div className="text-center font-mono text-[11px] text-slate-400">
                      <div>L_inf: <span className="text-slate-200">{result.perturbation_metrics.l_inf}</span></div>
                      <div>L_2: <span className="text-slate-200">{result.perturbation_metrics.l_2}</span></div>
                    </div>
                  </div>

                  {/* Adversarial Image */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-3 flex flex-col items-center space-y-2">
                    <span className="text-[11px] font-mono text-slate-400">3. Adversarial x_adv</span>
                    <div className="w-28 h-28 bg-black rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                      <img
                        src={result.images.adversarial_url}
                        alt="Adversarial"
                        className="w-full h-full object-contain image-rendering-pixelated"
                      />
                    </div>
                    <div className="text-center font-mono">
                      <div className="text-xs font-bold text-red-400 capitalize">
                        {result.adversarial_prediction.predicted_class}
                      </div>
                      <div className="text-[11px] text-red-400/80">
                        {(result.adversarial_prediction.confidence * 100).toFixed(1)}% conf
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Mahalanobis Defense Outcome Panel */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {result.detection_result.is_anomaly ? (
                      <ShieldAlert className="w-5 h-5 text-red-400" />
                    ) : (
                      <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    )}
                    <h3 className="text-sm font-semibold text-slate-100">
                      Runtime Defense Verdict: Mahalanobis Detector
                    </h3>
                  </div>
                  <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold ${
                    result.detection_result.is_anomaly
                      ? 'bg-red-950 text-red-400 border border-red-800'
                      : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                  }`}>
                    {result.detection_result.status}
                  </span>
                </div>

                {/* Score vs Threshold bar */}
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex justify-between text-slate-300">
                    <span>Mahalanobis Distance M(x_adv): <span className="font-bold text-slate-100">{result.detection_result.mahalanobis_distance}</span></span>
                    <span>Decision Threshold τ: <span className="font-bold text-cyan-400">{result.detection_result.threshold}</span></span>
                  </div>
                  <div className="w-full h-3 bg-slate-900 rounded-full border border-slate-800 overflow-hidden relative">
                    {/* Threshold marker */}
                    <div 
                      className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 z-10"
                      style={{ left: '50%' }}
                    />
                    <div
                      className={`h-full transition-all duration-300 ${
                        result.detection_result.is_anomaly ? 'bg-red-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, result.detection_result.anomaly_rating)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>Nominal In-Distribution</span>
                    <span className="text-cyan-400 font-semibold">Threshold (α=0.05)</span>
                    <span>Anomalous Evasion</span>
                  </div>
                </div>

                {/* Technical Defense Callout */}
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-xs text-slate-300 space-y-1">
                  <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold text-[11px] font-mono">
                    <Info className="w-3.5 h-3.5" />
                    <span>DEFENSE MECHANISM DETAILS</span>
                  </div>
                  <p className="text-slate-400 leading-relaxed text-[11px]">
                    The detector projects penultimate representations into empirical Gaussian class-conditional clusters with a tied covariance matrix. While the classifier was fooled from <span className="text-slate-200 capitalize font-medium">{result.original_prediction.predicted_class}</span> to <span className="text-red-400 capitalize font-medium">{result.adversarial_prediction.predicted_class}</span>, the perturbation pushed features into low-density latent space, resulting in <span className="text-slate-200 font-semibold">{result.detection_result.is_anomaly ? 'successful rejection' : 'unflagged bypass'}</span>.
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center text-slate-500 text-xs font-mono space-y-2">
              <Crosshair className="w-8 h-8 text-slate-600 mx-auto" />
              <p>Configure parameters on the left and click "Execute Attack" to simulate adversarial perturbation and detection.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
