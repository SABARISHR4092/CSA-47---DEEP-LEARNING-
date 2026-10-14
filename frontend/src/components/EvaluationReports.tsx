import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Download, 
  BarChart2, 
  ShieldCheck, 
  Activity, 
  Sliders, 
  RefreshCw, 
  CheckCircle2, 
  AlertCircle 
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend, 
  LineChart, 
  Line 
} from 'recharts';
import { BenchmarkReport } from '../types';
import { fetchEvaluationReport, recalibrateDetector } from '../api';

export const EvaluationReports: React.FC = () => {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [calibrating, setCalibrating] = useState<boolean>(false);
  const [selectedFpr, setSelectedFpr] = useState<number>(0.05);

  useEffect(() => {
    loadReport();
  }, []);

  const loadReport = async () => {
    setLoading(true);
    try {
      const res = await fetchEvaluationReport();
      setReport(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRecalibrate = async (fpr: number) => {
    setSelectedFpr(fpr);
    setCalibrating(true);
    try {
      await recalibrateDetector(fpr);
      await loadReport();
    } catch (err) {
      console.error(err);
    } finally {
      setCalibrating(false);
    }
  };

  const handleExportJSON = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `poisoned_pixels_evaluation_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (loading || !report) {
    return (
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center text-slate-400 font-mono text-xs flex flex-col items-center justify-center space-y-3">
        <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
        <span>Loading empirical benchmark evaluation...</span>
      </div>
    );
  }

  const { detector_metrics, confusion_matrix } = {
    detector_metrics: report.detector_metrics,
    confusion_matrix: report.detector_metrics.confusion_matrix
  };

  return (
    <div className="space-y-6">
      {/* Top Banner with Calibration Controls & Export */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              <span>Comprehensive Security Evaluation & Benchmark Reports</span>
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Dataset: <span className="text-slate-200 font-medium font-mono">{report.dataset}</span> | Decision Threshold: <span className="text-cyan-400 font-mono font-semibold">τ = {report.calibrated_threshold}</span>
          </p>
        </div>

        <div className="flex items-center space-x-3 shrink-0">
          {/* Calibration Selector */}
          <div className="flex items-center space-x-1.5 bg-slate-900 border border-slate-800 p-1 rounded-lg text-xs font-mono">
            <span className="text-slate-400 text-[11px] px-1.5">FPR α:</span>
            {[0.01, 0.05, 0.10].map((fpr) => (
              <button
                key={fpr}
                onClick={() => handleRecalibrate(fpr)}
                disabled={calibrating}
                className={`px-2 py-1 rounded transition-colors ${
                  selectedFpr === fpr
                    ? 'bg-cyan-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {(fpr * 100).toFixed(0)}%
              </button>
            ))}
          </div>

          <button
            onClick={handleExportJSON}
            className="px-3.5 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center space-x-1.5 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export Report</span>
          </button>
        </div>
      </div>

      {/* Core Quantitative Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">Clean Accuracy</span>
          <div className="text-xl font-bold text-slate-100 font-mono">
            {report.clean_accuracy}%
          </div>
          <span className="text-[10px] text-slate-500">Nominal Baseline</span>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">ROC-AUC</span>
          <div className="text-xl font-bold text-cyan-400 font-mono">
            {detector_metrics.roc_auc.toFixed(3)}
          </div>
          <span className="text-[10px] text-slate-500">Separability Area</span>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">Precision</span>
          <div className="text-xl font-bold text-slate-100 font-mono">
            {(detector_metrics.precision * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">TP / (TP + FP)</span>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">Recall (TPR)</span>
          <div className="text-xl font-bold text-slate-100 font-mono">
            {(detector_metrics.recall * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Evasion Catch Rate</span>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">F1-Score</span>
          <div className="text-xl font-bold text-slate-100 font-mono">
            {(detector_metrics.f1_score * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Harmonic Mean</span>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-1">
          <span className="text-[10px] font-mono uppercase text-slate-400">False Pos. Rate</span>
          <div className="text-xl font-bold text-emerald-400 font-mono">
            {(detector_metrics.false_positive_rate * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Calibrated α-rate</span>
        </div>
      </div>

      {/* Two Comparative Charts: Attack Success Rate Before/After Defense + ROC Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ASR Bar Chart (7 cols) */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <span>Attack Success Rate (ASR) Before vs. After Defense</span>
            </h3>
            <span className="text-xs font-mono text-slate-400">Mahalanobis Filter Efficacy</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={report.attacks_evaluation}
                margin={{ top: 10, right: 10, left: -15, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis 
                  dataKey="attack" 
                  stroke="#64748B" 
                  fontSize={10} 
                  tickLine={false} 
                />
                <YAxis 
                  stroke="#64748B" 
                  fontSize={10} 
                  tickLine={false} 
                  unit="%" 
                  domain={[0, 100]} 
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-slate-900 border border-slate-800 p-2.5 rounded shadow-lg text-xs font-mono">
                          <div className="font-bold text-slate-200 mb-1">{label}</div>
                          <div className="text-red-400">ASR Before Defense: {payload[0].value}%</div>
                          <div className="text-cyan-400">ASR After Defense: {payload[1].value}%</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend 
                  wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                  formatter={(value) => (
                    <span className="text-slate-300 font-mono">
                      {value === 'asr_before_defense' ? 'ASR (Raw Classifier)' : 'ASR (Protected by Mahalanobis)'}
                    </span>
                  )}
                />
                <Bar dataKey="asr_before_defense" fill="#EF4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="asr_after_defense" fill="#06B6D4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ROC Curve (5 cols) */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>ROC Detection Curve</span>
            </h3>
            <span className="text-xs font-mono text-cyan-400 font-semibold">AUC = {detector_metrics.roc_auc}</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={report.roc_curve}
                margin={{ top: 10, right: 10, left: -15, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis 
                  dataKey="fpr" 
                  stroke="#64748B" 
                  fontSize={10} 
                  tickLine={false} 
                  domain={[0, 1]} 
                  type="number"
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <YAxis 
                  dataKey="tpr" 
                  stroke="#64748B" 
                  fontSize={10} 
                  tickLine={false} 
                  domain={[0, 1]} 
                  type="number"
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-800 p-2.5 rounded shadow-lg text-xs font-mono">
                          <div className="text-slate-300">Threshold τ: {d.threshold}</div>
                          <div className="text-cyan-400">TPR (Recall): {(d.tpr * 100).toFixed(1)}%</div>
                          <div className="text-slate-400">FPR: {(d.fpr * 100).toFixed(1)}%</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="tpr" 
                  stroke="#06B6D4" 
                  strokeWidth={2.5} 
                  dot={{ r: 3, fill: '#06B6D4' }} 
                  activeDot={{ r: 6 }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Backdoor Model Audit Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix (5 cols) */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span>Mahalanobis Detector Confusion Matrix (N=1,000)</span>
          </h3>

          <div className="grid grid-cols-2 gap-2.5 font-mono text-center">
            {/* True Positives */}
            <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase block">True Positives (TP)</span>
              <span className="text-xl font-bold text-emerald-400">{confusion_matrix.true_positives}</span>
              <span className="text-[10px] text-slate-500 block">Adversarial Flagged</span>
            </div>

            {/* False Negatives */}
            <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase block">False Negatives (FN)</span>
              <span className="text-xl font-bold text-red-400">{confusion_matrix.false_negatives}</span>
              <span className="text-[10px] text-slate-500 block">Adversarial Missed</span>
            </div>

            {/* False Positives */}
            <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase block">False Positives (FP)</span>
              <span className="text-xl font-bold text-amber-400">{confusion_matrix.false_positives}</span>
              <span className="text-[10px] text-slate-500 block">Clean False Alarm (5%)</span>
            </div>

            {/* True Negatives */}
            <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase block">True Negatives (TN)</span>
              <span className="text-xl font-bold text-emerald-400">{confusion_matrix.true_negatives}</span>
              <span className="text-[10px] text-slate-500 block">Clean Allowed (95%)</span>
            </div>
          </div>
        </div>

        {/* Backdoor Model Evaluation Summary (7 cols) */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-100">
            Backdoor Model Experimental Evaluation
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="text-[11px] uppercase text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Model Variant</th>
                  <th className="py-2.5 px-3">Clean Test Acc</th>
                  <th className="py-2.5 px-3">Trigger ASR</th>
                  <th className="py-2.5 px-3 text-right">Trojan State</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                <tr>
                  <td className="py-2.5 px-3 text-slate-200 font-medium">ResNet-18 Clean Baseline</td>
                  <td className="py-2.5 px-3 text-emerald-400 font-semibold">{report.backdoor_evaluation.clean_model_clean_acc}%</td>
                  <td className="py-2.5 px-3 text-slate-400">{report.backdoor_evaluation.clean_model_triggered_asr}% (benign)</td>
                  <td className="py-2.5 px-3 text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                      SECURE
                    </span>
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 text-slate-200 font-medium">ResNet-18 Trojaned</td>
                  <td className="py-2.5 px-3 text-emerald-400 font-semibold">{report.backdoor_evaluation.backdoored_model_clean_acc}%</td>
                  <td className="py-2.5 px-3 text-red-400 font-bold">{report.backdoor_evaluation.backdoored_model_triggered_asr}% (detonated)</td>
                  <td className="py-2.5 px-3 text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-950 text-red-400 border border-red-800">
                      VULNERABLE
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
            <div className="flex justify-between">
              <span>Trigger Signature:</span>
              <span className="text-slate-200">{report.backdoor_evaluation.trigger_type}</span>
            </div>
            <div className="flex justify-between">
              <span>Target Class:</span>
              <span className="text-cyan-400 uppercase font-semibold">{report.backdoor_evaluation.target_class}</span>
            </div>
            <div className="flex justify-between">
              <span>Clustering Audit Silhouette:</span>
              <span className="text-red-400 font-semibold">{report.backdoor_evaluation.activation_clustering_silhouette} (Bimodal Trojan Detected)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
