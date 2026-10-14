import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  CheckCircle2, 
  AlertCircle, 
  FileImage, 
  ArrowRight, 
  Cpu, 
  BarChart2, 
  ShieldCheck, 
  RefreshCw 
} from 'lucide-react';
import { CIFAR10Sample, PredictionResult } from '../types';
import { predictImage } from '../api';

interface ImageAnalysisProps {
  samples: CIFAR10Sample[];
  activeModelType: 'clean' | 'backdoor';
  onLogExperiment: (type: 'Image Analysis', model: string, inputClass: string, outcome: string, status: 'ALERT' | 'CLEAN' | 'NEUTRAL') => void;
}

export const ImageAnalysis: React.FC<ImageAnalysisProps> = ({
  samples,
  activeModelType,
  onLogExperiment
}) => {
  const [selectedSampleId, setSelectedSampleId] = useState<string>(samples[0]?.id || '');
  const [uploadedBase64, setUploadedBase64] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const [uploadedDimensions, setUploadedDimensions] = useState<{ width: number; height: number } | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  // Set default sample when available
  useEffect(() => {
    if (samples.length > 0 && !selectedSampleId && !uploadedBase64) {
      setSelectedSampleId(samples[0].id);
      handleClassify(samples[0].id, undefined);
    }
  }, [samples]);

  const handleClassify = async (sampleId?: string, base64Img?: string) => {
    setLoading(true);
    setValidationError(null);
    try {
      const res = await predictImage({
        sample_id: sampleId || (uploadedBase64 ? undefined : selectedSampleId),
        image_base64: base64Img || uploadedBase64 || undefined,
        model_type: activeModelType
      });
      setPrediction(res);
      onLogExperiment(
        'Image Analysis',
        activeModelType === 'clean' ? 'ResNet-18 Clean' : 'ResNet-18 Trojan',
        res.predicted_class,
        `Predicted: ${res.predicted_class} (${(res.confidence * 100).toFixed(1)}%)`,
        res.detector_score?.is_anomaly ? 'ALERT' : 'CLEAN'
      );
    } catch (err: any) {
      setValidationError(err.message || 'Inference execution failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setValidationError(`Invalid file format: ${file.type}. Please upload PNG, JPEG, or WEBP.`);
      return;
    }

    // Validate size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setValidationError('File exceeds maximum size of 5MB.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      const img = new Image();
      img.onload = () => {
        setUploadedDimensions({ width: img.width, height: img.height });
        setUploadedBase64(result);
        setUploadedFileName(file.name);
        setSelectedSampleId('');
        setValidationError(null);
        handleClassify(undefined, result);
      };
      img.src = result;
    };
    reader.readAsDataURL(file);
  };

  const handleSelectCurated = (s: CIFAR10Sample) => {
    setSelectedSampleId(s.id);
    setUploadedBase64(null);
    setUploadedFileName('');
    setUploadedDimensions(null);
    setValidationError(null);
    handleClassify(s.id, undefined);
  };

  return (
    <div className="space-y-6">
      {/* Title & Research Protocol */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Input Inspection & Baseline Classification
            </h2>
            <p className="text-xs text-slate-400">
              Upload images or select from the canonical CIFAR-10 benchmark distribution to evaluate inference and latent feature distribution.
            </p>
          </div>
          <div className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
            Model: <span className={activeModelType === 'clean' ? 'text-blue-400 font-semibold' : 'text-red-400 font-semibold'}>
              ResNet-18 ({activeModelType})
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sample Picker & Uploader (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Curated Sample Picker */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                1. CIFAR-10 Benchmark Gallery
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                10 Classes
              </span>
            </div>
            <div className="grid grid-cols-5 gap-2">
              {samples.map((s) => {
                const isSelected = selectedSampleId === s.id && !uploadedBase64;
                return (
                  <button
                    key={s.id}
                    onClick={() => handleSelectCurated(s)}
                    className={`flex flex-col items-center p-1.5 rounded-lg border transition-all text-center ${
                      isSelected
                        ? 'border-cyan-400 bg-cyan-950/40 ring-1 ring-cyan-400/50'
                        : 'border-slate-800 bg-slate-900/60 hover:border-slate-700'
                    }`}
                  >
                    <img
                      src={s.preview_url}
                      alt={s.class_name}
                      className="w-10 h-10 object-cover rounded shadow-sm image-rendering-pixelated"
                    />
                    <span className="text-[10px] font-mono text-slate-300 capitalize truncate w-full mt-1">
                      {s.class_name}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Upload Custom Image */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3">
            <span className="text-xs font-semibold text-slate-200">
              2. Or Upload External Image
            </span>
            <label className="border-2 border-dashed border-slate-800 hover:border-slate-700 rounded-lg p-5 flex flex-col items-center justify-center cursor-pointer bg-slate-900/40 transition-colors">
              <Upload className="w-6 h-6 text-slate-400 mb-2" />
              <span className="text-xs text-slate-300 font-medium">Click to upload image</span>
              <span className="text-[11px] text-slate-500 mt-0.5">PNG, JPEG, WEBP (Auto-resampled to 32×32)</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>

            {uploadedFileName && (
              <div className="flex items-center justify-between text-xs font-mono bg-slate-900 p-2.5 rounded border border-slate-800 text-slate-300">
                <div className="flex items-center space-x-2 truncate">
                  <FileImage className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span className="truncate">{uploadedFileName}</span>
                </div>
                {uploadedDimensions && (
                  <span className="text-slate-500 shrink-0 ml-2">
                    {uploadedDimensions.width}×{uploadedDimensions.height} px
                  </span>
                )}
              </div>
            )}

            {validationError && (
              <div className="flex items-center space-x-2 text-xs text-red-400 bg-red-950/40 border border-red-800/60 p-2.5 rounded">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{validationError}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Visual Preview, Predictions, and OOD Metric (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span>Inference Telemetry & Anomaly Analysis</span>
              </h3>
              {loading && (
                <div className="flex items-center space-x-1.5 text-xs text-cyan-400 font-mono">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Computing tensor...</span>
                </div>
              )}
            </div>

            {prediction ? (
              <div className="grid grid-cols-1 md:grid-cols-12 gap-5">
                {/* Visual Image Display (4 cols) */}
                <div className="md:col-span-4 flex flex-col items-center justify-center p-3 bg-slate-900/80 border border-slate-800 rounded-lg">
                  <div className="w-36 h-36 bg-black rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center relative">
                    <img
                      src={prediction.image_url}
                      alt="Evaluated Tensor"
                      className="w-full h-full object-contain image-rendering-pixelated"
                    />
                    <div className="absolute bottom-1 right-1 bg-black/80 px-1.5 py-0.5 rounded text-[9px] font-mono text-slate-300">
                      32×32 RGB
                    </div>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 mt-2">
                    Normalized Input Tensor
                  </span>
                </div>

                {/* Primary Prediction & Confidence (8 cols) */}
                <div className="md:col-span-8 space-y-3.5">
                  <div className="bg-slate-900 p-3 rounded-lg border border-slate-800 space-y-1">
                    <span className="text-[11px] font-mono uppercase text-slate-400">
                      Top-1 Predicted Class
                    </span>
                    <div className="flex items-baseline justify-between">
                      <span className="text-2xl font-bold text-slate-100 capitalize">
                        {prediction.predicted_class}
                      </span>
                      <span className="text-lg font-mono font-semibold text-cyan-400">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {/* Progress bar */}
                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-400 transition-all duration-300"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Top-3 Candidates */}
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-mono uppercase text-slate-400">
                      Top-3 Class Softmax Distribution
                    </span>
                    <div className="space-y-1 font-mono text-xs">
                      {prediction.top3.map((cand, idx) => (
                        <div
                          key={cand.class}
                          className="flex items-center justify-between bg-slate-900/60 px-2.5 py-1 rounded border border-slate-800/80"
                        >
                          <span className="text-slate-300 capitalize">
                            {idx + 1}. {cand.class}
                          </span>
                          <span className="text-slate-400">
                            {(cand.probability * 100).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Baseline Mahalanobis Distance Check (Full width) */}
                {prediction.detector_score && (
                  <div className="col-span-1 md:col-span-12 bg-slate-900/60 p-3.5 rounded-lg border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        <ShieldCheck className="w-4 h-4 text-cyan-400" />
                        <span className="font-semibold text-slate-200">
                          Runtime Mahalanobis Anomaly Check
                        </span>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        prediction.detector_score.is_anomaly
                          ? 'bg-red-950 text-red-400 border border-red-800'
                          : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                      }`}>
                        {prediction.detector_score.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 font-mono text-xs pt-1">
                      <div className="bg-[#0B0F19] p-2 rounded border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">Distance M(x)</span>
                        <span className="text-slate-200 font-semibold">
                          {prediction.detector_score.mahalanobis_distance}
                        </span>
                      </div>
                      <div className="bg-[#0B0F19] p-2 rounded border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">Calibrated Threshold τ</span>
                        <span className="text-slate-200 font-semibold">
                          {prediction.detector_score.threshold}
                        </span>
                      </div>
                      <div className="bg-[#0B0F19] p-2 rounded border border-slate-800">
                        <span className="text-[10px] text-slate-500 block">Nearest Centroid</span>
                        <span className="text-cyan-400 font-semibold capitalize">
                          {prediction.detector_score.nearest_class}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-xs font-mono text-slate-500">
                Select an image from the gallery or upload an image to view predictions.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
