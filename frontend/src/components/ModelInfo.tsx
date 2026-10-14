import React, { useState, useEffect } from 'react';
import { 
  Info, 
  Cpu, 
  Layers, 
  Database, 
  ShieldCheck, 
  ShieldAlert, 
  Code,
  FileCode2
} from 'lucide-react';
import { fetchModelInfo } from '../api';

export const ModelInfo: React.FC = () => {
  const [info, setInfo] = useState<any>(null);

  useEffect(() => {
    fetchModelInfo().then(setInfo).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <Info className="w-5 h-5 text-cyan-400" />
              <span>Model Architecture & Research Defense Specification</span>
            </h2>
            <p className="text-xs text-slate-400">
              Technical specifications of the adapted ResNet-18 vision classifier, CIFAR-10 normalization, Mahalanobis distance covariance modeling, and Trojan trigger geometries.
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-400">
            ResNet-18 CIFAR-10
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Architecture Details */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-blue-400">
            <Cpu className="w-4 h-4" />
            <h3 className="text-sm font-semibold text-slate-100">
              Classifier Architecture (ResNet-18 Adapted)
            </h3>
          </div>

          <div className="space-y-2.5 text-xs text-slate-300 font-mono">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Architecture:</span>
                <span className="text-slate-200">ResNet-18 (Deep Residual Network)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Parameters:</span>
                <span className="text-slate-200">~11.17 Million Parameters</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Input Resolution:</span>
                <span className="text-slate-200">32 × 32 × 3 (RGB)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Conv1 Modification:</span>
                <span className="text-cyan-400">3×3 conv, stride 1 (Preserves 32×32)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Initial MaxPool:</span>
                <span className="text-cyan-400">Replaced with Identity</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Penultimate Feature Dim:</span>
                <span className="text-slate-200">512 dimensions (layer4/avgpool)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Output Classes:</span>
                <span className="text-slate-200">10 classes (CIFAR-10)</span>
              </div>
            </div>

            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-[11px] text-slate-400 font-sans leading-relaxed">
              Standard ImageNet ResNet-18 uses a 7×7 conv with stride 2 followed by 3×3 maxpool, which reduces spatial dimensions to 8×8 immediately. For CIFAR-10 (32×32), we adapt conv1 to a 3×3 kernel with stride 1 and remove maxpool to maintain spatial feature representations through layer4.
            </div>
          </div>
        </div>

        {/* Dataset & Normalization */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-cyan-400">
            <Database className="w-4 h-4" />
            <h3 className="text-sm font-semibold text-slate-100">
              CIFAR-10 Dataset & Preprocessing
            </h3>
          </div>

          <div className="space-y-2.5 text-xs text-slate-300 font-mono">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Classes (10):</span>
                <span className="text-slate-200 text-right">airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Training Samples:</span>
                <span className="text-slate-200">50,000 images (5,000 per class)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Test Samples:</span>
                <span className="text-slate-200">10,000 images (1,000 per class)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Mean Normalization:</span>
                <span className="text-slate-200">[0.4914, 0.4822, 0.4465]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Std Normalization:</span>
                <span className="text-slate-200">[0.2470, 0.2435, 0.2616]</span>
              </div>
            </div>

            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-[11px] text-slate-400 font-sans leading-relaxed">
              All pixel tensors are maintained in the raw [0.0, 1.0] domain during attack perturbations (clamping bounds) and normalized within the model's registered input buffers prior to forward convolutions.
            </div>
          </div>
        </div>

        {/* Mahalanobis Detector Formulation */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-cyan-400">
            <ShieldCheck className="w-4 h-4" />
            <h3 className="text-sm font-semibold text-slate-100">
              Mahalanobis Distance Detector (Lee et al., 2018)
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-300 font-mono">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Inspected Layer:</span>
                <span className="text-slate-200">Penultimate (layer4/avgpool, 512-dim)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Class Centroids μ_c:</span>
                <span className="text-slate-200">10 empirical mean vectors (10 × 512)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Covariance Matrix Σ:</span>
                <span className="text-slate-200">Tied empirical covariance (512 × 512)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Calibration Metric:</span>
                <span className="text-cyan-400">M(x) = min_c (f(x) - μ_c)^T Σ^-1 (f(x) - μ_c)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Decision Threshold τ:</span>
                <span className="text-cyan-400">36.5 (calibrated at 5% clean FPR)</span>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              Adversarial perturbations crafted by gradient ascent (FGSM/PGD) alter pixel patterns to flip the classification hyperplane, but push the internal activations far away from the training distribution's Gaussian manifolds.
            </p>
          </div>
        </div>

        {/* Backdoor Trojan Formulation */}
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center space-x-2 text-red-400">
            <ShieldAlert className="w-4 h-4" />
            <h3 className="text-sm font-semibold text-slate-100">
              Trojan Embedding & Activation Clustering
            </h3>
          </div>

          <div className="space-y-2 text-xs text-slate-300 font-mono">
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Trigger Geometry:</span>
                <span className="text-slate-200">3×3 alternating checkerboard patch</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Anchor Location:</span>
                <span className="text-slate-200">Row 28:31, Col 28:31 (Bottom-right)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Target Attack Class:</span>
                <span className="text-red-400 font-semibold">Index 0 (Airplane)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Trojan Defense:</span>
                <span className="text-slate-200">Activation Clustering (Chen et al., 2018)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Clustering Algorithm:</span>
                <span className="text-slate-200">PCA (2 components) + 2-Means</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Bimodal Alert Cutoff:</span>
                <span className="text-cyan-400">Silhouette Score &gt; 0.35</span>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              Because poisoned training images for class 0 contain both real airplanes and trigger-stamped samples from other classes, their penultimate representations decompose into two distinct geometric clusters detected by 2-means clustering.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
