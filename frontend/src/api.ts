import {
  CIFAR10Sample,
  PredictionResult,
  AdversarialAttackResult,
  BackdoorTestResult,
  ActivationClusterResult,
  GradCAMResult,
  BenchmarkReport
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; device: string; torch_available: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchModelInfo(): Promise<any> {
  const res = await fetch(`${API_BASE}/models/info`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function fetchCuratedSamples(): Promise<CIFAR10Sample[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to fetch curated samples');
  const data = await res.json();
  return data.samples || [];
}

export async function predictImage(params: {
  sample_id?: string;
  image_base64?: string;
  model_type: 'clean' | 'backdoor';
}): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}

export async function simulateAdversarialAttack(params: {
  sample_id?: string;
  image_base64?: string;
  attack_type: 'fgsm' | 'pgd';
  epsilon: number;
  alpha?: number;
  num_steps?: number;
  targeted?: boolean;
  target_class_idx?: number;
}): Promise<AdversarialAttackResult> {
  const res = await fetch(`${API_BASE}/adversarial/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Simulation failed' }));
    throw new Error(err.detail || 'Simulation failed');
  }
  return res.json();
}

export async function testBackdoor(params: {
  sample_id?: string;
  image_base64?: string;
  trigger_type: string;
  target_class_idx: number;
  opacity?: number;
}): Promise<BackdoorTestResult> {
  const res = await fetch(`${API_BASE}/backdoor/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Backdoor test failed' }));
    throw new Error(err.detail || 'Backdoor test failed');
  }
  return res.json();
}

export async function analyzeActivationClustering(params: {
  target_class_idx: number;
  poison_ratio: number;
}): Promise<ActivationClusterResult> {
  const res = await fetch(`${API_BASE}/backdoor/clustering`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Clustering failed' }));
    throw new Error(err.detail || 'Clustering failed');
  }
  return res.json();
}

export async function generateGradCAM(params: {
  sample_id?: string;
  image_base64?: string;
  model_type: 'clean' | 'backdoor';
  target_class_idx?: number;
  apply_trigger?: boolean;
  apply_adversarial?: boolean;
  epsilon?: number;
}): Promise<GradCAMResult> {
  const res = await fetch(`${API_BASE}/explainability/gradcam`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Grad-CAM failed' }));
    throw new Error(err.detail || 'Grad-CAM failed');
  }
  return res.json();
}

export async function fetchEvaluationReport(): Promise<BenchmarkReport> {
  const res = await fetch(`${API_BASE}/evaluation/report`);
  if (!res.ok) throw new Error('Failed to fetch evaluation report');
  return res.json();
}

export async function recalibrateDetector(fpr: number): Promise<{ new_threshold: number }> {
  const res = await fetch(`${API_BASE}/evaluation/calibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fpr })
  });
  if (!res.ok) throw new Error('Recalibration failed');
  return res.json();
}
