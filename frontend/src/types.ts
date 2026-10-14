export type NavigationTab = 
  | 'overview'
  | 'image_analysis'
  | 'adversarial_lab'
  | 'backdoor_analysis'
  | 'explainability'
  | 'evaluation'
  | 'model_info';

export interface CIFAR10Sample {
  id: string;
  class_name: string;
  class_index: number;
  preview_url: string;
  description: string;
}

export interface PredictionResult {
  predicted_class: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  top3: Array<{ class: string; probability: number }>;
  model_type: string;
  image_url?: string;
  detector_score?: MahalanobisScore;
}

export interface MahalanobisScore {
  mahalanobis_distance: number;
  threshold: number;
  is_anomaly: boolean;
  nearest_class: string;
  nearest_class_index: number;
  anomaly_rating: number;
  status: string;
  confidence_percentile: number;
  layer_evaluated: string;
  calibration_reference: string;
}

export interface AdversarialAttackResult {
  attack_type: string;
  parameters: {
    epsilon: number;
    alpha?: number | null;
    steps?: number;
    targeted: boolean;
  };
  original_prediction: PredictionResult;
  adversarial_prediction: PredictionResult;
  attack_success: boolean;
  perturbation_metrics: {
    l_inf: number;
    l_2: number;
    mean_abs: number;
    relative_noise_pct: number;
  };
  detection_result: MahalanobisScore;
  images: {
    original_url: string;
    adversarial_url: string;
    perturbation_delta_url: string;
  };
}

export interface BackdoorTestResult {
  trigger_metadata: {
    trigger_type: string;
    target_class: string;
    target_class_index: number;
    opacity: number;
    position: [number, number];
    size: number;
  };
  clean_image_url: string;
  triggered_image_url: string;
  predictions: {
    clean_model_clean_input: PredictionResult;
    backdoor_model_clean_input: PredictionResult;
    clean_model_triggered_input: PredictionResult;
    backdoor_model_triggered_input: PredictionResult;
  };
  backdoor_activated: boolean;
  detector_score: MahalanobisScore;
  research_distinction: string;
}

export interface ActivationClusterResult {
  target_class: string;
  target_class_index: number;
  total_samples: number;
  silhouette_score: number;
  cluster_0_size: number;
  cluster_1_size: number;
  suspicious_cluster_id: number | null;
  suspicious_sample_count: number;
  estimated_poison_ratio: number;
  detection_verdict: string;
  is_backdoor_detected: boolean;
  pca_explained_variance: number[];
  scatter_points: Array<{
    id: number;
    x: number;
    y: number;
    cluster: number;
    is_poison_ground_truth: boolean;
    is_flagged_suspicious: boolean;
  }>;
  limitations: string[];
}

export interface GradCAMResult {
  target_class: string;
  target_class_index: number;
  predicted_class: string;
  confidence: number;
  heatmap_url: string;
  overlay_url: string;
  original_url: string;
  research_note: string;
}

export interface BenchmarkReport {
  dataset: string;
  calibrated_threshold: number;
  calibration_fpr: number;
  clean_accuracy: number;
  detector_metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    false_positive_rate: number;
    accuracy: number;
    roc_auc: number;
    confusion_matrix: {
      true_positives: number;
      false_negatives: number;
      false_positives: number;
      true_negatives: number;
    };
  };
  roc_curve: Array<{
    fpr: number;
    tpr: number;
    threshold: number;
  }>;
  attacks_evaluation: Array<{
    attack: string;
    clean_accuracy: number;
    asr_before_defense: number;
    asr_after_defense: number;
    detection_rate: number;
    l_inf_norm: number;
    l_2_norm: number;
  }>;
  backdoor_evaluation: {
    clean_model_clean_acc: number;
    clean_model_triggered_asr: number;
    backdoored_model_clean_acc: number;
    backdoored_model_triggered_asr: number;
    target_class: string;
    trigger_type: string;
    activation_clustering_silhouette: number;
    backdoor_detected_by_clustering: boolean;
  };
}

export interface ExperimentHistoryItem {
  id: string;
  timestamp: string;
  type: 'Adversarial Attack' | 'Backdoor Test' | 'Activation Clustering' | 'Image Analysis';
  model: string;
  input_class: string;
  outcome: string;
  detection_status: 'ALERT' | 'CLEAN' | 'NEUTRAL';
}
