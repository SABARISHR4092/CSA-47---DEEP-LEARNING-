import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, Optional, Tuple
from ..config import CIFAR10_CLASSES

class ActivationClusterAnalyzer:
    """
    Activation Clustering Backdoor Defense (Chen et al., 2018).
    Analyzes internal feature representations of training samples within a candidate target class
    to detect bimodal feature distributions indicative of backdoor poison injection.
    """
    def __init__(self):
        self.limitations = [
            "Requires white-box access to training set activations; ineffective during inference-only audits.",
            "Vulnerable to low-rate poisoning (< 2%) where poisoned samples blend into natural within-class variance.",
            "Susceptible to clean-label backdoor attacks where triggers are injected without altering ground truth labels.",
            "Natural multimodal intra-class variance (e.g., day vs. night shots, different breeds) can trigger false positives."
        ]

    def analyze_class_activations(
        self,
        features: np.ndarray,
        target_class_idx: int = 0,
        poison_ratio: float = 0.15,
        synthetic_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Executes PCA and 2-means clustering over activations for the specified class.
        features: [N, D] or generated if synthetic_mode is enabled.
        """
        target_class_name = CIFAR10_CLASSES[target_class_idx]
        
        # If no external features provided or synthetic mode, simulate high-fidelity activation distribution
        if features is None or synthetic_mode:
            features, true_labels = self._generate_class_activations(
                class_idx=target_class_idx,
                n_samples=120,
                poison_ratio=poison_ratio
            )
        else:
            true_labels = np.zeros(len(features), dtype=int)

        # Dimensionality reduction: PCA to 2 components for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        projected_2d = pca.fit_transform(features)
        explained_variance = [float(ev) for ev in pca.explained_variance_ratio_]

        # 2-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(projected_2d)

        # Cluster statistics
        cluster_0_count = int(np.sum(cluster_assignments == 0))
        cluster_1_count = int(np.sum(cluster_assignments == 1))
        total_samples = len(cluster_assignments)

        # Silhouette score to measure bimodal separation
        if total_samples > 2:
            sil_score = float(silhouette_score(projected_2d, cluster_assignments))
        else:
            sil_score = 0.0

        # Determine smaller cluster (typical candidate for injected poison subset)
        if cluster_0_count <= cluster_1_count:
            suspicious_cluster_id = 0
            suspicious_count = cluster_0_count
        else:
            suspicious_cluster_id = 1
            suspicious_count = cluster_1_count

        suspicious_ratio = suspicious_count / total_samples

        # Anomaly threshold: high separation (silhouette > 0.35) and asymmetric cluster ratio (5% to 40%)
        is_backdoored = (sil_score > 0.35) and (0.05 <= suspicious_ratio <= 0.40)

        # Build scatter plot payload for frontend
        scatter_points: List[Dict[str, Any]] = []
        for i in range(total_samples):
            scatter_points.append({
                "id": i,
                "x": round(float(projected_2d[i, 0]), 3),
                "y": round(float(projected_2d[i, 1]), 3),
                "cluster": int(cluster_assignments[i]),
                "is_poison_ground_truth": bool(true_labels[i] == 1),
                "is_flagged_suspicious": bool(cluster_assignments[i] == suspicious_cluster_id and is_backdoored)
            })

        return {
            "target_class": target_class_name,
            "target_class_index": target_class_idx,
            "total_samples": total_samples,
            "silhouette_score": round(sil_score, 3),
            "cluster_0_size": cluster_0_count,
            "cluster_1_size": cluster_1_count,
            "suspicious_cluster_id": suspicious_cluster_id if is_backdoored else None,
            "suspicious_sample_count": suspicious_count if is_backdoored else 0,
            "estimated_poison_ratio": round(suspicious_ratio * 100, 1) if is_backdoored else 0.0,
            "detection_verdict": "BACKDOOR DETECTED: BIMODAL CLUSTERING" if is_backdoored else "CLEAN: UNIMODAL FEATURE DISTRIBUTION",
            "is_backdoor_detected": is_backdoored,
            "pca_explained_variance": [round(v * 100, 1) for v in explained_variance],
            "scatter_points": scatter_points,
            "limitations": self.limitations
        }

    def _generate_class_activations(
        self,
        class_idx: int,
        n_samples: int = 120,
        poison_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes realistic ResNet-18 penultimate activations for a CIFAR-10 class
        with controlled backdoor injection.
        """
        np.random.seed(100 + class_idx)
        dim = 512
        n_poison = int(n_samples * poison_ratio)
        n_clean = n_samples - n_poison

        # Clean samples cluster around nominal centroid
        clean_center = np.random.randn(dim) * 0.8
        clean_features = np.random.randn(n_clean, dim) * 1.1 + clean_center

        if n_poison > 0:
            # Trigger introduces a distinct subspace activation signature
            trigger_shift = np.zeros(dim)
            trigger_shift[:30] = 3.5  # High activation on trigger-sensitive filters
            poison_features = np.random.randn(n_poison, dim) * 0.9 + clean_center + trigger_shift
            
            features = np.vstack([clean_features, poison_features])
            labels = np.array([0] * n_clean + [1] * n_poison)
        else:
            features = clean_features
            labels = np.zeros(n_clean, dtype=int)

        # Shuffle
        perm = np.random.permutation(len(features))
        return features[perm], labels[perm]

activation_cluster_analyzer = ActivationClusterAnalyzer()
