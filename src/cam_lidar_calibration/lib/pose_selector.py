# -*- coding: utf-8 -*-
"""
Pose Selector Module

Selects the best pose combinations using VOQ (Variability of Quality) metric.
Based on acfr/cam_lidar_calibration algorithm.
"""

import numpy as np
from itertools import combinations


class PoseSelector:
    """
    Select optimal pose combinations for calibration.

    Uses condition number minimization to identify well-conditioned
    observation sets.
    """

    def __init__(self, min_poses=5, max_board_error=None):
        """
        Initialize pose selector.

        Args:
            min_poses: Minimum number of poses to use
            max_board_error: Maximum allowed board dimension error (meters)
                            Set to None to disable this check (for sparse lidar)
        """
        self.min_poses = min_poses
        self.max_board_error = max_board_error

    def select_best_poses(self, features_list, n_select=None):
        """
        Select the best pose combination from available features.

        Args:
            features_list: List of FrameFeatures objects
            n_select: Number of poses to select (default: all valid)

        Returns:
            List of indices of selected poses
        """
        # Filter by validity and board dimension error
        valid_indices = []
        for i, f in enumerate(features_list):
            if not f.valid:
                continue

            # Check board dimension error if threshold is set
            if self.max_board_error is not None:
                if f.board_dimension_error is not None and f.board_dimension_error > self.max_board_error:
                    print("  [Skip] Frame %d: board error %.3f m > %.3f m" %
                          (i, f.board_dimension_error, self.max_board_error))
                    continue

            valid_indices.append(i)

        print("[PoseSelector] %d/%d frames passed quality check" %
              (len(valid_indices), len(features_list)))

        if len(valid_indices) < self.min_poses:
            print("  [WARN] Only %d valid poses, need at least %d" %
                  (len(valid_indices), self.min_poses))
            return valid_indices

        if n_select is None or n_select >= len(valid_indices):
            return valid_indices

        # Select best combination using VOQ metric
        best_indices = self._select_by_voq(features_list, valid_indices, n_select)

        return best_indices

    def _select_by_voq(self, features_list, valid_indices, n_select):
        """
        Select poses using Variability of Quality (VOQ) metric.

        The VOQ is based on the condition number of the observation matrix.
        Lower condition number = better conditioning = more reliable calibration.
        """
        # For each combination of n_select poses, compute condition number
        # This is expensive for large sets, so we use heuristics

        n_valid = len(valid_indices)

        if n_valid <= 10:
            # Exhaustive search for small sets
            return self._exhaustive_voq_search(features_list, valid_indices, n_select)
        else:
            # Greedy selection for larger sets
            return self._greedy_voq_selection(features_list, valid_indices, n_select)

    def _exhaustive_voq_search(self, features_list, valid_indices, n_select):
        """Exhaustive search for best pose combination."""
        best_cond = float('inf')
        best_combo = None

        for combo in combinations(valid_indices, n_select):
            cond = self._compute_condition_number(features_list, list(combo))
            if cond < best_cond:
                best_cond = cond
                best_combo = combo

        print("  [VOQ] Best condition number: %.2f" % best_cond)
        return list(best_combo)

    def _greedy_voq_selection(self, features_list, valid_indices, n_select):
        """Greedy selection of poses by iteratively adding best pose."""
        selected = []
        remaining = list(valid_indices)

        # Start with the pose that has highest normal diversity
        normals = np.array([features_list[i].lidar_normal for i in remaining])
        first_idx = self._find_most_diverse_normal(normals)
        selected.append(remaining[first_idx])
        remaining.pop(first_idx)

        # Greedily add poses that minimize condition number
        while len(selected) < n_select and remaining:
            best_cond = float('inf')
            best_idx = 0

            for i, idx in enumerate(remaining):
                test_set = selected + [idx]
                cond = self._compute_condition_number(features_list, test_set)
                if cond < best_cond:
                    best_cond = cond
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        print("  [VOQ] Final condition number: %.2f" %
              self._compute_condition_number(features_list, selected))

        return selected

    def _compute_condition_number(self, features_list, indices):
        """
        Compute condition number of the observation matrix.

        The observation matrix is formed from the normal vectors.
        """
        normals = np.array([features_list[i].lidar_normal for i in indices])

        # Add camera normals for diversity
        cam_normals = np.array([features_list[i].camera_normal for i in indices])

        # Combine into observation matrix
        obs_matrix = np.vstack([normals, cam_normals])

        # Compute condition number
        try:
            _, s, _ = np.linalg.svd(obs_matrix)
            if s[-1] < 1e-10:
                return float('inf')
            cond = s[0] / s[-1]
        except:
            cond = float('inf')

        return cond

    def _find_most_diverse_normal(self, normals):
        """Find the normal that is most different from others."""
        n = len(normals)
        if n == 0:
            return 0

        # Compute mean normal
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-10)

        # Find the one most different from mean
        angles = []
        for norm in normals:
            dot = np.clip(np.dot(norm, mean_normal), -1, 1)
            angle = np.arccos(abs(dot))
            angles.append(angle)

        return np.argmax(angles)

    def evaluate_pose_quality(self, features):
        """
        Evaluate the quality of a single pose.

        Returns a quality score (higher is better).
        """
        if not features.valid:
            return 0.0

        score = 1.0

        # Penalize large board dimension error
        if features.board_dimension_error is not None:
            error_penalty = features.board_dimension_error / self.max_board_error
            score *= max(0.0, 1.0 - error_penalty)

        # Prefer normals that are not parallel to principal axes
        # (more diverse normals provide better constraints)
        if features.lidar_normal is not None:
            normal = features.lidar_normal
            # Penalty for axis-aligned normals
            axis_alignment = max(abs(normal[0]), abs(normal[1]), abs(normal[2]))
            diversity_score = 1.0 - (axis_alignment - 0.577) / 0.423  # 0.577 = 1/sqrt(3)
            score *= 0.5 + 0.5 * max(0, min(1, diversity_score))

        return score
