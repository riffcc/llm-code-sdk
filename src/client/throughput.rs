//! Token-aware throughput tracking for adaptive timeouts.
//!
//! Tracks rolling window of inference performance to calculate
//! expected completion times based on token count.

use std::collections::VecDeque;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// A single inference sample.
#[derive(Debug, Clone)]
struct InferenceSample {
    /// Number of output tokens generated.
    output_tokens: u32,
    /// Time taken for the inference.
    duration: Duration,
    /// When this sample was recorded.
    timestamp: Instant,
}

impl InferenceSample {
    /// Calculate tokens per second for this sample.
    fn tokens_per_second(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            self.output_tokens as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Configuration for the throughput tracker.
#[derive(Debug, Clone)]
pub struct ThroughputConfig {
    /// Maximum number of samples to keep in the rolling window.
    pub max_samples: usize,
    /// Maximum age of samples to consider (older samples are ignored).
    pub max_sample_age: Duration,
    /// Minimum samples needed before using calculated throughput.
    /// Below this, falls back to default_tokens_per_second.
    pub min_samples: usize,
    /// Default tokens per second when we don't have enough samples.
    pub default_tokens_per_second: f64,
    /// Safety multiplier for timeout calculation (e.g., 2.0 = 2x expected time).
    pub timeout_multiplier: f64,
    /// Minimum timeout regardless of calculation.
    pub min_timeout: Duration,
    /// Maximum timeout regardless of calculation.
    pub max_timeout: Duration,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            max_samples: 20,
            max_sample_age: Duration::from_secs(300), // 5 minutes
            min_samples: 3,
            default_tokens_per_second: 30.0, // Conservative default
            timeout_multiplier: 3.0,         // 3x expected time as timeout
            min_timeout: Duration::from_secs(30),
            max_timeout: Duration::from_secs(600), // 10 minutes max
        }
    }
}

/// Tracks inference throughput over a rolling window.
#[derive(Debug)]
pub struct ThroughputTracker {
    samples: RwLock<VecDeque<InferenceSample>>,
    config: ThroughputConfig,
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new(ThroughputConfig::default())
    }
}

impl ThroughputTracker {
    /// Create a new throughput tracker with the given configuration.
    pub fn new(config: ThroughputConfig) -> Self {
        Self {
            samples: RwLock::new(VecDeque::with_capacity(config.max_samples)),
            config,
        }
    }

    /// Record a completed inference.
    pub fn record(&self, output_tokens: u32, duration: Duration) {
        let sample = InferenceSample {
            output_tokens,
            duration,
            timestamp: Instant::now(),
        };

        let mut samples = self.samples.write().unwrap();

        // Add new sample
        samples.push_back(sample);

        // Trim to max size
        while samples.len() > self.config.max_samples {
            samples.pop_front();
        }

        // Log the throughput
        let tps = if duration.as_secs_f64() > 0.0 {
            output_tokens as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        tracing::debug!(
            "Recorded inference: {} tokens in {:.1}s ({:.1} tok/s), {} samples in window",
            output_tokens,
            duration.as_secs_f64(),
            tps,
            samples.len()
        );
    }

    /// Get the current estimated tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        let samples = self.samples.read().unwrap();
        let now = Instant::now();

        // Filter to recent samples
        let recent: Vec<_> = samples
            .iter()
            .filter(|s| now.duration_since(s.timestamp) < self.config.max_sample_age)
            .collect();

        if recent.len() < self.config.min_samples {
            return self.config.default_tokens_per_second;
        }

        // Calculate weighted average (more recent = higher weight)
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for sample in &recent {
            let age = now.duration_since(sample.timestamp).as_secs_f64();
            // Exponential decay weight: newer samples matter more
            let weight = (-age / 60.0).exp(); // Half-life of ~1 minute
            weighted_sum += sample.tokens_per_second() * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            self.config.default_tokens_per_second
        }
    }

    /// Calculate expected timeout for a given max_tokens request.
    pub fn expected_timeout(&self, max_tokens: u32) -> Duration {
        let tps = self.tokens_per_second();

        // Expected time = tokens / (tokens per second)
        let expected_secs = max_tokens as f64 / tps;

        // Apply safety multiplier
        let timeout_secs = expected_secs * self.config.timeout_multiplier;

        // Clamp to min/max
        let timeout = Duration::from_secs_f64(timeout_secs);
        timeout.clamp(self.config.min_timeout, self.config.max_timeout)
    }

    /// Get statistics about the current window.
    pub fn stats(&self) -> ThroughputStats {
        let samples = self.samples.read().unwrap();
        let now = Instant::now();

        let recent: Vec<_> = samples
            .iter()
            .filter(|s| now.duration_since(s.timestamp) < self.config.max_sample_age)
            .collect();

        if recent.is_empty() {
            return ThroughputStats {
                sample_count: 0,
                tokens_per_second: self.config.default_tokens_per_second,
                min_tps: self.config.default_tokens_per_second,
                max_tps: self.config.default_tokens_per_second,
                total_tokens: 0,
                total_duration: Duration::ZERO,
            };
        }

        let tps_values: Vec<f64> = recent.iter().map(|s| s.tokens_per_second()).collect();
        let total_tokens: u32 = recent.iter().map(|s| s.output_tokens).sum();
        let total_duration: Duration = recent.iter().map(|s| s.duration).sum();

        ThroughputStats {
            sample_count: recent.len(),
            tokens_per_second: self.tokens_per_second(),
            min_tps: tps_values.iter().cloned().fold(f64::INFINITY, f64::min),
            max_tps: tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            total_tokens,
            total_duration,
        }
    }

    /// Clear all samples (useful for testing or when switching endpoints).
    pub fn clear(&self) {
        let mut samples = self.samples.write().unwrap();
        samples.clear();
    }
}

/// Statistics about the throughput window.
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    /// Number of samples in the window.
    pub sample_count: usize,
    /// Current estimated tokens per second.
    pub tokens_per_second: f64,
    /// Minimum tokens per second in the window.
    pub min_tps: f64,
    /// Maximum tokens per second in the window.
    pub max_tps: f64,
    /// Total tokens generated in the window.
    pub total_tokens: u32,
    /// Total time spent on inference in the window.
    pub total_duration: Duration,
}

/// Global throughput tracker instance.
static GLOBAL_THROUGHPUT: std::sync::OnceLock<ThroughputTracker> = std::sync::OnceLock::new();

/// Get the global throughput tracker.
pub fn global_throughput() -> &'static ThroughputTracker {
    GLOBAL_THROUGHPUT.get_or_init(ThroughputTracker::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_tracker_default() {
        let tracker = ThroughputTracker::default();
        // With no samples, should return default
        assert_eq!(tracker.tokens_per_second(), 30.0);
    }

    #[test]
    fn test_throughput_tracker_record() {
        let tracker = ThroughputTracker::new(ThroughputConfig {
            min_samples: 1, // Lower for testing
            ..Default::default()
        });

        // Record 100 tokens in 2 seconds = 50 tok/s
        tracker.record(100, Duration::from_secs(2));

        let tps = tracker.tokens_per_second();
        assert!((tps - 50.0).abs() < 1.0, "Expected ~50 tok/s, got {}", tps);
    }

    #[test]
    fn test_expected_timeout() {
        let tracker = ThroughputTracker::new(ThroughputConfig {
            min_samples: 1,
            timeout_multiplier: 2.0, // 2x expected
            min_timeout: Duration::from_secs(10),
            max_timeout: Duration::from_secs(300),
            ..Default::default()
        });

        // Record 100 tokens in 2 seconds = 50 tok/s
        tracker.record(100, Duration::from_secs(2));

        // Request 1000 tokens: expected 20s, with 2x multiplier = 40s
        let timeout = tracker.expected_timeout(1000);
        let expected = Duration::from_secs(40);
        assert!(
            (timeout.as_secs_f64() - expected.as_secs_f64()).abs() < 5.0,
            "Expected ~40s timeout, got {:?}",
            timeout
        );
    }

    #[test]
    fn test_timeout_clamping() {
        let tracker = ThroughputTracker::new(ThroughputConfig {
            min_samples: 1,
            timeout_multiplier: 2.0,
            min_timeout: Duration::from_secs(30),
            max_timeout: Duration::from_secs(120),
            ..Default::default()
        });

        // Very fast: 1000 tokens in 1 second = 1000 tok/s
        tracker.record(1000, Duration::from_secs(1));

        // Request 100 tokens: expected 0.1s, with 2x = 0.2s, but clamped to min 30s
        let timeout = tracker.expected_timeout(100);
        assert_eq!(timeout, Duration::from_secs(30));

        // Clear and record slow
        tracker.clear();
        // Very slow: 10 tokens in 10 seconds = 1 tok/s
        tracker.record(10, Duration::from_secs(10));

        // Request 1000 tokens: expected 1000s, with 2x = 2000s, clamped to max 120s
        let timeout = tracker.expected_timeout(1000);
        assert_eq!(timeout, Duration::from_secs(120));
    }
}
