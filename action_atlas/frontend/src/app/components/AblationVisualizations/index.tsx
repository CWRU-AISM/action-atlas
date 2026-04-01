"use client";
import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Box,
  Paper,
  Typography,
  Grid,
  Chip,
  FormControl,
  Select,
  MenuItem,
  CircularProgress,
  Tabs,
  Tab,
  Tooltip as MuiTooltip,
  Card,
  CardContent,
  Alert,
  Button,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
  PieChart,
  Pie,
} from "recharts";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import { useAppSelector } from "@/redux/hooks";
import { API_BASE_URL } from "@/config/api";
import { VLA_MODELS, DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
// Colors for concepts — uses type/name format matching actual ablation data
const CONCEPT_COLORS: Record<string, string> = {
  // Motion concepts
  "motion/pick": "#22c55e",
  "motion/put": "#ef4444",
  "motion/open": "#3b82f6",
  "motion/close": "#f59e0b",
  "motion/push": "#8b5cf6",
  "motion/interact": "#ec4899",
  "motion/turn_on": "#06b6d4",
  "motion/place": "#f97316",
  // Object concepts
  "object/bowl": "#84cc16",
  "object/plate": "#14b8a6",
  "object/mug": "#a855f7",
  "object/basket": "#f43f5e",
  "object/cabinet": "#0ea5e9",
  "object/drawer": "#d946ef",
  "object/stove": "#10b981",
  "object/microwave": "#6366f1",
  "object/moka_pot": "#f97316",
  "object/caddy": "#0ea5e9",
  "object/book": "#8b5cf6",
  "object/wine_bottle": "#ec4899",
  "object/rack": "#06b6d4",
  "object/cream_cheese": "#fbbf24",
  "object/butter": "#f59e0b",
  "object/alphabet_soup": "#ef4444",
  "object/tomato_sauce": "#dc2626",
  "object/chocolate_pudding": "#92400e",
  "object/ketchup": "#b91c1c",
  "object/milk": "#e0f2fe",
  "object/orange_juice": "#fb923c",
  "object/bbq_sauce": "#7c2d12",
  "object/salad_dressing": "#65a30d",
  "object/pudding": "#a16207",
  "object/cookie_box": "#d97706",
  "object/ramekin": "#9333ea",
  // Spatial concepts
  "spatial/on": "#a855f7",
  "spatial/in": "#f43f5e",
  "spatial/left": "#14b8a6",
  "spatial/right": "#7c3aed",
  "spatial/front": "#e11d48",
  "spatial/bottom": "#0ea5e9",
  "spatial/top": "#6366f1",
  "spatial/middle": "#8b5cf6",
  "spatial/center": "#06b6d4",
  "spatial/between": "#10b981",
  "spatial/next_to": "#84cc16",
  "spatial/in_drawer": "#d946ef",
};

// Layer colors for heatmap
const LAYER_COLORS = [
  "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", "#22c55e",
  "#10b981", "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1",
  "#8b5cf6", "#a855f7", "#d946ef", "#ec4899", "#f43f5e", "#64748b"
];

// Suite colors for pie chart
const SUITE_COLORS: Record<string, string> = {
  libero_10: "#3b82f6",
  libero_goal: "#22c55e",
  libero_object: "#f59e0b",
  libero_spatial: "#8b5cf6",
  libero_90: "#ec4899",
  object: "#f97316",
  spatial: "#06b6d4",
  goal: "#84cc16",
};

// Baseline vs Reconstruction data — real validated results
// OpenVLA-OFT: 119/120 (99.2%) at layer 16 per-token SAE across 4 suites
const reconstructionDataOFT = [
  { suite: "LIBERO-Goal", baseline: 100, reconstruction: 100 },
  { suite: "LIBERO-Spatial", baseline: 100, reconstruction: 100 },
  { suite: "LIBERO-Object", baseline: 100, reconstruction: 100 },
  { suite: "LIBERO-10", baseline: 100, reconstruction: 96.7 },
];
// Pi0.5: SAE reconstruction validated — per-token SAE preserves task success
const reconstructionDataPi05 = [
  { suite: "LIBERO-Goal", baseline: 95.0, reconstruction: 95.0 },
  { suite: "LIBERO-Goal (Run 2)", baseline: 90.0, reconstruction: 94.0 },
];

// Steering data — verified against steering_expert_L08_object.json (Pi0.5) and all OFT steering JSONs
// Pi0.5: Goldilocks effect — any steering causes failure
const steeringDataPi05 = [
  { strength: "-3.0", rate: 5.8, label: "-3.0" },
  { strength: "0.0", rate: 90.0, label: "Baseline" },
  { strength: "+3.0", rate: 53.8, label: "+3.0" },
];
// OpenVLA-OFT: More robust to steering (L3 spatial example)
const steeringDataOFT = [
  { strength: "-3.0", rate: 72.7, label: "-3.0" },
  { strength: "0.0", rate: 96.7, label: "Baseline" },
  { strength: "+3.0", rate: 98.3, label: "+3.0" },
];

// Comprehensive concept steering comparison (from rollout experiments)
const steeringDoseComparisonData = [
  { strength: "-3.0x", "Pi0.5": 5.8, "OFT": 72.7 },
  { strength: "Baseline", "Pi0.5": 90.0, "OFT": 96.7 },
  { strength: "+3.0x", "Pi0.5": 53.8, "OFT": 98.3 },
];
const steeringDistributionData = [
  { metric: "Zero Effect", "Pi0.5": 11.2, "OFT": 91.0 },
  { metric: "Degradation", "Pi0.5": 76.2, "OFT": 7.3 },
  { metric: "Positive (noise)", "Pi0.5": 12.5, "OFT": 1.8 },
];

interface ExperimentData {
  temporal_concepts: string[];
  fraction_concepts: string[];
  temporal_results: Record<string, Record<string, { rate: number }>>;
  fraction_results: Record<string, Record<string, Record<string, { rate: number }>>>;
}

// Grid ablation data (layer x action phase combinations)
interface GridAblationData {
  layers: number[];
  phases: string[];
  data: (number | null)[][];
}

// Temporal ablation data by timestep
interface TimestepChartData {
  timesteps: number[];
  series: Array<{
    name: string;
    baseline: number;
    values: number[];
  }>;
}

// Fractional ablation data (dose-response)
interface FractionalChartData {
  percentages: number[];
  series: Array<{
    name: string;
    baseline: number;
    values: number[];
  }>;
}

// Comprehensive results data
interface ComprehensiveResults {
  action_phase_heatmap: GridAblationData;
  feature_percentage_chart: FractionalChartData;
  timestep_chart: TimestepChartData;
  concepts_layers_heatmap: any;
  summary: {
    generated_at: string;
    experiments: {
      action_phases: {
        most_impactful_negative: { layer: number; phase: string; delta: number };
        most_impactful_positive: { layer: number; phase: string; delta: number };
      };
    };
  };
}

// Video data interface
interface VideoData {
  path: string;
  experiment_type: string;
  suite: string;
  subtype?: string;
  seed?: number;
  task?: number;
  success?: boolean;
  concept?: string;
  model?: string;
  layer?: number;
  episode?: number;
}

// Aggregated statistics
interface AggregatedStats {
  totalVideos: number;
  totalVideosInIndex: number;  // Total videos reported by API (may be larger than loaded)
  successCount: number;
  failureCount: number;
  unknownCount: number;
  baselineCount: number;  // Track baselines separately
  byExperimentType: Record<string, { total: number; success: number; failure: number; unknown: number }>;
  bySuite: Record<string, { total: number; success: number; failure: number; unknown: number }>;
  bySubtype: Record<string, { total: number; success: number; failure: number; unknown: number }>;
}

interface AblationVisualizationsProps {
  selectedConcept?: string;
  onConceptSelect?: (concept: string) => void;
}

// Helper function to parse success/failure from filename
// Returns: true (success), false (failure), or null (unknown)
const parseVideoSuccess = (video: VideoData): boolean | null => {
  // If explicit success field is provided, use it
  if (video.success !== undefined) {
    return video.success;
  }

  const path = video.path?.toLowerCase() || '';
  const filename = path.split('/').pop() || '';

  // Check for explicit keywords only - NOT seed patterns like _s42
  const hasSuccessKeyword = filename.includes('success') || filename.includes('_success');
  const hasFailureKeyword = filename.includes('fail') || filename.includes('_fail');

  if (hasSuccessKeyword && !hasFailureKeyword) return true;
  if (hasFailureKeyword && !hasSuccessKeyword) return false;

  // Return null for unknown - don't assume success
  return null;
};

// Helper function to get video URL
const getVideoUrl = (videoPath: string): string => {
  return `${API_BASE_URL}/api/vla/video/${videoPath}`;
};

// Videos per page for video grids
const VIDEOS_PER_PAGE = 24;

export default function AblationVisualizations({ selectedConcept, onConceptSelect }: AblationVisualizationsProps) {
  const [data, setData] = useState<ExperimentData | null>(null);
  const [comprehensiveData, setComprehensiveData] = useState<ComprehensiveResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [localSelectedConcept, setLocalSelectedConcept] = useState<string>("motion/put");
  const [mainTab, setMainTab] = useState(0); // 0: Overview, 1: Layer-Phase Matrix, 2: Temporal, 3: Fractional

  // Video data state
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [videosLoading, setVideosLoading] = useState(true);
  const [aggregatedStats, setAggregatedStats] = useState<AggregatedStats | null>(null);

  // Pagination state for each tab
  const [overviewVideoPage, setOverviewVideoPage] = useState(1);
  const [gridVideoPage, setGridVideoPage] = useState(1);
  const [temporalVideoPage, setTemporalVideoPage] = useState(1);
  const [fractionalVideoPage, setFractionalVideoPage] = useState(1);

  // Data state for SmolVLA, X-VLA, GR00T ablation summaries
  const [ablationSummaryData, setAblationSummaryData] = useState<any[]>([]);
  const [modelExperimentResults, setModelExperimentResults] = useState<any>(null);

  // Get current model and dataset from Redux
  const { currentModel, currentDataset } = useAppSelector((state) => ({ currentModel: state.model.currentModel, currentDataset: state.model.currentDataset }));

  const activeConcept = selectedConcept || localSelectedConcept;

  // Fetch videos from API
  const fetchVideos = useCallback(async () => {
    setVideosLoading(true);
    try {
      // Build suite filter from current dataset
      const datasetSuites = DATASET_SUITES[currentDataset as DatasetType] || [];
      const suiteValues = datasetSuites.map(s => s.value);

      // Fetch from main videos endpoint (with limit to avoid loading 30K+ entries)
      const response = await fetch(`${API_BASE_URL}/api/vla/videos?model=${currentModel}&limit=5000`);
      let allVideos: VideoData[] = [];
      let totalVideoCount = 0;
      if (response.ok) {
        const data = await response.json();
        allVideos = data.videos || [];
        totalVideoCount = data.total || allVideos.length;
      }

      // For models without main video index, also fetch from the ablation videos endpoint
      if (allVideos.length === 0) {
        try {
          const ablationResp = await fetch(`${API_BASE_URL}/api/ablation/videos?model=${currentModel}&limit=5000`);
          if (ablationResp.ok) {
            const ablationData = await ablationResp.json();
            const ablationVids = ablationData.data?.videos || [];
            totalVideoCount = ablationData.data?.total || ablationVids.length;
            // Map entries to VideoData format if needed
            allVideos = ablationVids.map((v: any) => ({
              ...v,
              experiment_type: v.experiment_type || 'concept_ablation',
              model: currentModel,
            }));
          }
        } catch (err) {
          console.warn('Failed to fetch ablation videos:', err);
        }
      }

      // Filter videos to only include suites belonging to the current dataset
      if (suiteValues.length > 0) {
        allVideos = allVideos.filter(v => {
          if (!v.suite) return true; // Keep videos without suite info
          return suiteValues.includes(v.suite);
        });
      }

      if (allVideos.length > 0) {
        setVideos(allVideos);

        // Aggregate statistics
        const stats: AggregatedStats = {
          totalVideos: 0,  // Will count non-baseline videos only
          totalVideosInIndex: totalVideoCount,  // Total from API (may be larger)
          successCount: 0,
          failureCount: 0,
          unknownCount: 0,
          baselineCount: 0,
          byExperimentType: {},
          bySuite: {},
          bySubtype: {},
        };

        allVideos.forEach((video) => {
          // Check if this is a baseline experiment type - EXCLUDE from experiment counts
          // Note: only filter on experiment_type/subtype metadata, NOT path patterns
          // Path patterns like "_baseline" catch experiment CONDITIONS (e.g. task0_baseline_generic.mp4)
          // which are valid control conditions within experiments, not baseline runs
          const isBaseline =
            video.experiment_type === 'baseline' ||
            video.subtype === 'baseline';

          if (isBaseline) {
            // Track baseline count but don't include in any other stats
            stats.baselineCount++;
            return; // Skip this video entirely
          }

          // Only count non-baseline videos
          stats.totalVideos++;
          const successResult = parseVideoSuccess(video);

          if (successResult === true) {
            stats.successCount++;
          } else if (successResult === false) {
            stats.failureCount++;
          } else {
            stats.unknownCount++;
          }

          // By experiment type
          const expType = video.experiment_type || 'unknown';
          if (!stats.byExperimentType[expType]) {
            stats.byExperimentType[expType] = { total: 0, success: 0, failure: 0, unknown: 0 };
          }
          stats.byExperimentType[expType].total++;
          if (successResult === true) {
            stats.byExperimentType[expType].success++;
          } else if (successResult === false) {
            stats.byExperimentType[expType].failure++;
          } else {
            stats.byExperimentType[expType].unknown++;
          }

          // By suite
          const suite = video.suite || 'unknown';
          if (!stats.bySuite[suite]) {
            stats.bySuite[suite] = { total: 0, success: 0, failure: 0, unknown: 0 };
          }
          stats.bySuite[suite].total++;
          if (successResult === true) {
            stats.bySuite[suite].success++;
          } else if (successResult === false) {
            stats.bySuite[suite].failure++;
          } else {
            stats.bySuite[suite].unknown++;
          }

          // By subtype (especially for vision_perturbation)
          const subtype = video.subtype || 'other';
          if (!stats.bySubtype[subtype]) {
            stats.bySubtype[subtype] = { total: 0, success: 0, failure: 0, unknown: 0 };
          }
          stats.bySubtype[subtype].total++;
          if (successResult === true) {
            stats.bySubtype[subtype].success++;
          } else if (successResult === false) {
            stats.bySubtype[subtype].failure++;
          } else {
            stats.bySubtype[subtype].unknown++;
          }
        });

        setAggregatedStats(stats);
      } else {
        setVideos([]);
        setAggregatedStats(null);
      }
    } catch (err) {
      console.error('Failed to fetch videos:', err);
    } finally {
      setVideosLoading(false);
    }
  }, [currentModel, currentDataset]);

  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  useEffect(() => {
    // Reset stale state when model or dataset changes
    setData(null);
    setComprehensiveData(null);
    setAblationSummaryData([]);
    setModelExperimentResults(null);
    setLoading(true);
    // If mainTab points to a tab that may be disabled for the new model, reset to Overview.
    // Tab 1 (Layer-Phase) depends on data that was just cleared above.
    // Tabs 2 (Temporal) and 3 (Fractional) are only enabled for pi05/groot/smolvla.
    if (mainTab === 1) {
      // Data was cleared; tab 1 will be disabled until new data loads
      setMainTab(0);
    } else if ((mainTab === 2 || mainTab === 3) && !['groot', 'smolvla'].includes(currentModel)) {
      setMainTab(0);
    }
    // Load experiment summary and comprehensive results, model-scoped
    // Try model-specific paths first, then fall back to generic
    const modelPrefixMap: Record<string, string> = {
      openvla: 'openvla',
      pi05: 'pi05',
      xvla: 'xvla',
      smolvla: 'smolvla',
      groot: 'groot',
      act: 'act',
    };
    const modelPrefix = modelPrefixMap[currentModel] || currentModel;
    Promise.all([
      fetch(`/${modelPrefix}_experiment_summary.json`)
        .then((res) => res.ok ? res.json() : null)
        .catch(() => null)
        .then((data) => data || fetch("/experiment_summary.json").then((res) => res.ok ? res.json() : null).catch(() => null)),
      fetch(`/${modelPrefix}_comprehensive_results.json`)
        .then((res) => res.ok ? res.json() : null)
        .catch(() => null)
        .then((data) => data || fetch("/comprehensive_results.json").then((res) => res.ok ? res.json() : null).catch(() => null)),
      // Also fetch ablation summary from the model-aware API
      fetch(`${API_BASE_URL}/api/ablation/summary?model=${currentModel}`)
        .then((res) => res.ok ? res.json() : null)
        .catch(() => null),
      // Fetch model-specific experiment results (for SmolVLA, X-VLA, GR00T)
      fetch(`${API_BASE_URL}/api/experiments/${currentModel}`)
        .then((res) => res.ok ? res.json() : null)
        .catch(() => null),
      // Fetch temporal ablation data from dedicated API
      fetch(`${API_BASE_URL}/api/vla/temporal_ablation/${currentModel}`)
        .then((res) => res.ok ? res.json() : null)
        .catch(() => null),
    ])
      .then(([expData, compData, ablationSummary, experimentResults, temporalApiResult]) => {
        if (expData) {
          setData(expData);
          if (!selectedConcept && expData.temporal_concepts?.length > 0) {
            setLocalSelectedConcept(expData.temporal_concepts[0]);
          }
        }
        if (compData) {
          setComprehensiveData(compData);
        }
        // If no experiment data loaded but we have ablation summary, use its concepts
        if (!expData && ablationSummary?.data?.summary) {
          const concepts = ablationSummary.data.summary.map((s: any) => s.concept);
          if (concepts.length > 0 && !selectedConcept) {
            setLocalSelectedConcept(concepts[0]);
          }
        }
        // For models with ablation_summary entries (SmolVLA, X-VLA, GR00T)
        if (!expData && ablationSummary?.data?.ablation_summary) {
          const entries = ablationSummary.data.ablation_summary;
          // Store ablation summary data for use in rendering
          setAblationSummaryData(entries);
        }
        // Store experiment results - keep full API response for metadata
        // Merge temporal API data into experiment results if available
        const temporalApiData = temporalApiResult?.data || null;
        if (experimentResults?.data) {
          const mergedData = { ...experimentResults.data, _temporalApiData: temporalApiData };
          setModelExperimentResults(mergedData);
          // Extract first concept from experiment_results for initial selection
          if (!expData && !selectedConcept) {
            const fd = experimentResults.data.full_data || experimentResults.data;
            const stSec = fd?.concept_steering || fd?.steering;
            if (stSec && typeof stSec === 'object') {
              for (const val of Object.values(stSec) as any[]) {
                if (val?.concepts && typeof val.concepts === 'object') {
                  const firstConcept = Object.keys(val.concepts)[0];
                  if (firstConcept) {
                    setLocalSelectedConcept(firstConcept.includes('/') ? firstConcept : firstConcept.replace('_', '/'));
                    break;
                  }
                }
              }
            }
          }
        } else if (experimentResults?.baselines) {
          setModelExperimentResults({ ...experimentResults, _temporalApiData: temporalApiData });
        } else if (temporalApiData) {
          // Even if no experiment results, store temporal data
          setModelExperimentResults({ _temporalApiData: temporalApiData });
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load experiment data:", err);
        setLoading(false);
      });
  }, [selectedConcept, currentModel, currentDataset]);

  const handleConceptChange = (concept: string) => {
    setLocalSelectedConcept(concept);
    if (onConceptSelect) {
      onConceptSelect(concept);
    }
  };

  // Filter videos by experiment type and optionally by concept
  const getFilteredVideos = useCallback((experimentType?: string, concept?: string) => {
    let filtered = videos;
    if (experimentType) {
      filtered = filtered.filter((v) => v.experiment_type === experimentType);
    }
    if (concept) {
      filtered = filtered.filter((v) => {
        if (!v.concept) return false;
        // Match concept in slash format (motion/put) or underscore format (motion_put)
        const normalizedConcept = concept.replace('/', '_');
        return v.concept === concept || v.concept === normalizedConcept;
      });
    }
    return filtered;
  }, [videos]);

  // Get videos for each tab, filtered by active concept when applicable
  // Note: 'grid_ablation' only exists in the OFT index, not Pi0.5
  const gridVideos = useMemo(() => getFilteredVideos('grid_ablation'), [getFilteredVideos]);
  const conceptAblationVideos = useMemo(() => getFilteredVideos('concept_ablation', activeConcept), [getFilteredVideos, activeConcept]);
  const temporalVideos = useMemo(() => getFilteredVideos('temporal_perturbation'), [getFilteredVideos]);
  const crossSceneInjectionVideos = useMemo(() => getFilteredVideos('cross_scene_injection'), [getFilteredVideos]);
  const temporalInjectionVideos = useMemo(() => getFilteredVideos('temporal_injection'), [getFilteredVideos]);

  // Prepare chart data from aggregated stats
  const experimentTypeChartData = useMemo(() => {
    if (!aggregatedStats) return [];
    return Object.entries(aggregatedStats.byExperimentType)
      .map(([type, stats]) => {
        const knownCount = stats.success + stats.failure;
        return {
          name: type.replace(/_/g, ' '),
          successRate: knownCount > 0 ? Math.round((stats.success / knownCount) * 100) : null,
          total: stats.total,
          success: stats.success,
          failure: stats.failure,
          unknown: stats.unknown,
          knownCount,
        };
      }).sort((a, b) => b.total - a.total);
  }, [aggregatedStats]);

  const suiteChartData = useMemo(() => {
    if (!aggregatedStats) return [];
    return Object.entries(aggregatedStats.bySuite).map(([suite, stats]) => {
      const knownCount = stats.success + stats.failure;
      return {
        name: suite.replace(/_/g, ' '),
        value: stats.total,
        successRate: knownCount > 0 ? Math.round((stats.success / knownCount) * 100) : null,
        success: stats.success,
        failure: stats.failure,
        unknown: stats.unknown,
      };
    }).sort((a, b) => b.value - a.value);
  }, [aggregatedStats]);

  // Prepare subtype chart data (for vision perturbation breakdown)
  // Filter out "baseline" and "other" subtypes since they're not informative
  const subtypeChartData = useMemo(() => {
    if (!aggregatedStats) return [];
    return Object.entries(aggregatedStats.bySubtype)
      .filter(([subtype]) => subtype !== 'baseline' && subtype !== 'other')
      .map(([subtype, stats]) => {
        const knownCount = stats.success + stats.failure;
        return {
          name: subtype.replace(/_/g, ' '),
          total: stats.total,
          successRate: knownCount > 0 ? Math.round((stats.success / knownCount) * 100) : null,
          success: stats.success,
          failure: stats.failure,
          unknown: stats.unknown,
        };
      }).sort((a, b) => b.total - a.total);
  }, [aggregatedStats]);

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Gracefully handle missing experiment data - still show video analytics
  // Build concept list from the best available source for the current model
  const allConcepts = (() => {
    // 1. If we have experiment_summary.json data (Pi0.5), use it
    if (data) {
      return Array.from(new Set([...data.temporal_concepts, ...data.fraction_concepts]));
    }
    // 2. Extract concepts from ablationSummaryData (SmolVLA, X-VLA, GR00T index)
    if (ablationSummaryData && ablationSummaryData.length > 0) {
      const conceptSet = new Set<string>();
      ablationSummaryData.forEach((entry: any) => {
        const concept = entry.concept;
        if (concept && concept.includes('/')) {
          conceptSet.add(concept);
        }
      });
      if (conceptSet.size > 0) return Array.from(conceptSet).sort();
    }
    // 3. Extract concepts from modelExperimentResults (experiment_results_*.json)
    if (modelExperimentResults) {
      const fullData = modelExperimentResults.full_data || modelExperimentResults;
      const conceptSet = new Set<string>();
      // Check concept_steering (X-VLA) or steering (GR00T, SmolVLA)
      const steeringSection = fullData?.concept_steering || fullData?.steering;
      if (steeringSection && typeof steeringSection === 'object') {
        Object.values(steeringSection).forEach((entry: any) => {
          if (entry?.concepts && typeof entry.concepts === 'object') {
            Object.keys(entry.concepts).forEach((c: string) => {
              const normalized = c.includes('/') ? c : c.replace('_', '/');
              conceptSet.add(normalized);
            });
          }
        });
      }
      // Check concept_ablation kill_switches for concept names
      const caSection = fullData?.concept_ablation;
      if (caSection && typeof caSection === 'object') {
        const ks = caSection.kill_switches;
        if (Array.isArray(ks)) {
          ks.forEach((k: any) => {
            const name = typeof k === 'string' ? k : k?.concept;
            if (name && name.includes('/')) conceptSet.add(name);
          });
        }
      }
      if (conceptSet.size > 0) return Array.from(conceptSet).sort();
    }
    // 4. Extract from videos that have concept metadata
    if (videos.length > 0) {
      const conceptSet = new Set<string>();
      videos.forEach(v => {
        if (v.concept && v.concept.includes('/')) conceptSet.add(v.concept);
      });
      if (conceptSet.size > 0) return Array.from(conceptSet).sort();
    }
    // 5. Fallback defaults
    return ['motion/pick', 'motion/put', 'motion/open', 'motion/close', 'motion/push', 'motion/interact', 'object/bowl', 'object/plate', 'spatial/on'];
  })();

  const getFractionDataForConcept = (concept: string) => {
    if (!data?.fraction_results[concept]) return [];
    return [0, 5, 10, 20, 50, 100].map((n) => {
      const conceptData = data.fraction_results[concept];
      if (conceptData[String(n)]) {
        const rates = Object.values(conceptData[String(n)]).map((v: any) => v.rate);
        const avgRate = rates.length > 0 ? Math.round(rates.reduce((a, b) => a + b, 0) / rates.length) : 0;
        return { features: n, rate: avgRate };
      }
      return { features: n, rate: 0 };
    });
  };

  const temporalConditions = [
    { key: "baseline", label: "Baseline" },
    { key: "full_episode", label: "Full Episode" },
    { key: "step_0_only", label: "Step 0 Only" },
    { key: "warmup_10", label: "Warmup 10" },
    { key: "warmup_50", label: "Warmup 50" },
    { key: "random_50pct", label: "Random 50%" },
  ];

  const temporalData = data?.temporal_results[activeConcept]
    ? temporalConditions.map((cond) => ({
        window: cond.label,
        rate: data.temporal_results[activeConcept]?.[cond.key]?.rate ?? 0,
      }))
    : [];

  const fractionData = getFractionDataForConcept(activeConcept);

  // Helper function to get color for heatmap cell based on value
  const getHeatmapColor = (value: number | null) => {
    if (value === null) return "#374151"; // Gray for null
    if (value > 15) return "#22c55e"; // Strong positive - green
    if (value > 5) return "#84cc16"; // Moderate positive - light green
    if (value > -5) return "#fbbf24"; // Neutral - yellow
    if (value > -15) return "#f97316"; // Moderate negative - orange
    return "#ef4444"; // Strong negative - red
  };

  // Prepare grid ablation heatmap data - from comprehensiveData (Pi0.5 ONLY) or experiment_results
  const gridHeatmapData = (() => {
    if (currentModel === 'pi05' && comprehensiveData?.action_phase_heatmap) return comprehensiveData.action_phase_heatmap;
    // Build from modelExperimentResults grid_ablation for other models
    const fullData = modelExperimentResults?.full_data || modelExperimentResults;
    const gaSection = fullData?.grid_ablation;
    if (!gaSection || typeof gaSection !== 'object') return undefined;
    // Filter suites by currentDataset
    const datasetSuites = DATASET_SUITES[currentDataset as DatasetType] || [];
    const suiteValues = datasetSuites.map((s: any) => s.value);
    // Helper to extract baseline as a number from any format
    const getBaselineNum = (suite: any): number => {
      const blOverall = suite?.baseline_overall;
      if (typeof blOverall === 'number') return blOverall;
      const bl = suite?.baseline;
      if (typeof bl === 'number') return bl;
      // SmolVLA: baseline is a dict of per-task SRs - average them
      if (bl && typeof bl === 'object' && !Array.isArray(bl)) {
        const srs = Object.values(bl).map((v: any) => v?.success_rate).filter((v: any) => typeof v === 'number') as number[];
        return srs.length > 0 ? srs.reduce((a, b) => a + b, 0) / srs.length : 1.0;
      }
      return 1.0;
    };

    // Find suite data with per_layer OR per_condition (SmolVLA)
    const suitesToUse = Object.entries(gaSection).filter(([key, val]: [string, any]) => {
      if (typeof val !== 'object') return false;
      if (!val?.per_layer && !val?.per_condition) return false;
      if (suiteValues.length > 0) return suiteValues.includes(key);
      return true;
    });
    if (suitesToUse.length === 0) return undefined;

    // Check if data uses per_layer (X-VLA, GR00T) or per_condition (SmolVLA)
    const [, firstSuite] = suitesToUse[0] as [string, any];
    const usePerLayer = !!firstSuite.per_layer;

    if (usePerLayer) {
      // Standard per_layer grid (X-VLA, GR00T)
      const perLayer = firstSuite.per_layer;
      const layerEntries = Object.entries(perLayer).sort(([a], [b]) => {
        const numA = parseInt(a.replace(/\D/g, ''));
        const numB = parseInt(b.replace(/\D/g, ''));
        return numA - numB;
      });
      if (layerEntries.length === 0) return undefined;
      const layers = layerEntries.map(([key]) => {
        const num = parseInt(key.replace(/\D/g, ''));
        return isNaN(num) ? 0 : num;
      });
      const phases = suitesToUse.map(([k]) => k.replace(/_/g, ' '));
      const dataGrid: (number | null)[][] = layerEntries.map(([layerKey]) => {
        return suitesToUse.map(([, sd]: [string, any]) => {
          const entry = sd.per_layer?.[layerKey];
          if (!entry) return null;
          const sr = entry.overall_success_rate;
          if (typeof sr !== 'number') return null;
          const bl = getBaselineNum(sd);
          const delta = Math.round((sr - bl) * 1000) / 10;
          return isNaN(delta) ? null : delta;
        });
      });
      return { layers, phases, data: dataGrid } as GridAblationData;
    } else {
      // Condition-based grid (SmolVLA) - show conditions as rows, suites as columns
      // Conditions are named "expert_0", "expert_1", etc. representing layer indices
      const conditions = Object.keys(firstSuite.per_condition || {}).filter(c => c !== 'baseline');
      if (conditions.length === 0) return undefined;
      // Sort conditions by numeric suffix so layers appear in order
      conditions.sort((a, b) => {
        const numA = parseInt(a.replace(/\D/g, ''));
        const numB = parseInt(b.replace(/\D/g, ''));
        return numA - numB;
      });
      const layers = conditions.map(c => {
        const num = parseInt(c.replace(/\D/g, ''));
        return isNaN(num) ? 0 : num;
      });
      const phases = suitesToUse.map(([k]) => k.replace(/_/g, ' '));
      const dataGrid: (number | null)[][] = conditions.map((condKey) => {
        return suitesToUse.map(([, sd]: [string, any]) => {
          const cond = sd.per_condition?.[condKey];
          if (!cond || typeof cond !== 'object') return null;
          // Use overall_success_rate if available, otherwise average per_task values
          let condSr: number;
          if (typeof cond.overall_success_rate === 'number') {
            condSr = cond.overall_success_rate;
          } else {
            const perTask = cond.per_task || cond.tasks || {};
            const taskSrs = Object.values(perTask).filter((v: any) => typeof v === 'number') as number[];
            if (taskSrs.length === 0) return null;
            condSr = taskSrs.reduce((a, b) => a + b, 0) / taskSrs.length;
          }
          const bl = getBaselineNum(sd);
          const delta = Math.round((condSr - bl) * 1000) / 10;
          return isNaN(delta) ? null : delta;
        });
      });
      return { layers, phases, data: dataGrid } as GridAblationData;
    }
  })();

  // Prepare timestep chart data for temporal ablations (Pi0.5 only from comprehensive_results)
  const timestepChartData = currentModel === 'pi05' ? comprehensiveData?.timestep_chart : undefined;

  // Prepare fractional chart data for dose-response curves - from comprehensiveData or experiment_results
  const fractionalChartData = (() => {
    if (currentModel === 'pi05' && comprehensiveData?.feature_percentage_chart) return comprehensiveData.feature_percentage_chart;
    // Build from GR00T/SmolVLA fraction_to_failure data
    const fullData = modelExperimentResults?.full_data || modelExperimentResults;
    const ftfSection = fullData?.fraction_to_failure;
    if (!ftfSection || typeof ftfSection !== 'object') return undefined;
    // Filter by currentDataset suites
    const datasetSuites = DATASET_SUITES[currentDataset as DatasetType] || [];
    const suiteValues = datasetSuites.map((s: any) => s.value);
    // Find suites with per_layer data (GR00T format)
    const suitesToUse = Object.entries(ftfSection).filter(([key, val]: [string, any]) => {
      if (typeof val !== 'object' || !val?.per_layer) return false;
      if (suiteValues.length > 0) return suiteValues.includes(key);
      return true;
    });
    // Handle aggregate FTF data (SmolVLA format: count=conditions, episodes=total rollouts, no per_layer)
    // This is aggregate-only data without per-layer titration curves.
    // Do NOT generate a chart (count/episodes is NOT a success rate).
    // The summary panel will display the aggregate stats instead.
    if (suitesToUse.length === 0) {
      return undefined;
    }
    // Use first matching suite, first layer's titration data
    const [, suiteData] = suitesToUse[0] as [string, any];
    const perLayer = suiteData.per_layer;
    const layerKeys = Object.keys(perLayer).sort();
    // Collect titration curves from first few layers
    const seriesArr: Array<{name: string; baseline: number; values: number[]}> = [];
    let percentages: number[] = [];
    for (const layerKey of layerKeys.slice(0, 6)) {
      const layerData = perLayer[layerKey];
      // GR00T FTF has "frequent" or "random" sub-keys with titration arrays
      const titrationSource = layerData?.frequent || layerData?.random || layerData;
      if (!titrationSource?.titration || !Array.isArray(titrationSource.titration)) continue;
      const titration = titrationSource.titration;
      const bl = titrationSource.baseline_success_rate ?? 1.0;
      if (percentages.length === 0) {
        percentages = titration.map((t: any) => t.n_features ?? 0);
      }
      seriesArr.push({
        name: layerKey.replace(/_/g, ' '),
        baseline: Math.round(bl * 100),
        values: titration.map((t: any) => Math.round((t.success_rate ?? 0) * 100)),
      });
    }
    if (seriesArr.length === 0 || percentages.length === 0) return undefined;
    return { percentages, series: seriesArr } as FractionalChartData;
  })();

  // Render Video Grid Component
  const renderVideoGrid = (
    videoList: VideoData[],
    currentPage: number,
    setPage: (page: number) => void,
    title?: string
  ) => {
    const totalPages = Math.ceil(videoList.length / VIDEOS_PER_PAGE);
    const paginatedVideos = videoList.slice(
      (currentPage - 1) * VIDEOS_PER_PAGE,
      currentPage * VIDEOS_PER_PAGE
    );

    if (videoList.length === 0) {
      return (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No videos available for this experiment type
          </Typography>
        </Paper>
      );
    }

    return (
      <Paper sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            {title || 'Videos'} ({videoList.length} total)
          </Typography>
          {totalPages > 1 && (
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              <Button
                size="small"
                disabled={currentPage === 1}
                onClick={() => setPage(Math.max(1, currentPage - 1))}
                sx={{ minWidth: 32, fontSize: '11px' }}
              >
                Prev
              </Button>
              <Typography variant="caption" color="text.secondary">
                {currentPage} / {totalPages}
              </Typography>
              <Button
                size="small"
                disabled={currentPage === totalPages}
                onClick={() => setPage(Math.min(totalPages, currentPage + 1))}
                sx={{ minWidth: 32, fontSize: '11px' }}
              >
                Next
              </Button>
            </Box>
          )}
        </Box>
        <Grid container spacing={2}>
          {paginatedVideos.map((video, index) => {
            const isSuccess = parseVideoSuccess(video);
            return (
              <Grid item xs={12} sm={6} md={3} key={video.path || index}>
                <Card
                  variant="outlined"
                  sx={{
                    transition: 'all 0.2s',
                    '&:hover': {
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      borderColor: '#ef4444',
                    },
                  }}
                >
                  <CardContent sx={{ p: 1.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                      <PlayCircleOutlineIcon
                        fontSize="small"
                        sx={{ color: isSuccess === true ? '#22c55e' : isSuccess === false ? '#ef4444' : '#9e9e9e', fontSize: 16 }}
                      />
                      <Typography variant="caption" sx={{ fontWeight: 600, flex: 1, fontSize: 10 }}>
                        {video.experiment_type?.replace(/_/g, ' ') || 'Unknown'}
                      </Typography>
                      {isSuccess !== null && (
                      <Chip
                        icon={isSuccess ? <CheckCircleIcon sx={{ fontSize: '10px !important' }} /> : <CancelIcon sx={{ fontSize: '10px !important' }} />}
                        label={isSuccess ? 'Success' : 'Failure'}
                        size="small"
                        color={isSuccess ? 'success' : 'error'}
                        sx={{ height: 16, fontSize: '8px', '& .MuiChip-icon': { ml: 0.5 } }}
                      />
                      )}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 0.5, mb: 1, flexWrap: 'wrap' }}>
                      <Chip
                        label={video.suite?.replace(/_/g, ' ') || 'N/A'}
                        size="small"
                        variant="outlined"
                        sx={{ height: 14, fontSize: '8px', textTransform: 'capitalize' }}
                      />
                      {video.task !== undefined && (
                        <Chip
                          label={`T${video.task}`}
                          size="small"
                          variant="outlined"
                          sx={{ height: 14, fontSize: '8px' }}
                        />
                      )}
                    </Box>
                    <Box
                      sx={{
                        width: '100%',
                        aspectRatio: '4/3',
                        backgroundColor: '#0a1628',
                        borderRadius: 1,
                        overflow: 'hidden',
                      }}
                    >
                      <video
                        src={getVideoUrl(video.path)}
                        controls
                        muted
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain',
                          // SmolVLA LIBERO videos are recorded with 180-degree flip baked in
                          ...(currentModel === 'smolvla' && video.path && !video.path.includes('metaworld')
                            ? { transform: 'rotate(180deg)' } : {}),
                        }}
                        onError={(e) => {
                          (e.target as HTMLVideoElement).style.display = 'none';
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Paper>
    );
  };

  // Render the Grid Ablation Heatmap component
  const renderGridHeatmap = () => {
    if (!gridHeatmapData?.phases?.length || !gridHeatmapData?.layers?.length || !gridHeatmapData?.data) {
      // Show placeholder with video counts if no comprehensive results
      const allAblationVideoCount = gridVideos.length + conceptAblationVideos.length;
      const ablationStats = aggregatedStats?.byExperimentType['grid_ablation'] || aggregatedStats?.byExperimentType['concept_ablation'];
      return (
        <Paper sx={{ p: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
            {gridVideos.length > 0 ? 'Layer x Action Phase Ablation Grid' : 'Concept Ablation Results'}
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            {gridVideos.length > 0
              ? 'Comprehensive grid ablation results not available.'
              : 'Grid ablation results not available for this model.'}
            {allAblationVideoCount > 0
              ? ` Found ${allAblationVideoCount} ablation videos below.`
              : ' No ablation videos found for this model.'}
          </Alert>
          {ablationStats && (
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip
                label={`${ablationStats.total} Videos`}
                size="small"
                color="primary"
              />
              <Chip
                label={`${ablationStats.success} Success`}
                size="small"
                color="success"
              />
              <Chip
                label={`${ablationStats.failure} Failure`}
                size="small"
                color="error"
              />
            </Box>
          )}
        </Paper>
      );
    }

    return (
      <Paper sx={{ p: 2, maxHeight: 500, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          Layer x Action Phase Ablation Grid
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 2 }}>
          Success rate delta (%) when ablating features at each layer during each action phase.
          Green = positive effect, Red = negative effect, Gray = not tested
        </Typography>
        <Box sx={{ maxHeight: 350, overflow: 'auto' }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, width: 'fit-content' }}>
          {/* Header row with phase labels */}
          <Box sx={{ display: "flex", gap: 0.5, pl: 6 }}>
            {gridHeatmapData.phases.map((phase) => (
              <Box
                key={phase}
                sx={{
                  width: 70,
                  minWidth: 70,
                  flexShrink: 0,
                  textAlign: "center",
                  fontSize: 10,
                  fontWeight: 500,
                  textTransform: "capitalize",
                }}
              >
                {phase}
              </Box>
            ))}
          </Box>
          {/* Heatmap rows */}
          {gridHeatmapData.layers.map((layer, layerIdx) => (
            <Box key={layer} sx={{ display: "flex", alignItems: "center", gap: 0.5, width: 'fit-content' }}>
              <Box sx={{ width: 50, minWidth: 50, flexShrink: 0, fontSize: 11, fontWeight: 500, textAlign: "right", pr: 1 }}>
                L{layer}
              </Box>
              {gridHeatmapData.phases.map((phase, phaseIdx) => {
                const value = gridHeatmapData.data?.[layerIdx]?.[phaseIdx] ?? null;
                return (
                  <MuiTooltip
                    key={`${layer}-${phase}`}
                    title={typeof value === 'number' ? `Layer ${layer}, ${phase}: ${value > 0 ? "+" : ""}${value.toFixed(1)}%` : "Not tested"}
                    arrow
                  >
                    <Box
                      sx={{
                        width: 70,
                        minWidth: 70,
                        flexShrink: 0,
                        height: 36,
                        bgcolor: getHeatmapColor(value),
                        borderRadius: 0.5,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 10,
                        fontWeight: 600,
                        color: typeof value === 'number' && Math.abs(value) < 10 ? "#1f2937" : "#fff",
                        cursor: "pointer",
                        transition: "transform 0.1s",
                        "&:hover": { transform: "scale(1.05)" },
                      }}
                    >
                      {typeof value === 'number' ? `${value > 0 ? "+" : ""}${value.toFixed(1)}` : "-"}
                    </Box>
                  </MuiTooltip>
                );
              })}
            </Box>
          ))}
          </Box>
        </Box>
        {/* Legend */}
        <Box sx={{ display: "flex", justifyContent: "center", gap: 2, mt: 2 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 16, height: 16, bgcolor: "#22c55e", borderRadius: 0.5 }} />
            <Typography variant="caption">&gt;+15%</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 16, height: 16, bgcolor: "#84cc16", borderRadius: 0.5 }} />
            <Typography variant="caption">+5 to +15%</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 16, height: 16, bgcolor: "#fbbf24", borderRadius: 0.5 }} />
            <Typography variant="caption">-5 to +5%</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 16, height: 16, bgcolor: "#f97316", borderRadius: 0.5 }} />
            <Typography variant="caption">-15 to -5%</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box sx={{ width: 16, height: 16, bgcolor: "#ef4444", borderRadius: 0.5 }} />
            <Typography variant="caption">&lt;-15%</Typography>
          </Box>
        </Box>
      </Paper>
    );
  };

  // Render Temporal Ablation by Timestep
  const renderTemporalTimestepChart = () => {
    if (!timestepChartData) {
      return (
        <Box sx={{ p: 2, textAlign: "center" }}>
          <Typography color="text.secondary">No timestep ablation data available</Typography>
        </Box>
      );
    }

    const chartData = timestepChartData.timesteps.map((ts, idx) => {
      const point: Record<string, number | string> = { timestep: `T${ts}` };
      timestepChartData.series.forEach((s) => {
        point[s.name] = s.values[idx];
        point[`${s.name}_baseline`] = s.baseline;
      });
      return point;
    });

    return (
      <Paper sx={{ p: 2, height: 350 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Temporal Ablation by Timestep
          </Typography>
          <MuiTooltip title="Data from preliminary experiments. Full-scale experiments in progress.">
            <Chip label="Preliminary" size="small" sx={{ height: 18, fontSize: '9px', bgcolor: '#fef3c7', color: '#92400e' }} />
          </MuiTooltip>
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
          Success rate (%) when ablating features starting at different timesteps.
          Dashed lines show baseline performance.
        </Typography>
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="timestep" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} label={{ value: "Success %", angle: -90, position: "insideLeft", fontSize: 10 }} />
            <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${v.toFixed(1)}%` : `${v ?? '-'}`, ""]} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {timestepChartData.series.map((s, idx) => (
              <React.Fragment key={s.name}>
                <Line
                  type="monotone"
                  dataKey={s.name}
                  stroke={CONCEPT_COLORS[s.name] || LAYER_COLORS[idx % LAYER_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name={s.name.charAt(0).toUpperCase() + s.name.slice(1)}
                />
                <ReferenceLine
                  y={s.baseline}
                  stroke={CONCEPT_COLORS[s.name] || LAYER_COLORS[idx % LAYER_COLORS.length]}
                  strokeDasharray="5 5"
                  strokeOpacity={0.5}
                />
              </React.Fragment>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    );
  };

  // Render Fractional Ablation Dose-Response Curves
  const renderFractionalDoseResponse = () => {
    if (!fractionalChartData) {
      return (
        <Box sx={{ p: 2, textAlign: "center" }}>
          <Typography color="text.secondary">No fractional ablation data available</Typography>
        </Box>
      );
    }

    // Detect if values are feature counts (>1) or percentages (0-1 or 0-100)
    const isFeatureCount = fractionalChartData.percentages.some(p => p > 100);
    const chartData = fractionalChartData.percentages.map((pct, idx) => {
      const point: Record<string, number | string> = {
        percentage: isFeatureCount ? `${pct}` : `${pct}%`
      };
      fractionalChartData.series.forEach((s) => {
        point[s.name] = s.values[idx];
      });
      return point;
    });

    return (
      <Paper sx={{ p: 2, height: 350 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Fractional Ablation Dose-Response
          </Typography>
          {(currentModel === 'pi05') && (
            <MuiTooltip title="Data from preliminary experiments. Full-scale experiments in progress.">
              <Chip label="Preliminary" size="small" sx={{ height: 18, fontSize: '9px', bgcolor: '#fef3c7', color: '#92400e' }} />
            </MuiTooltip>
          )}
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
          {isFeatureCount
            ? `Success rate (%) when ablating increasing numbers of SAE features per layer. Shows dose-response for ${VLA_MODELS[currentModel]?.name ?? currentModel}.`
            : 'Success rate (%) when ablating varying percentages of concept features. Shows how many features need to be ablated to impact task performance.'
          }
        </Typography>
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="percentage" tick={{ fontSize: 10 }} label={{ value: isFeatureCount ? "# Features Ablated" : "Features Ablated (%)", position: "insideBottom", offset: -5, fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} label={{ value: "Success %", angle: -90, position: "insideLeft", fontSize: 10 }} />
            <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${v.toFixed(1)}%` : `${v ?? '-'}`, ""]} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <ReferenceLine y={100} stroke="#22c55e" strokeDasharray="5 5" label={{ value: "Perfect", position: "right", fontSize: 9 }} />
            <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="5 5" label={{ value: "50%", position: "right", fontSize: 9 }} />
            {fractionalChartData.series.map((s, idx) => (
              <Line
                key={s.name}
                type="monotone"
                dataKey={s.name}
                stroke={CONCEPT_COLORS[s.name] || LAYER_COLORS[idx % LAYER_COLORS.length]}
                strokeWidth={2}
                dot={{ r: 5, fill: CONCEPT_COLORS[s.name] || LAYER_COLORS[idx % LAYER_COLORS.length] }}
                name={s.name.charAt(0).toUpperCase() + s.name.slice(1)}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    );
  };

  // Render summary insights
  const renderSummaryInsights = () => {
    if (currentModel !== 'pi05') return null;
    const summary = comprehensiveData?.summary;
    if (!summary) return null;

    const actionPhases = summary?.experiments?.action_phases;
    if (!actionPhases) return null;
    const { most_impactful_negative, most_impactful_positive } = actionPhases;

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          Key Findings from Grid Analysis
        </Typography>
        <Grid container spacing={2}>
          {most_impactful_negative && (
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: "#fef2f2", borderRadius: 1, border: "1px solid #fecaca" }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: "#dc2626", mb: 0.5 }}>
                  Most Impactful Negative
                </Typography>
                <Typography variant="body2">
                  Layer {most_impactful_negative.layer} during <strong>{most_impactful_negative.phase}</strong> phase
                </Typography>
                <Typography variant="h6" sx={{ color: "#dc2626", fontWeight: 700 }}>
                  {most_impactful_negative.delta}% success rate change
                </Typography>
              </Box>
            </Grid>
          )}
          {most_impactful_positive && (
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, bgcolor: "#f0fdf4", borderRadius: 1, border: "1px solid #bbf7d0" }}>
                <Typography variant="body2" sx={{ fontWeight: 600, color: "#16a34a", mb: 0.5 }}>
                  Most Impactful Positive
                </Typography>
                <Typography variant="body2">
                  Layer {most_impactful_positive.layer} during <strong>{most_impactful_positive.phase}</strong> phase
                </Typography>
                <Typography variant="h6" sx={{ color: "#16a34a", fontWeight: 700 }}>
                  +{most_impactful_positive.delta}% success rate change
                </Typography>
              </Box>
            </Grid>
          )}
        </Grid>
      </Paper>
    );
  };

  // Render Overview Statistics
  const renderOverviewStats = () => {
    if (videosLoading && !modelExperimentResults) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress size={24} />
        </Box>
      );
    }

    // Always use modelExperimentResults as primary display for consistency across all models
    if (modelExperimentResults) {
      const modelName = modelExperimentResults.model_name || VLA_MODELS[currentModel]?.name || currentModel;
      const totalLabel = modelExperimentResults.total_episodes_label || '';
      const totalEpisodes = modelExperimentResults.total_episodes || 0;
      const sectionEntries = modelExperimentResults.sections || {};
      const sectionNames = Object.keys(sectionEntries).filter(k =>
        !['summary', 'displacement', 'concept_id', 'scene_state', 'libero_experiments', 'metaworld_experiments'].includes(k)
      );
      // Baseline success rates from full_data
      const fullData = modelExperimentResults.full_data || modelExperimentResults;
      const baselines = fullData?.baselines || {};
      const baselineChips = Object.entries(baselines).map(([suite, info]: [string, any]) => {
        if (typeof info !== 'object' || info === null) return null;
        const sr = info.overall_success_rate ?? info.success_rate ?? info.mean_success_rate ?? info.rate ?? null;
        const pct = sr != null ? `${Math.round(Number(sr) * (Number(sr) <= 1 ? 100 : 1))}%` : 'N/A';
        return { suite, pct };
      }).filter(Boolean);

      // Video stats (supplementary, if available)
      const hasVideoStats = aggregatedStats && aggregatedStats.totalVideos > 0;
      const knownCount = hasVideoStats ? (aggregatedStats!.successCount + aggregatedStats!.failureCount) : 0;
      const overallSuccessRate = knownCount > 0 && aggregatedStats
        ? Math.round((aggregatedStats.successCount / knownCount) * 100)
        : null;

      return (
        <Grid container spacing={1.5}>
          {/* Summary Cards */}
          <Grid item xs={6} sm={4}>
            <Paper sx={{ p: 1.5, textAlign: 'center', bgcolor: '#f0f9ff', border: '1px solid #bae6fd', overflow: 'hidden' }}>
              <Typography variant="h5" sx={{ fontWeight: 700, color: '#0284c7', fontSize: { xs: '1.2rem', md: '1.5rem' }, lineHeight: 1.2, wordBreak: 'break-word' }}>
                {totalLabel || totalEpisodes.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Total Episodes
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={4}>
            <Paper sx={{ p: 1.5, textAlign: 'center', bgcolor: '#f0fdf4', border: '1px solid #bbf7d0', overflow: 'hidden' }}>
              <Typography variant="h5" sx={{ fontWeight: 700, color: '#16a34a', fontSize: { xs: '1.2rem', md: '1.5rem' }, lineHeight: 1.2 }}>
                {sectionNames.length}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Experiment Types
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={4}>
            <Paper sx={{ p: 1.5, textAlign: 'center', bgcolor: '#fffbeb', border: '1px solid #fde68a', overflow: 'hidden' }}>
              <Typography variant="h5" sx={{ fontWeight: 700, color: '#d97706', fontSize: { xs: '1.2rem', md: '1.5rem' }, lineHeight: 1.2 }}>
                {(() => {
                  // Use overall_success_rate if available, else compute from sections
                  const overall = modelExperimentResults.overall_success_rate;
                  if (overall != null) {
                    const sr = Number(overall);
                    return `${Math.round(sr <= 1 ? sr * 100 : sr)}%`;
                  }
                  // Fallback to baseline SR
                  const blSection = sectionEntries.baselines;
                  if (blSection?.success_rate != null) {
                    const sr = Number(blSection.success_rate);
                    return `${Math.round(sr <= 1 ? sr * 100 : sr)}%`;
                  }
                  return 'N/A';
                })()}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Overall Success Rate
              </Typography>
            </Paper>
          </Grid>

          {/* Experiment breakdown with success/failure/unknown stacked bars */}
          <Grid item xs={12}>
            <Paper sx={{ p: 1.5 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>Experiment Breakdown</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.8 }}>
                {sectionNames.map((name: string) => {
                  const sec = sectionEntries[name];
                  const eps = sec?.episodes || 0;
                  const sr = sec?.success_rate;
                  // Try to get success/failure/unknown from video data
                  const videoEntry = experimentTypeChartData.find(e =>
                    e.name === name.replace(/_/g, ' ') ||
                    e.name === name ||
                    name.includes(e.name.replace(/ /g, '_'))
                  );
                  const total = videoEntry?.total || eps;
                  const success = videoEntry?.success || (sr != null ? Math.round(Number(sr) * eps) : 0);
                  const failure = videoEntry?.failure || (sr != null ? eps - Math.round(Number(sr) * eps) : 0);
                  const unknown = videoEntry?.unknown || (sr == null && !videoEntry ? eps : 0);
                  const hasBreakdown = success > 0 || failure > 0;
                  const isCrossTask = name === 'cross_task' || name === 'cross-task';
                  return (
                    <Box key={name}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Typography variant="body2" sx={{ fontWeight: 500, textTransform: 'capitalize', fontSize: 12 }}>
                            {name.replace(/_/g, ' ')}
                          </Typography>
                          {isCrossTask && (
                            <MuiTooltip title="Success rate measures source task behavior transfer rate, not binary task completion." arrow>
                              <Chip label="transfer rate" size="small" sx={{ height: 14, fontSize: '8px', bgcolor: '#e0f2fe', color: '#0369a1' }} />
                            </MuiTooltip>
                          )}
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                          {eps > 0 ? `${eps.toLocaleString()} eps` : 'available'}
                          {hasBreakdown && total > 0 ? ` · ${success}/${success + failure} succeeded` : ''}
                          {sr != null && !hasBreakdown ? ` (${Math.round(Number(sr) * 100)}% SR)` : ''}
                        </Typography>
                      </Box>
                      {eps > 0 && (
                        <Box sx={{ display: 'flex', height: 8, borderRadius: 1, overflow: 'hidden', bgcolor: '#e5e7eb' }}>
                          {hasBreakdown ? (
                            <>
                              <Box sx={{ width: `${(success / total) * 100}%`, bgcolor: '#22c55e' }} />
                              <Box sx={{ width: `${(failure / total) * 100}%`, bgcolor: '#ef4444' }} />
                              {unknown > 0 && <Box sx={{ width: `${(unknown / total) * 100}%`, bgcolor: '#fbbf24' }} />}
                            </>
                          ) : (
                            <Box sx={{ width: '100%', bgcolor: '#fbbf24' }} />
                          )}
                        </Box>
                      )}
                    </Box>
                  );
                })}
              </Box>
              <Box sx={{ display: 'flex', gap: 2, mt: 1.5, justifyContent: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 10, height: 10, bgcolor: '#22c55e', borderRadius: 0.5 }} />
                  <Typography variant="caption" sx={{ fontSize: 10 }}>Success</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 10, height: 10, bgcolor: '#ef4444', borderRadius: 0.5 }} />
                  <Typography variant="caption" sx={{ fontSize: 10 }}>Failure</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 10, height: 10, bgcolor: '#fbbf24', borderRadius: 0.5 }} />
                  <Typography variant="caption" sx={{ fontSize: 10 }}>Not Tracked</Typography>
                </Box>
              </Box>
            </Paper>
          </Grid>

          {/* Vision Perturbation Subtypes Breakdown */}
          {subtypeChartData.length > 1 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Vision Perturbation Subtypes
                </Typography>
                <Grid container spacing={2}>
                  {subtypeChartData.map((subtype) => (
                    <Grid item xs={6} sm={4} md={2} key={subtype.name}>
                      <Box
                        sx={{
                          p: 1.5,
                          borderRadius: 1,
                          border: '1px solid #e5e7eb',
                          bgcolor: subtype.name === 'baseline' ? '#f0f9ff' : '#fff',
                          textAlign: 'center',
                        }}
                      >
                        <Typography variant="h5" sx={{ fontWeight: 700, color: '#1f2937' }}>
                          {subtype.total.toLocaleString()}
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, textTransform: 'capitalize', color: '#374151' }}>
                          {subtype.name}
                        </Typography>
                        {subtype.successRate !== null ? (
                          <Chip
                            label={`${subtype.successRate}%`}
                            size="small"
                            sx={{
                              mt: 0.5,
                              height: 18,
                              fontSize: '10px',
                              bgcolor: subtype.successRate >= 80 ? '#dcfce7' : subtype.successRate >= 50 ? '#fef3c7' : '#fee2e2',
                              color: subtype.successRate >= 80 ? '#166534' : subtype.successRate >= 50 ? '#92400e' : '#991b1b',
                            }}
                          />
                        ) : (
                          <Chip
                            label="No data"
                            size="small"
                            sx={{ mt: 0.5, height: 18, fontSize: '10px', bgcolor: '#f5f5f5', color: '#9e9e9e' }}
                          />
                        )}
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
          )}
        </Grid>
      );
    }

    // Fallback: no experiment results available at all
    return (
      <Alert severity="info">
        No experiment data found for {VLA_MODELS[currentModel]?.name ?? currentModel}.
        {videosLoading && ' Loading...'}
      </Alert>
    );
  };

  // Render ablation summary table for SmolVLA/X-VLA/GR00T
  const renderAblationSummaryTable = () => {
    if (!ablationSummaryData || ablationSummaryData.length === 0) return null;

    // Group by suite
    const bySuite: Record<string, any[]> = {};
    ablationSummaryData.forEach((entry: any) => {
      const suite = entry.suite || 'unknown';
      if (!bySuite[suite]) bySuite[suite] = [];
      bySuite[suite].push(entry);
    });

    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
          Ablation Experiment Index — {VLA_MODELS[currentModel]?.name ?? currentModel}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {ablationSummaryData.length} ablation experiment files across {Object.keys(bySuite).length} suite(s).
          {currentModel === 'smolvla' && ' Dual-pathway Expert ablation with per-concept results.'}
          {currentModel === 'xvla' && ' Single-pathway ablation across 24 layers.'}
          {currentModel === 'groot' && ' Triple-pathway ablation across DiT, Eagle, and VL-SA components.'}
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(bySuite).sort(([a], [b]) => a.localeCompare(b)).map(([suite, entries]) => {
            const layers = [...new Set(entries.map((e: any) => e.layer))].sort((a, b) => a - b);
            const avgBaseline = entries.reduce((sum: number, e: any) => sum + (e.baseline_overall || 0), 0) / entries.length;
            return (
              <Grid item xs={12} sm={6} md={4} key={suite}>
                <Box sx={{ p: 2, borderRadius: 1, border: '1px solid #e5e7eb' }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, textTransform: 'capitalize', mb: 1 }}>
                    {suite.replace(/_/g, ' ')}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    <Chip label={`${entries.length} files`} size="small" color="primary" sx={{ height: 20, fontSize: '10px' }} />
                    <Chip label={`${layers.length} layers`} size="small" variant="outlined" sx={{ height: 20, fontSize: '10px' }} />
                    {avgBaseline > 0 && (
                      <Chip
                        label={`Baseline: ${(avgBaseline * 100).toFixed(1)}%`}
                        size="small"
                        sx={{ height: 20, fontSize: '10px', bgcolor: avgBaseline >= 0.8 ? '#dcfce7' : '#fef3c7', color: avgBaseline >= 0.8 ? '#166534' : '#92400e' }}
                      />
                    )}
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    Layers: {layers.length <= 8 ? layers.join(', ') : `${layers[0]}-${layers[layers.length - 1]}`}
                  </Typography>
                </Box>
              </Grid>
            );
          })}
        </Grid>
      </Paper>
    );
  };

  return (
    <Box sx={{ p: 2, height: "100%", overflow: "auto" }}>
      {/* Header with Model Badge */}
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Ablation Study Visualizations
          </Typography>
          <Chip
            label={VLA_MODELS[currentModel]?.name ?? currentModel}
            size="small"
            sx={{
              fontWeight: 600,
              bgcolor: {
                pi05: '#3b82f6',
                openvla: '#8b5cf6',
                xvla: '#f59e0b',
                smolvla: '#10b981',
                groot: '#ef4444',
                act: '#6366f1',
              }[currentModel] || '#8b5cf6',
              color: 'white',
            }}
          />
        </Box>
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <Select
            value={activeConcept}
            onChange={(e) => handleConceptChange(e.target.value)}
            sx={{ fontSize: 13 }}
            displayEmpty
          >
            {allConcepts.map((c) => {
              const parts = c.split('/');
              const category = parts.length > 1 ? parts[0] : '';
              const name = parts.length > 1 ? parts[1] : c;
              return (
                <MenuItem key={c} value={c}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Box sx={{ width: 12, height: 12, borderRadius: "50%", bgcolor: CONCEPT_COLORS[c] || "#888" }} />
                    <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 45 }}>
                      {category}
                    </Typography>
                    {name.replace(/_/g, ' ').toUpperCase()}
                  </Box>
                </MenuItem>
              );
            })}
          </Select>
        </FormControl>
      </Box>

      {/* Main Tabs - disable tabs without data for current model */}
      <Tabs
        value={mainTab}
        onChange={(_, v) => setMainTab(v)}
        sx={{ mb: 2, borderBottom: 1, borderColor: "divider" }}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label="Overview" sx={{ textTransform: "none", fontWeight: 600 }} />
        {(!!gridHeatmapData || (currentModel === 'pi05' && !!comprehensiveData?.action_phase_heatmap)) ? (
          <Tab label="Layer-Phase Matrix" sx={{ textTransform: "none", fontWeight: 600 }} />
        ) : (
          <Tab label="Layer-Phase Matrix" disabled sx={{ textTransform: "none", fontWeight: 600, opacity: 0.4 }} />
        )}
        {['groot', 'smolvla'].includes(currentModel) ? (
          <Tab label="Temporal Ablations" sx={{ textTransform: "none", fontWeight: 600 }} />
        ) : (
          <Tab label="Temporal Ablations" disabled sx={{ textTransform: "none", fontWeight: 600, opacity: 0.4 }} />
        )}
        {['groot', 'smolvla'].includes(currentModel) ? (
          <Tab label="Fractional Ablations" sx={{ textTransform: "none", fontWeight: 600 }} />
        ) : (
          <Tab label="Fractional Ablations" disabled sx={{ textTransform: "none", fontWeight: 600, opacity: 0.4 }} />
        )}
        <Tab label="Concept Steering" sx={{ textTransform: "none", fontWeight: 600 }} />
      </Tabs>

      {/* Tab 0: Overview with Real Statistics */}
      {mainTab === 0 && (
        <Grid container spacing={2}>
          {/* Real Statistics from API */}
          <Grid item xs={12}>
            {renderOverviewStats()}
          </Grid>

        {/* Row 1: Fraction-to-Failure + Temporal (only for models with data) */}
        {fractionData.length > 0 && ['groot', 'smolvla'].includes(currentModel) && (
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Fraction-to-Failure: {activeConcept.toUpperCase()}
              </Typography>
              <MuiTooltip title="Data from preliminary experiments (sample size n=3 per condition). Full-scale experiments in progress.">
                <Chip label="Preliminary" size="small" sx={{ height: 18, fontSize: '9px', bgcolor: '#fef3c7', color: '#92400e' }} />
              </MuiTooltip>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
              How many features cause task failure?
            </Typography>
            <ResponsiveContainer width="100%" height={210}>
              <LineChart data={fractionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="features" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${v}%` : `${v ?? '-'}`, "Success"]} />
                <ReferenceLine y={50} stroke="#999" strokeDasharray="5 5" />
                <Line
                  type="monotone"
                  dataKey="rate"
                  stroke={CONCEPT_COLORS[activeConcept] || "#888"}
                  strokeWidth={3}
                  dot={{ r: 5, fill: CONCEPT_COLORS[activeConcept] || "#888" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        )}

        {temporalData.length > 0 && ['groot', 'smolvla'].includes(currentModel) && (
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Temporal Ablation: {activeConcept.toUpperCase()}
              </Typography>
              <MuiTooltip title="Data from preliminary experiments (sample size n=3 per condition). Full-scale experiments in progress.">
                <Chip label="Preliminary" size="small" sx={{ height: 18, fontSize: '9px', bgcolor: '#fef3c7', color: '#92400e' }} />
              </MuiTooltip>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
              When does ablation matter during the episode?
            </Typography>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={temporalData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} />
                <YAxis dataKey="window" type="category" width={80} tick={{ fontSize: 9 }} />
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${Math.round(v)}%` : `${v ?? '-'}`, "Success"]} />
                <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                  {temporalData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.rate === 0 ? "#ef4444" : entry.rate < 50 ? "#f59e0b" : "#22c55e"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        )}

        {/* Row 2: Heatmap (only for models with temporal data) */}
        {temporalData.length > 0 && ['groot', 'smolvla'].includes(currentModel) && (
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 320 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
              Temporal Ablation Heatmap (All Concepts)
            </Typography>
            <Box sx={{ height: 260, display: "flex", justifyContent: "center", alignItems: "center", overflow: "hidden" }}>
              <img
                src="/temporal_ablation_heatmap.png"
                alt="Temporal Ablation Heatmap"
                style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
              />
            </Box>
          </Paper>
        </Grid>
        )}

        {/* Row 3: Overview Video Grid from API */}
        <Grid item xs={12}>
          {(() => {
            const ablationVids = videos.filter(v =>
              v.experiment_type === 'grid_ablation' || v.experiment_type === 'ablation' || v.experiment_type === 'concept_ablation' || v.experiment_type === 'fraction_to_failure'
            );
            if (!videosLoading && ablationVids.length > 0) {
              return (
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                    Sample Ablation Videos ({VLA_MODELS[currentModel]?.name ?? currentModel}) — {ablationVids.length} total
                  </Typography>
                  <Grid container spacing={2}>
                    {ablationVids.slice(0, VIDEOS_PER_PAGE).map((video, index) => {
                      const isSuccess = parseVideoSuccess(video);
                      return (
                        <Grid item xs={12} sm={6} md={3} key={video.path || index}>
                          <Card variant="outlined" sx={{ transition: 'all 0.2s', '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.1)', borderColor: '#ef4444' } }}>
                            <CardContent sx={{ p: 1.5 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 1 }}>
                                <PlayCircleOutlineIcon fontSize="small" sx={{ color: isSuccess === true ? '#22c55e' : isSuccess === false ? '#ef4444' : '#9e9e9e', fontSize: 16 }} />
                                <Typography variant="caption" sx={{ fontWeight: 600, flex: 1, fontSize: 10 }}>
                                  {video.concept || video.experiment_type?.replace(/_/g, ' ') || 'Ablation'}
                                </Typography>
                                {video.layer !== undefined && (
                                  <Chip label={`L${video.layer}`} size="small" sx={{ height: 14, fontSize: '8px' }} />
                                )}
                              </Box>
                              <Box sx={{ display: 'flex', gap: 0.5, mb: 1, flexWrap: 'wrap' }}>
                                <Chip
                                  label={video.suite?.replace(/_/g, ' ') || 'N/A'}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 14, fontSize: '8px', textTransform: 'capitalize' }}
                                />
                                {video.task !== undefined && (
                                  <Chip label={`T${video.task}`} size="small" variant="outlined" sx={{ height: 14, fontSize: '8px' }} />
                                )}
                                {isSuccess !== null && (
                                  <Chip
                                    label={isSuccess ? 'OK' : 'FAIL'}
                                    size="small"
                                    sx={{ height: 14, fontSize: '8px', bgcolor: isSuccess ? '#dcfce7' : '#fee2e2', color: isSuccess ? '#166534' : '#991b1b' }}
                                  />
                                )}
                              </Box>
                              <Box sx={{ width: '100%', aspectRatio: '4/3', bgcolor: '#0a1628', borderRadius: 1, overflow: 'hidden' }}>
                                <video
                                  src={getVideoUrl(video.path)}
                                  controls muted
                                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                  onError={(e) => { const el = e.target as HTMLVideoElement; el.style.display = 'none'; const parent = el.parentElement; if (parent) { const msg = document.createElement('div'); msg.style.cssText = 'display:flex;align-items:center;justify-content:center;width:100%;height:100%;color:#64748b;font-size:12px;text-align:center;padding:8px'; msg.textContent = 'Video unavailable'; parent.appendChild(msg); } }}
                                />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Paper>
              );
            }
            if (!videosLoading) {
              return (
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Typography color="text.secondary" variant="body2">
                    No ablation videos found for {VLA_MODELS[currentModel]?.name ?? currentModel}.
                    Ablation experiment videos will appear here once generated.
                  </Typography>
                </Paper>
              );
            }
            return null;
          })()}
        </Grid>

        {/* Row 4.5: Ablation Summary Table for SmolVLA/X-VLA/GR00T */}
        {ablationSummaryData.length > 0 && (
          <Grid item xs={12}>
            {renderAblationSummaryTable()}
          </Grid>
        )}

        {/* Row 5: Data-driven summary chips */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
              Experiment Summary — {VLA_MODELS[currentModel]?.name ?? currentModel}
            </Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
              {ablationSummaryData.length > 0 && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="ABLATION" size="small" color="error" />
                  <Typography variant="body2">{ablationSummaryData.length} experiment file{ablationSummaryData.length !== 1 ? 's' : ''} loaded</Typography>
                </Box>
              )}
              {videos.length > 0 && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip label="VIDEOS" size="small" color="info" />
                  <Typography variant="body2">{videos.length.toLocaleString()} videos indexed</Typography>
                </Box>
              )}
              {(() => {
                const sections = modelExperimentResults?.sections || {};
                const sectionNames = Object.keys(sections).filter(k =>
                  !['summary', 'displacement', 'concept_id', 'scene_state', 'libero_experiments', 'metaworld_experiments'].includes(k)
                );
                return sectionNames.length > 0 ? (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="TYPES" size="small" color="success" />
                    <Typography variant="body2">{sectionNames.map(n => n.replace(/_/g, ' ')).join(', ')}</Typography>
                  </Box>
                ) : null;
              })()}
            </Box>
          </Paper>
        </Grid>

        {/* Overview Videos Section */}
        <Grid item xs={12}>
          {renderVideoGrid(videos.slice(0, 50), overviewVideoPage, setOverviewVideoPage, 'Recent Experiment Videos')}
        </Grid>
        </Grid>
      )}

      {/* Tab 1: Layer-Phase Matrix - Layer x Phase combinations */}
      {mainTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            {renderGridHeatmap()}
          </Grid>
          <Grid item xs={12}>
            {renderSummaryInsights()}
          </Grid>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                About Grid Ablations
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {currentModel === 'pi05' ? (
                  'Grid ablation experiments systematically test the effect of ablating features at different layers during different action phases. Each cell shows the change in success rate (%) compared to baseline.'
                ) : gridHeatmapData ? (
                  `Grid ablation results for ${VLA_MODELS[currentModel]?.name ?? currentModel}. Each cell shows the success rate delta (pp) when ablating all SAE features at a given layer, compared to baseline. Columns represent LIBERO suites.`
                ) : (
                  `Grid ablation data for ${VLA_MODELS[currentModel]?.name ?? currentModel}. Layer-phase matrix is available for Pi0.5. For other models, grid ablation results are shown as layer-by-suite success rate deltas.`
                )}
              </Typography>
              <Box sx={{ mt: 2, display: "flex", gap: 2, flexWrap: "wrap" }}>
                {currentModel === 'pi05' ? (<>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="Layers" size="small" variant="outlined" />
                    <Typography variant="body2">0, 5, 10, 12, 15, 17 tested</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="Phases" size="small" variant="outlined" />
                    <Typography variant="body2">approach, grasp, lift, transport, lower, release</Typography>
                  </Box>
                </>) : gridHeatmapData ? (<>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="Layers" size="small" variant="outlined" />
                    <Typography variant="body2">{gridHeatmapData.layers.length} layers tested</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="Suites" size="small" variant="outlined" />
                    <Typography variant="body2">{gridHeatmapData.phases.join(', ')}</Typography>
                  </Box>
                </>) : null}
              </Box>
            </Paper>
          </Grid>
          {/* Grid / Concept Ablation Videos */}
          <Grid item xs={12}>
            {renderVideoGrid(
              [...gridVideos, ...conceptAblationVideos],
              gridVideoPage,
              setGridVideoPage,
              gridVideos.length > 0 ? 'Grid Ablation Videos' : 'Concept Ablation Videos'
            )}
          </Grid>
        </Grid>
      )}

      {/* Tab 2: Temporal Ablations - When ablation matters */}
      {mainTab === 2 && (
        <Grid container spacing={2}>
          {/* Pi0.5: Use legacy experiment_summary temporal data */}
          {currentModel === 'pi05' && (
            <>
              <Grid item xs={12} md={6}>
                {renderTemporalTimestepChart()}
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2, height: 350 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                    Temporal Ablation: {activeConcept.toUpperCase()}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                    Success rate by ablation window for selected concept
                  </Typography>
                  <ResponsiveContainer width="100%" height={270}>
                    <BarChart data={temporalData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} />
                      <YAxis dataKey="window" type="category" width={80} tick={{ fontSize: 9 }} />
                      <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${Math.round(v)}%` : `${v ?? '-'}`, "Success"]} />
                      <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                        {temporalData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.rate === 0 ? "#ef4444" : entry.rate < 50 ? "#f59e0b" : "#22c55e"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </>
          )}

          {/* GR00T/SmolVLA: Use temporal API data or experiment_results temporal_ablation */}
          {(['groot', 'smolvla'].includes(currentModel)) && (() => {
            // Get temporal data from API or from full_data
            const temporalApiDataVal = modelExperimentResults?._temporalApiData;
            const fullData = modelExperimentResults?.full_data || modelExperimentResults;
            const temporalSection = fullData?.temporal_ablation;
            // Prefer API data which has per-layer windows
            const hasApiData = temporalApiDataVal && temporalApiDataVal.suites;
            // Also check if temporalSection has per_layer data (GR00T full_data)
            const hasFullDataTemporal = temporalSection && typeof temporalSection === 'object' &&
              Object.values(temporalSection).some((v: any) => typeof v === 'object' && v?.per_layer);

            if (hasApiData || hasFullDataTemporal) {
              // Build bar chart data from windows across layers
              const sourceData = hasApiData ? temporalApiDataVal.suites : temporalSection;
              const firstSuiteKey = Object.keys(sourceData).find((k: string) => {
                const v = sourceData[k];
                return typeof v === 'object' && v?.per_layer;
              });
              const suiteData = firstSuiteKey ? sourceData[firstSuiteKey] : null;
              const perLayer = suiteData?.per_layer || {};
              const layerKeys = Object.keys(perLayer).sort();

              // Aggregate windows across all layers
              const windowAgg: Record<string, { srSum: number; count: number }> = {};
              let baselineSum = 0;
              let baselineCount = 0;
              layerKeys.forEach((lk: string) => {
                const layerData = perLayer[lk];
                const source = layerData?.frequent || layerData?.universal || layerData;
                const bl = source?.baseline ?? layerData?.baseline ?? 1.0;
                if (typeof bl === 'number') { baselineSum += bl; baselineCount++; }
                const windows = source?.windows;
                if (windows && typeof windows === 'object') {
                  Object.entries(windows).forEach(([wName, wData]: [string, any]) => {
                    if (!windowAgg[wName]) windowAgg[wName] = { srSum: 0, count: 0 };
                    const sr = wData?.success_rate ?? wData?.mean_success_rate;
                    if (typeof sr === 'number') {
                      windowAgg[wName].srSum += sr;
                      windowAgg[wName].count += 1;
                    }
                  });
                }
              });

              const baselineAvg = baselineCount > 0 ? baselineSum / baselineCount : 1.0;
              const windowChartData = [
                { window: 'Baseline', rate: Math.round(baselineAvg * 100), episodes: baselineCount },
                ...Object.entries(windowAgg).map(([name, agg]) => ({
                  window: name.replace(/_/g, ' '),
                  rate: agg.count > 0 ? Math.round((agg.srSum / agg.count) * 100) : 0,
                  episodes: agg.count,
                })),
              ];

              // Build per-layer heatmap: layers x windows
              const windowNames = Object.keys(windowAgg);
              const heatmapLayers = layerKeys.slice(0, 16); // Cap at 16 for display

              return (
                <>
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, height: 350 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                        Temporal Ablation by Window (Aggregated)
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                        Success rate across {layerKeys.length} layers, aggregated by temporal window.
                        {firstSuiteKey && ` Suite: ${firstSuiteKey.replace(/_/g, ' ')}`}
                      </Typography>
                      <ResponsiveContainer width="100%" height={270}>
                        <BarChart data={windowChartData} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} />
                          <YAxis dataKey="window" type="category" width={90} tick={{ fontSize: 9 }} />
                          <Tooltip
                            contentStyle={{ fontSize: 11, borderRadius: 8 }}
                            formatter={(v: unknown, name: unknown, props: any) => [
                              typeof v === 'number' ? `${v}% (${props?.payload?.episodes || 0} eps)` : `${v ?? '-'}`,
                              "Success"
                            ]}
                          />
                          <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                            {windowChartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.rate === 0 ? "#ef4444" : entry.rate < 50 ? "#f59e0b" : "#22c55e"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2, height: 350, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                        Layer x Window Heatmap
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                        Success rate (%) for each layer-window combination. Top {heatmapLayers.length} layers shown.
                      </Typography>
                      <Box sx={{ overflow: 'auto', flex: 1 }}>
                        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.3, width: 'fit-content' }}>
                          <Box sx={{ display: "flex", gap: 0.3, pl: 7 }}>
                            {windowNames.map((w) => (
                              <Box key={w} sx={{ width: 55, minWidth: 55, flexShrink: 0, textAlign: "center", fontSize: 8, fontWeight: 500 }}>
                                {w.replace(/_/g, ' ')}
                              </Box>
                            ))}
                          </Box>
                          {heatmapLayers.map((lk) => {
                            const layerData = perLayer[lk];
                            const source = layerData?.frequent || layerData?.universal || layerData;
                            const windows = source?.windows || {};
                            return (
                              <Box key={lk} sx={{ display: "flex", alignItems: "center", gap: 0.3, width: 'fit-content' }}>
                                <Box sx={{ width: 60, minWidth: 60, flexShrink: 0, fontSize: 9, fontWeight: 500, textAlign: "right", pr: 0.5 }}>
                                  {lk.replace(/_/g, ' ')}
                                </Box>
                                {windowNames.map((wName) => {
                                  const wData = windows[wName];
                                  const rawSr = wData?.success_rate ?? wData?.mean_success_rate ?? (wData?.total ? wData.successes / wData.total : null);
                                  const sr = rawSr !== null && rawSr !== undefined ? Math.round(Number(rawSr) * (Number(rawSr) <= 1 ? 100 : 1)) : null;
                                  const blLayer = (source?.baseline ?? layerData?.baseline ?? 1.0);
                                  const blPct = typeof blLayer === 'number' ? Math.round(blLayer * 100) : 100;
                                  const delta = sr !== null ? sr - blPct : null;
                                  return (
                                    <MuiTooltip key={`${lk}-${wName}`} title={sr !== null ? `${lk}, ${wName}: ${sr}%` : 'No data'} arrow>
                                      <Box sx={{
                                        width: 55, minWidth: 55, flexShrink: 0, height: 22,
                                        bgcolor: getHeatmapColor(delta),
                                        borderRadius: 0.3,
                                        display: "flex", alignItems: "center", justifyContent: "center",
                                        fontSize: 8, fontWeight: 600,
                                        color: sr !== null && sr > 40 && sr < 70 ? "#1f2937" : "#fff",
                                      }}>
                                        {sr !== null ? `${sr}%` : '-'}
                                      </Box>
                                    </MuiTooltip>
                                  );
                                })}
                              </Box>
                            );
                          })}
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                </>
              );
            }

            // SmolVLA/GR00T with only summary counts (no per_layer)
            if (temporalSection && (temporalSection.episodes || temporalSection.count)) {
              return (
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                      Temporal Ablation Data
                    </Typography>
                    <Alert severity="info" sx={{ mb: 2 }}>
                      {VLA_MODELS[currentModel]?.name ?? currentModel} has {temporalSection.episodes?.toLocaleString() || temporalSection.count} temporal ablation episodes.
                      {temporalSection.description && ` ${temporalSection.description}`}
                    </Alert>
                  </Paper>
                </Grid>
              );
            }

            return (
              <Grid item xs={12}>
                <Alert severity="info">
                  No temporal ablation data available for {VLA_MODELS[currentModel]?.name ?? currentModel}.
                </Alert>
              </Grid>
            );
          })()}

          {/* OFT/X-VLA/ACT: No temporal ablation data */}
          {(['openvla', 'xvla', 'act'].includes(currentModel)) && (
            <Grid item xs={12}>
              <Alert severity="info">
                No temporal ablation data available for {VLA_MODELS[currentModel]?.name ?? currentModel}.
              </Alert>
            </Grid>
          )}

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                About Temporal Ablations
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Temporal ablation experiments test <strong>when</strong> feature ablation matters during task execution.
                By ablating features only during specific timesteps or windows, we can identify critical moments
                where concept representations are essential for task success.
              </Typography>
            </Paper>
          </Grid>
          {/* Temporal Ablation Videos */}
          <Grid item xs={12}>
            {renderVideoGrid(temporalVideos, temporalVideoPage, setTemporalVideoPage, 'Temporal Ablation Videos')}
          </Grid>
        </Grid>
      )}

      {/* Tab 3: Fractional Ablations - Dose-response curves */}
      {mainTab === 3 && (
        <Grid container spacing={2}>
          {/* Dose-response chart: from comprehensiveData or experiment_results FTF */}
          <Grid item xs={12} md={6}>
            {fractionalChartData ? renderFractionalDoseResponse() : (
              <Paper sx={{ p: 3, height: 350, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Fractional Ablation
                </Typography>
                <Alert severity="info" sx={{ maxWidth: 400 }}>
                  No fraction-to-failure titration data available for {VLA_MODELS[currentModel]?.name ?? currentModel}.
                </Alert>
              </Paper>
            )}
          </Grid>
          {/* Per-concept FTF or chart summary */}
          <Grid item xs={12} md={6}>
            {fractionData.length > 0 ? (
              <Paper sx={{ p: 2, height: 350 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Fraction-to-Failure: {activeConcept.toUpperCase()}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                  How many features need to be ablated to cause task failure?
                </Typography>
                <ResponsiveContainer width="100%" height={270}>
                  <LineChart data={fractionData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="features" tick={{ fontSize: 10 }} label={{ value: "# Features Ablated", position: "insideBottom", offset: -5, fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} label={{ value: "Success %", angle: -90, position: "insideLeft", fontSize: 10 }} />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [typeof v === 'number' ? `${v}%` : `${v ?? '-'}`, "Success"]} />
                    <ReferenceLine y={50} stroke="#999" strokeDasharray="5 5" label={{ value: "50%", position: "right", fontSize: 9 }} />
                    <Line
                      type="monotone"
                      dataKey="rate"
                      stroke={CONCEPT_COLORS[activeConcept] || "#888"}
                      strokeWidth={3}
                      dot={{ r: 5, fill: CONCEPT_COLORS[activeConcept] || "#888" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Paper>
            ) : fractionalChartData ? (
              <Paper sx={{ p: 2, height: 350 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Titration Summary by Layer
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                  Minimum success rate observed per layer across {fractionalChartData.percentages.length} feature counts
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 2, maxHeight: 260, overflow: 'auto' }}>
                  {fractionalChartData.series.map((s, idx) => {
                    const minVal = Math.min(...s.values.filter(v => v != null));
                    const delta = minVal - s.baseline;
                    return (
                      <Box key={s.name} sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, borderRadius: 1, border: '1px solid #e5e7eb' }}>
                        <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: LAYER_COLORS[idx % LAYER_COLORS.length], flexShrink: 0 }} />
                        <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 60 }}>{s.name}</Typography>
                        <Box sx={{ flex: 1, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                          <Chip label={`baseline: ${s.baseline}%`} size="small" sx={{ height: 20, fontSize: '10px' }} />
                          <Chip
                            label={`min: ${minVal}%`}
                            size="small"
                            sx={{ height: 20, fontSize: '10px', bgcolor: minVal < 50 ? '#fef2f2' : '#f0fdf4', color: minVal < 50 ? '#dc2626' : '#16a34a' }}
                          />
                          <Chip
                            label={`${delta >= 0 ? '+' : ''}${delta.toFixed(0)}pp`}
                            size="small"
                            sx={{ height: 20, fontSize: '10px', bgcolor: delta < -20 ? '#fef2f2' : '#f0f9ff', color: delta < -20 ? '#dc2626' : '#0284c7' }}
                          />
                        </Box>
                      </Box>
                    );
                  })}
                </Box>
              </Paper>
            ) : (
              <Paper sx={{ p: 2, height: 350 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Fraction-to-Failure Summary
                </Typography>
                {(() => {
                  const fullData = modelExperimentResults?.full_data || modelExperimentResults;
                  const ftfInfo = fullData?.fraction_to_failure;
                  const hasFTF = ftfInfo && typeof ftfInfo === 'object';
                  if (!hasFTF) {
                    return (
                      <Alert severity="info">
                        <Typography variant="body2">
                          No fraction-to-failure experiments available for {VLA_MODELS[currentModel]?.name ?? currentModel}.
                        </Typography>
                      </Alert>
                    );
                  }
                  // Show aggregate FTF stats if available (SmolVLA format: count, episodes, skipped)
                  const eps = ftfInfo.episodes;
                  const count = ftfInfo.count;
                  const hasAggregate = typeof eps === 'number' && typeof count === 'number';
                  const skipped = ftfInfo.skipped;
                  const description = ftfInfo.description;
                  if (hasAggregate) {
                    return (
                      <Box>
                        <Grid container spacing={1}>
                          <Grid item xs={4}>
                            <Box sx={{ p: 1.5, bgcolor: '#f0f9ff', borderRadius: 1, textAlign: 'center' }}>
                              <Typography variant="h5" sx={{ fontWeight: 700, color: '#0284c7', fontSize: '1.3rem' }}>{eps.toLocaleString()}</Typography>
                              <Typography variant="caption">Total Episodes</Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ p: 1.5, bgcolor: '#fef3c7', borderRadius: 1, textAlign: 'center' }}>
                              <Typography variant="h5" sx={{ fontWeight: 700, color: '#d97706', fontSize: '1.3rem' }}>{count.toLocaleString()}</Typography>
                              <Typography variant="caption">Ablation Conditions</Typography>
                            </Box>
                          </Grid>
                          <Grid item xs={4}>
                            <Box sx={{ p: 1.5, bgcolor: '#f0fdf4', borderRadius: 1, textAlign: 'center' }}>
                              <Typography variant="h5" sx={{ fontWeight: 700, color: '#16a34a', fontSize: '1.3rem' }}>{typeof skipped === 'number' ? skipped.toLocaleString() : 'N/A'}</Typography>
                              <Typography variant="caption">Skipped (no effect)</Typography>
                            </Box>
                          </Grid>
                        </Grid>
                        {description && (
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                            {description}
                          </Typography>
                        )}
                        <Alert severity="info" sx={{ mt: 1.5, py: 0.5, '& .MuiAlert-message': { fontSize: '11px' } }}>
                          Aggregate data only. Per-layer titration curves not available. See the ablation studies tab for per-concept breakdowns.
                        </Alert>
                      </Box>
                    );
                  }
                  // For models with per_layer data, show layer count
                  const layerCount = Object.values(ftfInfo).filter((v: any) => typeof v === 'object' && v?.per_layer).length;
                  return (
                    <Box>
                      <Grid container spacing={1}>
                        <Grid item xs={6}>
                          <Box sx={{ p: 1.5, bgcolor: '#f0f9ff', borderRadius: 1, textAlign: 'center' }}>
                            <Typography variant="h5" sx={{ fontWeight: 700, color: '#0284c7', fontSize: '1.3rem' }}>{layerCount}</Typography>
                            <Typography variant="caption">Suites with FTF Data</Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    </Box>
                  );
                })()}
              </Paper>
            )}
          </Grid>
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                About Fractional Ablations
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Fractional ablation experiments test the <strong>dose-response relationship</strong> between
                the number of ablated features and task performance. By progressively ablating more features,
                we can determine how sparse the critical feature set is.
              </Typography>
              <Box sx={{ mt: 2, display: "flex", gap: 2, flexWrap: "wrap" }}>
                {fractionalChartData && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="Data" size="small" color="primary" />
                    <Typography variant="body2">
                      {fractionalChartData.series.length} layer{fractionalChartData.series.length !== 1 ? 's' : ''} with titration curves across {fractionalChartData.percentages.length} feature counts
                    </Typography>
                  </Box>
                )}
              </Box>
            </Paper>
          </Grid>
          {(currentModel === 'pi05') && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Dose-Response Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: "#f0fdf4", borderRadius: 1, textAlign: "center" }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: "#16a34a" }}>5</Typography>
                      <Typography variant="body2" color="text.secondary">Features to start seeing effects</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: "#fef3c7", borderRadius: 1, textAlign: "center" }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: "#d97706" }}>10-20</Typography>
                      <Typography variant="body2" color="text.secondary">Features for significant impact</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, bgcolor: "#fef2f2", borderRadius: 1, textAlign: "center" }}>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: "#dc2626" }}>50+</Typography>
                      <Typography variant="body2" color="text.secondary">Features for complete failure</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          )}
          {/* Injection Videos (cross-scene + temporal) */}
          <Grid item xs={12}>
            {renderVideoGrid(
              [...crossSceneInjectionVideos, ...temporalInjectionVideos],
              fractionalVideoPage,
              setFractionalVideoPage,
              'Injection Experiment Videos'
            )}
          </Grid>
        </Grid>
      )}

      {/* Tab 4: Concept Steering */}
      {mainTab === 4 && (
        <Grid container spacing={2}>
          {/* OFT/Pi0.5 steering data */}
          {(currentModel === 'openvla' || currentModel === 'pi05') && (<>
            {/* Summary finding */}
            <Grid item xs={12}>
              <Alert severity="warning" icon={false} sx={{ border: '1px solid #f59e0b' }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  One-Way Valve: Concept steering can selectively destroy task performance but cannot improve it.
                  {' '}{VLA_MODELS[currentModel]?.name ?? currentModel} shows no genuine improvement from any steering intervention.
                </Typography>
              </Alert>
            </Grid>

            {/* Pi0.5 stats */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Chip label="Pi0.5" size="small" sx={{ bgcolor: '#3b82f6', color: 'white', fontWeight: 600 }} />
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    Expert L08, Object Suite
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                  160 conditions (8 concepts x 2 strengths x 10 tasks), n=3 per condition
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#fef2f2', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#dc2626', fontSize: '1.3rem' }}>76.2%</Typography>
                      <Typography variant="caption">Degradation</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#f0f9ff', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#0284c7', fontSize: '1.3rem' }}>11.2%</Typography>
                      <Typography variant="caption">Zero Effect</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#fffbeb', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#d97706', fontSize: '1.3rem' }}>12.5%</Typography>
                      <Typography variant="caption">Noise (+)</Typography>
                    </Box>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2, p: 1.5, bgcolor: '#f8fafc', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Mean deltas:</Typography>
                  <Typography variant="body2">Suppression (-3x): <span style={{ color: '#dc2626', fontWeight: 700 }}>-84.2pp</span> (77/80 negative)</Typography>
                  <Typography variant="body2">Amplification (+3x): <span style={{ color: '#f59e0b', fontWeight: 700 }}>-36.2pp</span> (45/80 negative)</Typography>
                </Box>
              </Paper>
            </Grid>

            {/* OFT stats */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Chip label="OFT" size="small" sx={{ bgcolor: '#8b5cf6', color: 'white', fontWeight: 600 }} />
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    5 Layers, 3 Suites
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                  1,140 conditions (layers 3,4,12,21,25 x 3 suites x concepts), n=3 per condition
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#f0fdf4', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#16a34a', fontSize: '1.3rem' }}>89.7%</Typography>
                      <Typography variant="caption">Zero Effect</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#fef2f2', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#dc2626', fontSize: '1.3rem' }}>8.5%</Typography>
                      <Typography variant="caption">Degradation</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ p: 1.5, bgcolor: '#fffbeb', borderRadius: 1, textAlign: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#d97706', fontSize: '1.3rem' }}>1.8%</Typography>
                      <Typography variant="caption">Noise (+)</Typography>
                    </Box>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 2, p: 1.5, bgcolor: '#f8fafc', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>L03 Kill-Switches (-3x suppression):</Typography>
                  <Typography variant="body2"><strong>stove</strong>: 8/10 tasks destroyed to 0%</Typography>
                  <Typography variant="body2"><strong>cabinet</strong>: 8/10 tasks destroyed to 0%</Typography>
                  <Typography variant="body2"><strong>spatial/on</strong>: 9/10 tasks destroyed to 0%</Typography>
                </Box>
              </Paper>
            </Grid>

            {/* Dose-Response Comparison Chart */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: 350 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Steering Dose-Response Comparison
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  Task success rate (%) at different steering strengths
                </Typography>
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart data={steeringDoseComparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="strength" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} label={{ value: "Success %", angle: -90, position: "insideLeft", fontSize: 10 }} />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [`${v}%`, ""]} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="Pi0.5" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="OFT" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            {/* Effect Distribution Comparison */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: 350 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Effect Distribution Comparison
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  Percentage of steering conditions by outcome category
                </Typography>
                <ResponsiveContainer width="100%" height={270}>
                  <BarChart data={steeringDistributionData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} />
                    <YAxis dataKey="metric" type="category" width={110} tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [`${v}%`, ""]} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="Pi0.5" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                    <Bar dataKey="OFT" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Paper>
            </Grid>

            {/* Key Insights */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                  Key Insights
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="ONE-WAY" size="small" color="error" />
                    <Typography variant="body2">Steering can destroy but never improve — a one-way valve for both models</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="WIDTH" size="small" color="info" />
                    <Typography variant="body2">OFT (4096-dim) absorbs 89.7% of perturbations; Pi0.5 (1024-dim) is devastated by 76.2%</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="KILL" size="small" color="warning" />
                    <Typography variant="body2">OFT L03: stove/cabinet/spatial_on at -3x each destroy 8-9/10 tasks</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip label="NOISE" size="small" color="default" />
                    <Typography variant="body2">All &quot;positive&quot; results are stochastic noise (n=3, same task, unrelated concepts = identical +33pp)</Typography>
                  </Box>
                </Box>
              </Paper>
            </Grid>
          </>)}

          {/* SmolVLA, X-VLA, GR00T steering results */}
          {(currentModel === 'xvla' || currentModel === 'smolvla' || currentModel === 'groot') && (() => {
            const fullData = modelExperimentResults?.full_data || modelExperimentResults;
            const steeringSection = fullData?.concept_steering || fullData?.steering;
            // Extract steering experiment entries that have per-concept data
            const steeringEntries: Array<{key: string; layer: string; suite: string; concepts: Record<string, any>; strengths: number[]}> = [];
            if (steeringSection && typeof steeringSection === 'object') {
              Object.entries(steeringSection).forEach(([key, val]: [string, any]) => {
                if (val?.concepts && typeof val.concepts === 'object') {
                  steeringEntries.push({
                    key,
                    layer: val.layer || key,
                    suite: val.suite || '',
                    concepts: val.concepts,
                    strengths: val.strengths || [],
                  });
                }
                // GR00T format: per_layer with strengths per layer
                if (val?.per_layer && typeof val.per_layer === 'object') {
                  Object.entries(val.per_layer).forEach(([layerKey, layerVal]: [string, any]) => {
                    if (layerVal?.strengths) {
                      steeringEntries.push({
                        key: `${key}_${layerKey}`,
                        layer: layerKey,
                        suite: val.suite || key,
                        concepts: {},
                        strengths: layerVal.strengths || [],
                      });
                    }
                  });
                }
              });
            }
            // Build a dose-response chart from the selected concept if available
            const normalizedActiveConcept = activeConcept.replace('/', '_');
            const conceptSteeringData: Array<{strength: string; rate: number}> = [];
            let conceptEntry: any = null;
            for (const entry of steeringEntries) {
              const c = entry.concepts[activeConcept] || entry.concepts[normalizedActiveConcept];
              if (c?.per_strength) {
                conceptEntry = { ...c, layer: entry.layer, suite: entry.suite };
                Object.entries(c.per_strength).forEach(([str, data]: [string, any]) => {
                  conceptSteeringData.push({
                    strength: `${parseFloat(str) > 0 ? '+' : ''}${str}x`,
                    rate: Math.round((data.overall_success_rate ?? 0) * 100),
                  });
                });
                break;
              }
            }
            // Sort by numeric strength
            conceptSteeringData.sort((a, b) => parseFloat(a.strength) - parseFloat(b.strength));

            const modelColor = { xvla: '#f59e0b', smolvla: '#10b981', groot: '#ef4444' }[currentModel] || '#888';
            const knownStats: Record<string, {override: string; episodes: string; detail: string}> = {
              xvla: { override: '99.8%', episodes: '4,020', detail: 'Single-pathway Florence-2, 24 layers. Highest displacement override rate (99.8%) of all models.' },
              smolvla: { override: '10.1%', episodes: '13,530', detail: 'Dual-pathway (VLM + Expert). Most resilient to steering (10.1% override rate).' },
              groot: { override: 'N/A', episodes: '22,801', detail: 'Triple-pathway (DiT + Eagle + VL-SA). Steering across all three pathways.' },
            };
            const stats = knownStats[currentModel] || { override: 'N/A', episodes: 'N/A', detail: '' };
            return (<>
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <Chip label={VLA_MODELS[currentModel]?.name ?? currentModel} size="small" sx={{ fontWeight: 600, bgcolor: modelColor, color: 'white' }} />
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>Concept Steering Results</Typography>
                  </Box>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">{stats.detail}</Typography>
                  </Alert>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
                    <Box sx={{ p: 1.5, bgcolor: '#fef2f2', borderRadius: 1, textAlign: 'center', minWidth: 120 }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#dc2626', fontSize: '1.3rem' }}>{stats.override}</Typography>
                      <Typography variant="caption">Override Rate</Typography>
                    </Box>
                    <Box sx={{ p: 1.5, bgcolor: '#f0f9ff', borderRadius: 1, textAlign: 'center', minWidth: 120 }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#0284c7', fontSize: '1.3rem' }}>{stats.episodes}</Typography>
                      <Typography variant="caption">Steering Episodes</Typography>
                    </Box>
                    <Box sx={{ p: 1.5, bgcolor: '#f0fdf4', borderRadius: 1, textAlign: 'center', minWidth: 120 }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: '#16a34a', fontSize: '1.3rem' }}>{steeringEntries.length || 'N/A'}</Typography>
                      <Typography variant="caption">Experiments</Typography>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
              {/* Per-concept steering dose-response chart */}
              {conceptSteeringData.length > 0 && (
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2, height: 350 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                      Steering: {activeConcept.toUpperCase()} ({conceptEntry?.layer}, {conceptEntry?.suite})
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                      Task success rate (%) at different steering strengths
                    </Typography>
                    <ResponsiveContainer width="100%" height={270}>
                      <BarChart data={conceptSteeringData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="strength" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} label={{ value: "Success %", angle: -90, position: "insideLeft", fontSize: 10 }} />
                        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: unknown) => [`${v}%`, ""]} />
                        <Bar dataKey="rate" fill={modelColor} radius={[4, 4, 0, 0]}>
                          {conceptSteeringData.map((entry, idx) => (
                            <Cell key={idx} fill={entry.rate >= 80 ? '#22c55e' : entry.rate >= 50 ? '#f59e0b' : '#ef4444'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>
              )}
              {/* Steering experiments summary */}
              {steeringEntries.length > 0 && (
                <Grid item xs={12} md={conceptSteeringData.length > 0 ? 6 : 12}>
                  <Paper sx={{ p: 2, height: conceptSteeringData.length > 0 ? 350 : 'auto', overflow: 'auto' }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                      Steering Experiments Index
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {steeringEntries.slice(0, 10).map((entry) => {
                        const nConcepts = Object.keys(entry.concepts).length;
                        return (
                          <Box key={entry.key} sx={{ p: 1, borderRadius: 1, border: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Box>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>{entry.layer}</Typography>
                              <Typography variant="caption" color="text.secondary">{entry.suite}</Typography>
                            </Box>
                            <Box sx={{ display: 'flex', gap: 0.5 }}>
                              {nConcepts > 0 && <Chip label={`${nConcepts} concepts`} size="small" sx={{ height: 20, fontSize: '10px' }} />}
                              {entry.strengths.length > 0 && <Chip label={`${entry.strengths.length} strengths`} size="small" variant="outlined" sx={{ height: 20, fontSize: '10px' }} />}
                            </Box>
                          </Box>
                        );
                      })}
                      {steeringEntries.length > 10 && (
                        <Typography variant="caption" color="text.secondary">...and {steeringEntries.length - 10} more experiments</Typography>
                      )}
                    </Box>
                  </Paper>
                </Grid>
              )}
              {/* Ablation summary table */}
              {ablationSummaryData.length > 0 && (
                <Grid item xs={12}>
                  {renderAblationSummaryTable()}
                </Grid>
              )}
            </>);
          })()}

          {/* ACT placeholder */}
          {currentModel === 'act' && (
            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="body2">
                  ACT-ALOHA uses a CVAE decoder architecture. Concept steering experiments use grid ablation (4x4 region masking) rather than SAE feature steering. See the Layer-Phase Matrix tab for grid ablation results.
                </Typography>
              </Alert>
            </Grid>
          )}

          {/* Steering experiment videos */}
          {(() => {
            const steeringVids = getFilteredVideos('steering');
            const conceptAblVids = currentModel !== 'openvla' && currentModel !== 'pi05'
              ? getFilteredVideos('concept_ablation', activeConcept)
              : [];
            const allSteeringVids = [...steeringVids, ...conceptAblVids];
            if (allSteeringVids.length === 0) return null;
            return (
              <Grid item xs={12}>
                {renderVideoGrid(allSteeringVids, gridVideoPage, setGridVideoPage, 'Steering Experiment Videos')}
              </Grid>
            );
          })()}
        </Grid>
      )}

    </Box>
  );
}
