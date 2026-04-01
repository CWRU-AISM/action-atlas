"use client";
import React, { useEffect, useState, useMemo, useCallback } from "react";
import {
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  OutlinedInput,
  Checkbox,
  ListItemText,
  Tabs,
  Tab,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
  Alert,
  Button,
} from "@mui/material";
import Grid from "@mui/material/Grid2";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import FilterListIcon from "@mui/icons-material/FilterList";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import { useAppSelector } from "@/redux/hooks";
import { API_BASE_URL, apiUrl } from "@/config/api";
import { VLA_MODELS, DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";

// Video data interface matching backend response
interface VideoData {
  path: string;  // Relative path to video
  experiment_type: string;
  suite: string;
  subtype?: string;
  seed?: number;
  task?: number;
  success?: boolean;
  concept?: string;
  // Computed fields
  id?: string;
  filename?: string;
  model?: string;
}

// API response interface
interface VideosApiResponse {
  videos: VideoData[];
  total: number;
  filters: {
    models: string[];
    experiment_types: string[];
    suites: string[];
    concepts: string[];
  };
}

// Filter types
type ExperimentType = 'counterfactual' | 'cross_task' | 'vision_perturbation' | 'concept_ablation' | 'temporal_perturbation' | 'cross_scene_injection' | 'temporal_injection' | 'grid_ablation' | 'fraction_to_failure' | 'baseline' | 'sae_reconstruction' | 'steering';
type SuiteType = 'object' | 'spatial' | 'goal' | 'libero_10' | 'libero_90' | 'libero_goal' | 'libero_object' | 'libero_spatial' | 'libero_long';
type SuccessFilter = 'all' | 'success' | 'failure';

// Default filter options (will be updated from API response)
// Pi0.5 index: counterfactual, cross_task, vision_perturbation, concept_ablation, temporal_perturbation, cross_scene_injection, temporal_injection
// OFT index: adds grid_ablation
// GR00T index: fraction_to_failure
const DEFAULT_EXPERIMENT_TYPES: ExperimentType[] = ['counterfactual', 'cross_task', 'vision_perturbation', 'concept_ablation', 'temporal_perturbation', 'cross_scene_injection', 'temporal_injection', 'grid_ablation', 'fraction_to_failure', 'baseline', 'steering'];
const DEFAULT_SUITE_TYPES: SuiteType[] = ['object', 'spatial', 'goal', 'libero_10', 'libero_90', 'libero_goal', 'libero_object', 'libero_spatial', 'libero_long'];
const DEFAULT_CONCEPT_OPTIONS = ['baseline', 'put', 'open', 'push', 'pick', 'place', 'close', 'turn', 'grasp', 'lift'];

// LIBERO task description mapping by suite and task index
const LIBERO_TASK_DESCRIPTIONS: Record<string, Record<number, string>> = {
  libero_goal: {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
  },
  goal: {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
  },
  libero_spatial: {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
  },
  spatial: {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
  },
  libero_object: {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
  },
  object: {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
  },
  libero_10: {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
  },
  "10": {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
  },
};

// Extract task description from a video object
const getTaskDescription = (video: VideoData, model: string): string => {
  const path = video.path || '';

  if (model === 'openvla') {
    // OFT: path looks like "openvla/baseline/libero_goal/pick_up_the_alphabet_soup_and_place_it_in_the_basket/episode_0.mp4"
    const parts = path.split('/');
    // Find the segment after the suite name (libero_goal, libero_object, etc.)
    const suiteIdx = parts.findIndex(p => p.startsWith('libero_'));
    if (suiteIdx >= 0 && suiteIdx + 1 < parts.length) {
      const taskSegment = parts[suiteIdx + 1];
      // Skip date-like segments (e.g., 20260130_214236) and known non-task segments
      if (!/^\d{8}/.test(taskSegment) && taskSegment !== 'episode_0.mp4') {
        return taskSegment.replace(/_/g, ' ');
      }
    }
  }

  // Pi0.5 or fallback: use task number to look up description
  const suite = video.suite || '';
  const taskNum = video.task;
  if (taskNum !== undefined) {
    // Try exact suite name first, then with libero_ prefix
    const desc = LIBERO_TASK_DESCRIPTIONS[suite]?.[taskNum]
      || LIBERO_TASK_DESCRIPTIONS[`libero_${suite}`]?.[taskNum];
    if (desc) return desc;
  }

  // Try to extract task number from path for Pi0.5
  // Path: pi05/counterfactual/goal/libero_goal_20260125_150703/wrist_videos/baseline_0000_baseline_0_s42.mp4
  const taskMatch = path.match(/_(\d+)_(?:baseline|s\d|f\d)/);
  if (taskMatch) {
    const extractedTask = parseInt(taskMatch[1], 10);
    const suiteName = suite || 'goal';
    const desc = LIBERO_TASK_DESCRIPTIONS[suiteName]?.[extractedTask]
      || LIBERO_TASK_DESCRIPTIONS[`libero_${suiteName}`]?.[extractedTask];
    if (desc) return desc;
  }

  return '';
};

// Helper function to get video URL from backend
const getVideoUrl = (videoPath: string, model: string = 'pi05'): string => {
  // Video paths from API already include model prefix (e.g., "pi05/counterfactual/...")
  // Just append to the video endpoint
  return `${API_BASE_URL}/api/vla/video/${videoPath}`;
};

// Ablation video data (fetched dynamically)
interface AblationVideoData {
  path: string;
  concept: string;
  task?: number;
  success?: boolean;
  suite?: string;
  layer?: number;
}

// Ablation summary data from API
interface AblationSummaryData {
  vla_model: string;
  summary?: Array<{
    concept: string;
    n_features: number;
    concept_tasks_delta: number;
    other_tasks_delta: number;
    selectivity: number;
  }>;
  overview?: Record<string, any>;
  note?: string;
}

// Flipped video component - wraps video in container to flip content but not controls
function FlippedVideo({ src, maxHeight = "200px" }: { src: string; maxHeight?: string }) {
  return (
    <Box
      sx={{
        position: 'relative',
        width: '100%',
        maxHeight,
        borderRadius: '6px',
        overflow: 'hidden',
        backgroundColor: '#000',
      }}
    >
      <video
        src={src}
        controls
        autoPlay
        loop
        muted
        style={{
          width: '100%',
          maxHeight,
          transform: 'scaleY(-1)',  // Flip vertically only (video content)
        }}
      />
    </Box>
  );
}

// Model color map for chips and accents
const MODEL_COLORS: Record<string, string> = {
  pi05: '#3b82f6',
  openvla: '#8b5cf6',
  xvla: '#f59e0b',
  smolvla: '#10b981',
  groot: '#ef4444',
  act: '#6366f1',
};

// Model-specific descriptions for ablation effects
const ABLATION_DESCRIPTIONS: Record<string, string> = {
  pi05: 'Pi0.5 concept ablation is catastrophic (-60 to -100pp) due to the 1024-dim hidden space concentrating critical information. Ablating top 30/64 SAE features for object/motion concepts causes near-complete task failure.',
  openvla: 'OpenVLA-OFT concept ablation shows sparse effects: 91.6% of task-concept pairs have zero impact, while 8.4% show >10pp changes. The 4096-dim hidden space provides redundant encoding that resists targeted ablation.',
  xvla: 'X-VLA concept ablation across 24 layers and 4 LIBERO suites. The 1024-dim hidden space with flow-matching action generation shows moderate sensitivity to feature ablation.',
  smolvla: 'SmolVLA concept ablation across interleaved VLM + Expert pathways. The 480-dim expert pathway concentrates action-critical information, making ablation effects significant in expert layers.',
  groot: 'GR00T N1.5 fraction-to-failure analysis across DiT, Eagle, and VL-SA pathways. Progressive feature ablation reveals pathway-specific robustness profiles with 137K+ evaluation episodes.',
  act: 'ACT-ALOHA grid ablation and injection experiments on bimanual manipulation tasks.',
};

// Model-specific descriptions for Key Finding box
const KEY_FINDING_DESCRIPTIONS: Record<string, string> = {
  pi05: 'Per-token SAE reconstruction preserves task behavior. Both baseline and SAE reconstruction successfully complete the task, demonstrating that the SAE captures essential action information without losing critical behavioral details.',
  openvla: 'OpenVLA-OFT per-token SAE achieves 99.2% reconstruction fidelity (119/120 trials). The SAE captures essential action information in the 4096-dimensional hidden space.',
  xvla: 'X-VLA per-token SAE analysis across 24 TransformerBlocks with Florence-2 backbone. SAE features capture manipulation concepts that transfer across LIBERO suites and SimplerEnv environments.',
  smolvla: 'SmolVLA dual-pathway analysis reveals distinct concept specialization: the VLM pathway encodes visual scene understanding while the expert pathway captures action-specific motor primitives.',
  groot: 'GR00T N1.5 triple-pathway architecture (DiT + Eagle + VL-SA) shows distributed concept encoding with pathway-specific specialization for action generation via diffusion.',
  act: 'ACT-ALOHA CVAE decoder with action chunking. Grid ablation reveals critical encoder layers for bimanual coordination.',
};

// Model-specific SAE reconstruction notes
const SAE_NOTES: Record<string, string> = {
  pi05: 'Pi0.5 per-token SAE reconstruction preserves task behavior across all tested suites. SAE reconstruction rollout videos will be added soon. The baseline videos above show standard policy execution for comparison.',
  openvla: 'OpenVLA-OFT per-token SAE achieves 99.2% reconstruction fidelity (119/120 trials across all 4 LIBERO suites). SAE reconstruction rollout videos will be added soon. The baseline videos above confirm the policy works correctly without SAE intervention.',
  xvla: 'X-VLA SAE reconstruction analysis is in progress. The baseline videos above show standard policy execution for comparison.',
  smolvla: 'SmolVLA dual-pathway SAE analysis is in progress across VLM and expert pathways. Baseline videos show standard policy execution.',
  groot: 'GR00T N1.5 SAE analysis spans DiT, Eagle, and VL-SA pathways. Baseline videos show standard policy execution for comparison.',
  act: 'ACT-ALOHA uses a CVAE architecture. Baseline and intervention videos from grid ablation experiments are shown above.',
};

// Pagination constants
const VIDEOS_PER_PAGE = 24;

// Helper: check if a video's suite belongs to the current dataset.
// Handles mismatches where API returns "metaworld" but DATASET_SUITES
// expects "metaworld_easy", "metaworld_medium", etc.
// We match if either the video suite starts with any dataset suite value,
// or any dataset suite value starts with the video suite.
function suiteMatchesDataset(videoSuite: string, datasetSuiteValues: string[], datasetKey: string): boolean {
  if (!videoSuite) return true; // Keep videos without suite info
  // Direct match
  if (datasetSuiteValues.includes(videoSuite)) return true;
  // Prefix match: video suite "metaworld" matches dataset values "metaworld_easy", etc.
  // Also handles: dataset value "metaworld_easy" starts with video suite "metaworld"
  if (datasetSuiteValues.some(dsv => dsv.startsWith(videoSuite) || videoSuite.startsWith(dsv))) return true;
  // Also match against the dataset key itself (e.g., "metaworld")
  if (videoSuite === datasetKey || videoSuite.startsWith(datasetKey) || datasetKey.startsWith(videoSuite)) return true;
  return false;
}

export default function DemoVisualization() {
  // Video data state
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalVideos, setTotalVideos] = useState(0);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);

  // Available filter options from API
  const [availableExperimentTypes, setAvailableExperimentTypes] = useState<string[]>(DEFAULT_EXPERIMENT_TYPES);
  const [availableSuites, setAvailableSuites] = useState<string[]>(DEFAULT_SUITE_TYPES);
  const [availableConcepts, setAvailableConcepts] = useState<string[]>(DEFAULT_CONCEPT_OPTIONS);

  const [filterConcepts, setFilterConcepts] = useState<string[]>([]);
  const [videoTab, setVideoTab] = useState(0);
  const [selectedTask, setSelectedTask] = useState(3);

  // Ablation data state (fetched dynamically)
  const [ablationVideos, setAblationVideos] = useState<AblationVideoData[]>([]);
  const [ablationSummary, setAblationSummary] = useState<AblationSummaryData | null>(null);
  const [ablationLoading, setAblationLoading] = useState(false);

  // Filter states for video grid
  const [experimentFilter, setExperimentFilter] = useState<ExperimentType[]>([]);
  const [suiteFilter, setSuiteFilter] = useState<SuiteType[]>([]);
  const [successFilter, setSuccessFilter] = useState<SuccessFilter>('all');
  const [conceptFilter, setConceptFilter] = useState<string[]>([]);

  // Get selected concept from Redux (ConceptSelector)
  const { selectedConcept } = useAppSelector((state) => state.concept);

  // Get current model and dataset from Redux
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);

  // Fetch videos from backend API
  const fetchVideos = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch videos for the current model
      const url = `${API_BASE_URL}/api/vla/videos?model=${currentModel}`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to fetch videos: ${response.status} ${response.statusText}`);
      }

      const data: VideosApiResponse = await response.json();

      // Store all videos for filtering
      let allVideos = data.videos || [];

      // Filter by current dataset suites
      const datasetSuiteValues = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
      if (datasetSuiteValues.length > 0) {
        allVideos = allVideos.filter(v => suiteMatchesDataset(v.suite, datasetSuiteValues, currentDataset));
      }

      setTotalVideos(allVideos.length);

      // Apply filters client-side
      if (experimentFilter.length > 0) {
        allVideos = allVideos.filter(v => experimentFilter.includes(v.experiment_type as ExperimentType));
      }
      if (suiteFilter.length > 0) {
        allVideos = allVideos.filter(v => suiteFilter.includes(v.suite as SuiteType));
      }
      if (successFilter !== 'all') {
        const wantSuccess = successFilter === 'success';
        allVideos = allVideos.filter(v => {
          // Check various success indicators in the path or filename
          // Pattern: _s42 = success with seed 42, _f42 = failure with seed 42
          // Also check for explicit success/failure in path
          const path = v.path?.toLowerCase() || '';
          const filename = path.split('/').pop() || '';

          // Check for seed-based patterns: _s followed by digit (success), _f followed by digit (failure)
          const hasSuccessSeed = /_s\d/.test(filename);
          const hasFailureSeed = /_f\d/.test(filename);

          // Also check for explicit keywords
          const hasSuccessKeyword = path.includes('success');
          const hasFailureKeyword = path.includes('fail');

          // Determine if video is success or failure
          const isSuccess = hasSuccessSeed || hasSuccessKeyword || (v.success === true);
          const isFailure = hasFailureSeed || hasFailureKeyword || (v.success === false);

          if (wantSuccess) {
            return isSuccess && !isFailure;
          } else {
            return isFailure || (!isSuccess && !hasSuccessSeed);
          }
        });
      }

      // Sort videos: by experiment_type, then suite, then failures first, then path
      allVideos.sort((a, b) => {
        // Sort by experiment_type first
        const expCompare = (a.experiment_type || '').localeCompare(b.experiment_type || '');
        if (expCompare !== 0) return expCompare;

        // Then by suite
        const suiteCompare = (a.suite || '').localeCompare(b.suite || '');
        if (suiteCompare !== 0) return suiteCompare;

        // Then by success status (failures first for more interesting content)
        const aSuccess = a.success === true ? 1 : 0;
        const bSuccess = b.success === true ? 1 : 0;
        if (aSuccess !== bSuccess) return aSuccess - bSuccess;

        // Finally by path
        return (a.path || '').localeCompare(b.path || '');
      });

      setVideos(allVideos);

      // Update available filter options from API response
      if (data.filters) {
        if (data.filters.experiment_types?.length > 0) {
          setAvailableExperimentTypes(data.filters.experiment_types);
        }
        if (data.filters.suites?.length > 0) {
          setAvailableSuites(data.filters.suites);
        }
        if (data.filters.concepts?.length > 0) {
          setAvailableConcepts(data.filters.concepts);
        }
      }

      // Reset to page 1 when filters change
      setCurrentPage(1);
    } catch (err) {
      console.error('Error fetching videos:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch videos');
      setVideos([]);
    } finally {
      setLoading(false);
    }
  }, [experimentFilter, suiteFilter, successFilter, currentModel, currentDataset]);

  // Reset selections when model or dataset changes
  useEffect(() => {
    setExperimentFilter([]);
    setSuiteFilter([]);
    setSuccessFilter('all');
    setConceptFilter([]);
    setFilterConcepts([]);
    setCurrentPage(1);
    setVideoTab(0);
    setVideos([]);
    setAblationVideos([]);
    setAblationSummary(null);
  }, [currentModel, currentDataset]);

  // Fetch videos on mount and when filters/dataset change
  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  // Fetch ablation data (videos + summary) when model or dataset changes
  useEffect(() => {
    const fetchAblationData = async () => {
      setAblationLoading(true);
      try {
        const [videosRes, summaryRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/ablation/videos?model=${currentModel}&limit=50`),
          fetch(`${API_BASE_URL}/api/ablation/summary?model=${currentModel}`),
        ]);

        // Filter ablation videos by current dataset suites
        const datasetSuiteValues = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);

        if (videosRes.ok) {
          const vData = await videosRes.json();
          let ablVids = vData.data?.videos || vData.videos || [];
          if (datasetSuiteValues.length > 0) {
            ablVids = ablVids.filter((v: { suite?: string }) => suiteMatchesDataset(v.suite || '', datasetSuiteValues, currentDataset));
          }
          setAblationVideos(ablVids);
        }

        if (summaryRes.ok) {
          const sData = await summaryRes.json();
          setAblationSummary(sData.data || null);
        }
      } catch (err) {
        console.error('Error fetching ablation data:', err);
      } finally {
        setAblationLoading(false);
      }
    };

    fetchAblationData();
  }, [currentModel, currentDataset]);

  // Get available ablation concepts from fetched data
  const ablationConcepts = useMemo(() => {
    const concepts = new Set<string>();
    ablationVideos.forEach(v => {
      if (v.concept) concepts.add(v.concept);
    });
    // Also add concepts from summary
    if (ablationSummary?.summary) {
      ablationSummary.summary.forEach(s => concepts.add(s.concept));
    }
    return Array.from(concepts);
  }, [ablationVideos, ablationSummary]);

  // Get the active ablation video for comparison
  const activeAblationVideo = useMemo(() => {
    const concept = filterConcepts.length > 0 ? filterConcepts[0] : ablationConcepts[0] || 'put';
    // Find a video matching the selected concept
    const match = ablationVideos.find(v =>
      v.concept?.toLowerCase().includes(concept.toLowerCase())
    );
    return { video: match, concept };
  }, [filterConcepts, ablationConcepts, ablationVideos]);

  // Filter videos locally based on concept filter (since API may not support it)
  const filteredVideos = useMemo(() => {
    return videos.filter((video) => {
      // Concept filter (applied locally)
      if (conceptFilter.length > 0 && video.concept && !conceptFilter.includes(video.concept)) {
        return false;
      }
      return true;
    });
  }, [videos, conceptFilter]);

  // Paginated videos - only show current page
  const paginatedVideos = useMemo(() => {
    const startIndex = (currentPage - 1) * VIDEOS_PER_PAGE;
    return filteredVideos.slice(startIndex, startIndex + VIDEOS_PER_PAGE);
  }, [filteredVideos, currentPage]);

  const totalPages = Math.ceil(filteredVideos.length / VIDEOS_PER_PAGE);

  // Clear all filters
  const clearAllFilters = () => {
    setExperimentFilter([]);
    setSuiteFilter([]);
    setSuccessFilter('all');
    setConceptFilter([]);
  };

  const hasActiveFilters = experimentFilter.length > 0 || suiteFilter.length > 0 || successFilter !== 'all' || conceptFilter.length > 0;

  // Sync filter with global concept selection
  useEffect(() => {
    if (selectedConcept) {
      setFilterConcepts((prev) => {
        if (!prev.includes(selectedConcept)) {
          return [...prev, selectedConcept];
        }
        return prev;
      });
    }
  }, [selectedConcept]);

  // Active ablation concept label
  const activeAblationConcept = activeAblationVideo.concept;

  return (
    <Paper className="h-full flex flex-col rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className="h-10 flex items-center px-4 bg-[#0a1628] rounded-t-lg">
        <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
          Demo Videos
        </Typography>
        <Chip
          label={VLA_MODELS[currentModel].name}
          size="small"
          sx={{
            ml: 2,
            height: 18,
            fontSize: '9px',
            bgcolor: MODEL_COLORS[currentModel] || '#8b5cf6',
            color: 'white',
          }}
        />
      </div>

      {/* Content */}
      <Box className="flex-1 overflow-auto bg-gray-50 p-3">
        {loading ? (
          <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" height="100%" gap={2}>
            <CircularProgress size={32} sx={{ color: '#ef4444' }} />
            <Typography color="text.secondary">Loading videos...</Typography>
          </Box>
        ) : error ? (
          <Box display="flex" flexDirection="column" justifyContent="center" alignItems="center" height="100%" gap={2}>
            <Alert severity="error" sx={{ maxWidth: 400 }}>
              {error}
            </Alert>
            <Chip
              label="Retry"
              onClick={fetchVideos}
              sx={{
                backgroundColor: '#ef4444',
                color: 'white',
                '&:hover': { backgroundColor: '#dc2626' }
              }}
            />
          </Box>
        ) : (
          <Box>
            {/* Tabs for video types */}
            <Tabs value={videoTab} onChange={(_, v) => setVideoTab(v)} sx={{ mb: 2 }}>
              <Tab label="Baseline vs SAE" sx={{ fontSize: '11px' }} />
              <Tab label="Ablation Comparisons" sx={{ fontSize: '11px' }} />
              <Tab label="Video Library" sx={{ fontSize: '11px' }} />
            </Tabs>

            {videoTab === 0 && (
              /* Baseline vs SAE Reconstruction Tab */
              <Box>
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <Typography variant="body2" color="text.secondary" flex={1}>
                    Compare baseline policy with SAE reconstruction (encode → decode). Both should succeed.
                  </Typography>
                  <Chip
                    label={VLA_MODELS[currentModel].name}
                    size="small"
                    sx={{
                      fontSize: '9px',
                      bgcolor: MODEL_COLORS[currentModel] || '#8b5cf6',
                      color: 'white',
                    }}
                  />
                </Box>

                {/* Show baseline + SAE reconstruction videos */}
                {(() => {
                  const baselineVideos = videos.filter(v =>
                    v.experiment_type === 'baseline' ||
                    v.subtype === 'baseline' ||
                    (v.experiment_type === 'counterfactual' && v.subtype === 'baseline')
                  );

                  const saeVideos = videos.filter(v =>
                    v.experiment_type === 'sae_reconstruction' ||
                    v.subtype === 'sae_reconstruction' ||
                    v.experiment_type === 'sae' ||
                    v.subtype === 'sae' ||
                    (v.path && v.path.toLowerCase().includes('sae'))
                  );

                  const hasSaeVideos = saeVideos.length > 0;
                  const BASELINE_DISPLAY_LIMIT = 20;
                  const displayedBaseline = baselineVideos.slice(0, BASELINE_DISPLAY_LIMIT);
                  const displayedSae = saeVideos.slice(0, BASELINE_DISPLAY_LIMIT);

                  if (baselineVideos.length === 0 && saeVideos.length === 0) {
                    return (
                      <Box py={4} textAlign="center">
                        <Typography color="text.secondary">
                          No baseline videos available for {VLA_MODELS[currentModel].name} on {currentDataset}.
                          Try switching to a different dataset or the Video Library tab.
                        </Typography>
                      </Box>
                    );
                  }

                  return (
                    <Box>
                      {/* Baseline Videos Section */}
                      {displayedBaseline.length > 0 && (
                        <Box mb={3}>
                          <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                            <Typography variant="subtitle2" fontWeight="bold">
                              Baseline Rollouts
                            </Typography>
                            <Chip
                              label={`${baselineVideos.length} total`}
                              size="small"
                              variant="outlined"
                              sx={{ height: 18, fontSize: '9px' }}
                            />
                            {baselineVideos.length > BASELINE_DISPLAY_LIMIT && (
                              <Typography variant="caption" color="text.secondary">
                                (showing first {BASELINE_DISPLAY_LIMIT})
                              </Typography>
                            )}
                          </Box>
                          <Grid container spacing={2}>
                            {displayedBaseline.map((video, idx) => {
                              const taskDesc = getTaskDescription(video, currentModel);
                              return (
                              <Grid key={video.path || idx} size={{ xs: 12, sm: 6, md: 4 }}>
                                <Card variant="outlined" sx={{
                                  transition: 'all 0.2s',
                                  '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.1)', borderColor: '#3b82f6' }
                                }}>
                                  <CardContent sx={{ p: 2 }}>
                                    {/* Centered task description label */}
                                    {taskDesc && (
                                      <Box display="flex" justifyContent="center" mb={1}>
                                        <Chip
                                          label={taskDesc}
                                          size="small"
                                          sx={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            '& .MuiChip-label': {
                                              whiteSpace: 'normal',
                                              textAlign: 'center',
                                              py: 0.5,
                                            },
                                            fontSize: '10px',
                                            bgcolor: '#eff6ff',
                                            color: '#1d4ed8',
                                            border: '1px solid #bfdbfe',
                                          }}
                                        />
                                      </Box>
                                    )}
                                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                                      <PlayCircleOutlineIcon fontSize="small" color="primary" />
                                      <Typography variant="caption" fontWeight="bold" sx={{ textTransform: 'capitalize', flex: 1, fontSize: '10px' }}>
                                        {video.suite?.replace(/_/g, ' ') || 'Baseline'}
                                      </Typography>
                                      {video.task !== undefined && (
                                        <Chip label={typeof video.task === 'number' ? `T${video.task}` : String(video.task).slice(0, 20)} size="small" variant="outlined" sx={{ height: 16, fontSize: '8px' }} />
                                      )}
                                      {video.success !== undefined && (
                                        <Chip
                                          label={video.success ? 'Success' : 'Failure'}
                                          size="small"
                                          color={video.success ? 'success' : 'error'}
                                          sx={{ height: 16, fontSize: '8px' }}
                                        />
                                      )}
                                    </Box>
                                    <Box
                                      sx={{
                                        width: '100%',
                                        aspectRatio: '4/3',
                                        bgcolor: '#000',
                                        borderRadius: 1,
                                        overflow: 'hidden',
                                      }}
                                    >
                                      <video
                                        src={getVideoUrl(video.path, currentModel)}
                                        controls
                                        muted
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        onError={(e) => { (e.target as HTMLVideoElement).style.display = 'none'; }}
                                      />
                                    </Box>
                                  </CardContent>
                                </Card>
                              </Grid>
                              );
                            })}
                          </Grid>
                        </Box>
                      )}

                      {/* SAE Reconstruction Videos Section */}
                      {hasSaeVideos && (
                        <Box mb={3}>
                          <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                            <Typography variant="subtitle2" fontWeight="bold">
                              SAE Reconstruction Rollouts
                            </Typography>
                            <Chip
                              label={`${saeVideos.length} total`}
                              size="small"
                              variant="outlined"
                              sx={{ height: 18, fontSize: '9px' }}
                            />
                          </Box>
                          <Grid container spacing={2}>
                            {displayedSae.map((video, idx) => {
                              const taskDesc = getTaskDescription(video, currentModel);
                              return (
                              <Grid key={video.path || idx} size={{ xs: 12, sm: 6, md: 4 }}>
                                <Card variant="outlined" sx={{
                                  transition: 'all 0.2s',
                                  borderColor: '#8b5cf6',
                                  '&:hover': { boxShadow: '0 4px 12px rgba(139,92,246,0.15)', borderColor: '#7c3aed' }
                                }}>
                                  <CardContent sx={{ p: 2 }}>
                                    {/* Centered task description label */}
                                    {taskDesc && (
                                      <Box display="flex" justifyContent="center" mb={1}>
                                        <Chip
                                          label={taskDesc}
                                          size="small"
                                          sx={{
                                            maxWidth: '100%',
                                            height: 'auto',
                                            '& .MuiChip-label': {
                                              whiteSpace: 'normal',
                                              textAlign: 'center',
                                              py: 0.5,
                                            },
                                            fontSize: '10px',
                                            bgcolor: '#f5f3ff',
                                            color: '#6d28d9',
                                            border: '1px solid #ddd6fe',
                                          }}
                                        />
                                      </Box>
                                    )}
                                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                                      <PlayCircleOutlineIcon fontSize="small" sx={{ color: '#8b5cf6' }} />
                                      <Typography variant="caption" fontWeight="bold" sx={{ textTransform: 'capitalize', flex: 1, fontSize: '10px' }}>
                                        {video.suite?.replace(/_/g, ' ') || 'SAE Reconstruction'}
                                      </Typography>
                                      {video.success !== undefined && (
                                        <Chip
                                          label={video.success ? 'Success' : 'Failure'}
                                          size="small"
                                          color={video.success ? 'success' : 'error'}
                                          sx={{ height: 16, fontSize: '8px' }}
                                        />
                                      )}
                                    </Box>
                                    <Box
                                      sx={{
                                        width: '100%',
                                        aspectRatio: '4/3',
                                        bgcolor: '#000',
                                        borderRadius: 1,
                                        overflow: 'hidden',
                                      }}
                                    >
                                      <video
                                        src={getVideoUrl(video.path, currentModel)}
                                        controls
                                        muted
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        onError={(e) => { (e.target as HTMLVideoElement).style.display = 'none'; }}
                                      />
                                    </Box>
                                  </CardContent>
                                </Card>
                              </Grid>
                              );
                            })}
                          </Grid>
                        </Box>
                      )}

                      {/* Explanatory note when no SAE reconstruction videos exist */}
                      {!hasSaeVideos && baselineVideos.length > 0 && (
                        <Alert severity="info" sx={{ mb: 2 }}>
                          <Typography variant="body2" fontWeight="bold" gutterBottom>
                            SAE Reconstruction Videos Not Yet Available
                          </Typography>
                          <Typography variant="body2">
                            {SAE_NOTES[currentModel] || 'SAE reconstruction analysis is in progress. Baseline videos show standard policy execution.'}
                          </Typography>
                        </Alert>
                      )}
                    </Box>
                  );
                })()}

                <Box mt={2} p={2} bgcolor="white" borderRadius={1} border="1px solid #e5e7eb">
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Key Finding
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {KEY_FINDING_DESCRIPTIONS[currentModel] || 'SAE feature analysis captures essential action information for this model architecture.'}
                  </Typography>
                </Box>
              </Box>
            )}

            {videoTab === 1 && (
              /* Ablation Comparisons Tab */
              <Box>
                {/* Filter Controls */}
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <Typography variant="body2" color="text.secondary" flex={1}>
                    Side-by-side comparison of baseline vs concept-ablated rollouts ({VLA_MODELS[currentModel].name}).
                  </Typography>
                  {ablationConcepts.length > 0 && (
                    <FormControl size="small" sx={{ minWidth: 200 }}>
                      <InputLabel id="concept-filter-label">
                        <Box display="flex" alignItems="center" gap={0.5}>
                          <FilterListIcon fontSize="small" />
                          Filter by Concept
                        </Box>
                      </InputLabel>
                      <Select
                        labelId="concept-filter-label"
                        multiple
                        value={filterConcepts}
                        onChange={(e) => setFilterConcepts(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                        input={<OutlinedInput label="Filter by Concept" />}
                        renderValue={(selected) => selected.length === 0 ? 'All' : selected.join(', ')}
                        sx={{ fontSize: '12px' }}
                      >
                        {ablationConcepts.map((concept) => (
                          <MenuItem key={concept} value={concept}>
                            <Checkbox checked={filterConcepts.includes(concept)} size="small" />
                            <ListItemText primary={concept} primaryTypographyProps={{ sx: { textTransform: 'capitalize', fontSize: '12px' } }} />
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                  {filterConcepts.length > 0 && (
                    <Chip
                      label="Clear filters"
                      size="small"
                      onDelete={() => setFilterConcepts([])}
                      sx={{ fontSize: '10px' }}
                    />
                  )}
                </Box>

                {ablationLoading ? (
                  <Box display="flex" justifyContent="center" py={4}>
                    <CircularProgress size={24} sx={{ color: '#ef4444' }} />
                  </Box>
                ) : ablationVideos.length === 0 ? (
                  <Box py={4} textAlign="center">
                    <Typography color="text.secondary" mb={1}>
                      No ablation videos available for {VLA_MODELS[currentModel].name} on {currentDataset}.
                    </Typography>
                    {ablationSummary && (
                      <Typography variant="body2" color="text.secondary">
                        {ablationSummary.note || `${ablationSummary.vla_model} ablation data loaded from summary.`}
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <>
                    {/* Ablation Video Grid */}
                    <Grid container spacing={2}>
                      {ablationVideos
                        .filter(v => filterConcepts.length === 0 || filterConcepts.some(c => v.concept?.toLowerCase().includes(c.toLowerCase())))
                        .slice(0, VIDEOS_PER_PAGE)
                        .map((video, idx) => {
                          const ablationTaskDesc = video.task !== undefined
                            ? (LIBERO_TASK_DESCRIPTIONS[video.suite || '']?.[video.task]
                              || LIBERO_TASK_DESCRIPTIONS[`libero_${video.suite || ''}`]?.[video.task]
                              || '')
                            : '';
                          return (
                        <Grid key={video.path || idx} size={{ xs: 12, sm: 6, md: 3 }}>
                          <Card variant="outlined" sx={{
                            transition: 'all 0.2s',
                            '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.1)', borderColor: '#ef4444' }
                          }}>
                            <CardContent sx={{ p: 2 }}>
                              {/* Task description at top of card */}
                              {ablationTaskDesc && (
                                <Box display="flex" justifyContent="center" mb={1}>
                                  <Chip
                                    label={`T${video.task}: ${ablationTaskDesc}`}
                                    size="small"
                                    sx={{
                                      maxWidth: '100%',
                                      height: 'auto',
                                      '& .MuiChip-label': {
                                        whiteSpace: 'normal',
                                        textAlign: 'center',
                                        py: 0.5,
                                      },
                                      fontSize: '9px',
                                      bgcolor: '#fef2f2',
                                      color: '#b91c1c',
                                      border: '1px solid #fecaca',
                                    }}
                                  />
                                </Box>
                              )}
                              <Box display="flex" alignItems="center" gap={1} mb={1}>
                                <PlayCircleOutlineIcon fontSize="small" sx={{ color: video.success ? '#22c55e' : '#ef4444' }} />
                                <Typography variant="caption" fontWeight="bold" sx={{ flex: 1, fontSize: '10px' }}>
                                  {video.concept || 'Ablated'}
                                </Typography>
                                <Chip
                                  label={video.success ? 'Success' : 'Failure'}
                                  size="small"
                                  color={video.success ? 'success' : 'error'}
                                  sx={{ height: 16, fontSize: '8px' }}
                                />
                              </Box>
                              <Box display="flex" gap={0.5} mb={1} flexWrap="wrap">
                                {video.suite && (
                                  <Chip label={video.suite.replace(/_/g, ' ')} size="small" variant="outlined" sx={{ height: 14, fontSize: '8px' }} />
                                )}
                                {video.task !== undefined && (
                                  <Chip label={`T${video.task}`} size="small" variant="outlined" sx={{ height: 14, fontSize: '8px' }} />
                                )}
                                {video.layer !== undefined && (
                                  <Chip label={`L${video.layer}`} size="small" variant="outlined" sx={{ height: 14, fontSize: '8px' }} />
                                )}
                              </Box>
                              <Box sx={{ width: '100%', aspectRatio: '4/3', bgcolor: '#0a1628', borderRadius: 1, overflow: 'hidden' }}>
                                <video
                                  src={video.path.startsWith('/api/') ? `${API_BASE_URL}${video.path}` : getVideoUrl(video.path, currentModel)}
                                  controls
                                  muted
                                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                  onError={(e) => { (e.target as HTMLVideoElement).style.display = 'none'; }}
                                />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                          );
                      })}
                    </Grid>
                  </>
                )}

                {/* Ablation Summary */}
                <Box mt={2} p={2} bgcolor="white" borderRadius={1} border="1px solid #e5e7eb">
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Ablation Effect — {VLA_MODELS[currentModel].name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {ABLATION_DESCRIPTIONS[currentModel] || `${VLA_MODELS[currentModel].name} concept ablation analysis across available experiment types and suites.`}
                  </Typography>
                  {ablationSummary?.summary && (
                    <Box mt={1} display="flex" gap={1} flexWrap="wrap">
                      {ablationSummary.summary.map(s => (
                        <Chip
                          key={s.concept}
                          label={`${s.concept}: ${s.concept_tasks_delta > 0 ? '+' : ''}${s.concept_tasks_delta}pp`}
                          size="small"
                          sx={{
                            fontSize: '10px',
                            bgcolor: s.concept_tasks_delta < -50 ? '#fee2e2' : s.concept_tasks_delta < -10 ? '#fef3c7' : '#f0fdf4',
                            color: s.concept_tasks_delta < -50 ? '#dc2626' : s.concept_tasks_delta < -10 ? '#92400e' : '#166534',
                          }}
                        />
                      ))}
                    </Box>
                  )}
                </Box>
              </Box>
            )}

            {videoTab === 2 && (
              /* Video Library Tab with Filters */
              <Box>
                {/* Filter Panel - Dark theme with red accents */}
                <Box
                  className="p-4 mb-4 rounded-lg"
                  sx={{
                    backgroundColor: '#0a1628',
                    border: '1px solid #1e3a5f'
                  }}
                >
                  <Box display="flex" alignItems="center" gap={1} mb={3}>
                    <FilterListIcon sx={{ color: '#ef4444', fontSize: 20 }} />
                    <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
                      Filter Videos
                    </Typography>
                    {hasActiveFilters && (
                      <Chip
                        label="Clear All"
                        size="small"
                        onClick={clearAllFilters}
                        sx={{
                          ml: 'auto',
                          fontSize: '10px',
                          height: 24,
                          backgroundColor: '#ef4444',
                          color: 'white',
                          '&:hover': { backgroundColor: '#dc2626' }
                        }}
                      />
                    )}
                  </Box>

                  <Grid container spacing={2}>
                    {/* Experiment Type Filter */}
                    <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel
                          id="experiment-filter-label"
                          sx={{ color: '#94a3b8', '&.Mui-focused': { color: '#ef4444' } }}
                        >
                          Experiment Type
                        </InputLabel>
                        <Select
                          labelId="experiment-filter-label"
                          multiple
                          value={experimentFilter}
                          onChange={(e) => setExperimentFilter(e.target.value as ExperimentType[])}
                          input={<OutlinedInput label="Experiment Type" />}
                          renderValue={(selected) => selected.length === 0 ? 'All' : selected.join(', ')}
                          sx={{
                            backgroundColor: '#1e293b',
                            color: 'white',
                            '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '& .MuiSvgIcon-root': { color: '#94a3b8' },
                            fontSize: '12px',
                          }}
                          MenuProps={{
                            PaperProps: {
                              sx: { backgroundColor: '#1e293b', color: 'white' }
                            }
                          }}
                        >
                          {availableExperimentTypes.map((type) => (
                            <MenuItem key={type} value={type} sx={{ '&:hover': { backgroundColor: '#334155' } }}>
                              <Checkbox
                                checked={experimentFilter.includes(type as ExperimentType)}
                                size="small"
                                sx={{ color: '#94a3b8', '&.Mui-checked': { color: '#ef4444' } }}
                              />
                              <ListItemText
                                primary={type.replace('_', ' ')}
                                primaryTypographyProps={{ sx: { textTransform: 'capitalize', fontSize: '12px' } }}
                              />
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    {/* Suite Filter */}
                    <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel
                          id="suite-filter-label"
                          sx={{ color: '#94a3b8', '&.Mui-focused': { color: '#ef4444' } }}
                        >
                          Suite
                        </InputLabel>
                        <Select
                          labelId="suite-filter-label"
                          multiple
                          value={suiteFilter}
                          onChange={(e) => setSuiteFilter(e.target.value as SuiteType[])}
                          input={<OutlinedInput label="Suite" />}
                          renderValue={(selected) => selected.length === 0 ? 'All' : selected.join(', ')}
                          sx={{
                            backgroundColor: '#1e293b',
                            color: 'white',
                            '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '& .MuiSvgIcon-root': { color: '#94a3b8' },
                            fontSize: '12px',
                          }}
                          MenuProps={{
                            PaperProps: {
                              sx: { backgroundColor: '#1e293b', color: 'white' }
                            }
                          }}
                        >
                          {availableSuites.map((suite) => (
                            <MenuItem key={suite} value={suite} sx={{ '&:hover': { backgroundColor: '#334155' } }}>
                              <Checkbox
                                checked={suiteFilter.includes(suite as SuiteType)}
                                size="small"
                                sx={{ color: '#94a3b8', '&.Mui-checked': { color: '#ef4444' } }}
                              />
                              <ListItemText
                                primary={suite.replace('_', ' ')}
                                primaryTypographyProps={{ sx: { textTransform: 'capitalize', fontSize: '12px' } }}
                              />
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    {/* Concept Filter */}
                    <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel
                          id="concept-grid-filter-label"
                          sx={{ color: '#94a3b8', '&.Mui-focused': { color: '#ef4444' } }}
                        >
                          Concept
                        </InputLabel>
                        <Select
                          labelId="concept-grid-filter-label"
                          multiple
                          value={conceptFilter}
                          onChange={(e) => setConceptFilter(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                          input={<OutlinedInput label="Concept" />}
                          renderValue={(selected) => selected.length === 0 ? 'All' : selected.join(', ')}
                          sx={{
                            backgroundColor: '#1e293b',
                            color: 'white',
                            '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                            '& .MuiSvgIcon-root': { color: '#94a3b8' },
                            fontSize: '12px',
                          }}
                          MenuProps={{
                            PaperProps: {
                              sx: { backgroundColor: '#1e293b', color: 'white' }
                            }
                          }}
                        >
                          {availableConcepts.map((concept) => (
                            <MenuItem key={concept} value={concept} sx={{ '&:hover': { backgroundColor: '#334155' } }}>
                              <Checkbox
                                checked={conceptFilter.includes(concept)}
                                size="small"
                                sx={{ color: '#94a3b8', '&.Mui-checked': { color: '#ef4444' } }}
                              />
                              <ListItemText
                                primary={concept}
                                primaryTypographyProps={{ sx: { textTransform: 'capitalize', fontSize: '12px' } }}
                              />
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>

                    {/* Success/Failure Toggle */}
                    <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                      <Box>
                        <Typography variant="caption" sx={{ color: '#94a3b8', mb: 0.5, display: 'block' }}>
                          Result
                        </Typography>
                        <ToggleButtonGroup
                          value={successFilter}
                          exclusive
                          onChange={(_, value) => value && setSuccessFilter(value)}
                          size="small"
                          sx={{
                            '& .MuiToggleButton-root': {
                              color: '#94a3b8',
                              borderColor: '#334155',
                              fontSize: '11px',
                              px: 1.5,
                              py: 0.5,
                              '&.Mui-selected': {
                                backgroundColor: '#ef4444',
                                color: 'white',
                                '&:hover': { backgroundColor: '#dc2626' }
                              },
                              '&:hover': { backgroundColor: '#334155' }
                            }
                          }}
                        >
                          <ToggleButton value="all">All</ToggleButton>
                          <ToggleButton value="success">
                            <CheckCircleIcon sx={{ fontSize: 14, mr: 0.5 }} />
                            Success
                          </ToggleButton>
                          <ToggleButton value="failure">
                            <CancelIcon sx={{ fontSize: 14, mr: 0.5 }} />
                            Failure
                          </ToggleButton>
                        </ToggleButtonGroup>
                      </Box>
                    </Grid>
                  </Grid>
                </Box>

                {/* Results count and pagination */}
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Typography variant="body2" color="text.secondary">
                    Showing {paginatedVideos.length} of {filteredVideos.length} videos (page {currentPage}/{totalPages || 1})
                  </Typography>
                  {totalPages > 1 && (
                    <Box display="flex" gap={1} alignItems="center">
                      <Button
                        size="small"
                        disabled={currentPage === 1}
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
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
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        sx={{ minWidth: 32, fontSize: '11px' }}
                      >
                        Next
                      </Button>
                    </Box>
                  )}
                  {hasActiveFilters && (
                    <Box display="flex" gap={1} flexWrap="wrap">
                      {experimentFilter.map((exp) => (
                        <Chip
                          key={exp}
                          label={exp.replace('_', ' ')}
                          size="small"
                          onDelete={() => setExperimentFilter(prev => prev.filter(e => e !== exp))}
                          sx={{
                            fontSize: '10px',
                            height: 20,
                            backgroundColor: '#fee2e2',
                            color: '#dc2626',
                            '& .MuiChip-deleteIcon': { color: '#dc2626', fontSize: 14 }
                          }}
                        />
                      ))}
                      {suiteFilter.map((suite) => (
                        <Chip
                          key={suite}
                          label={suite.replace('_', ' ')}
                          size="small"
                          onDelete={() => setSuiteFilter(prev => prev.filter(s => s !== suite))}
                          sx={{
                            fontSize: '10px',
                            height: 20,
                            backgroundColor: '#fee2e2',
                            color: '#dc2626',
                            '& .MuiChip-deleteIcon': { color: '#dc2626', fontSize: 14 }
                          }}
                        />
                      ))}
                      {conceptFilter.map((concept) => (
                        <Chip
                          key={concept}
                          label={concept}
                          size="small"
                          onDelete={() => setConceptFilter(prev => prev.filter(c => c !== concept))}
                          sx={{
                            fontSize: '10px',
                            height: 20,
                            backgroundColor: '#fee2e2',
                            color: '#dc2626',
                            '& .MuiChip-deleteIcon': { color: '#dc2626', fontSize: 14 }
                          }}
                        />
                      ))}
                    </Box>
                  )}
                </Box>

                {/* Video Grid */}
                {paginatedVideos.length === 0 ? (
                  <Box
                    display="flex"
                    flexDirection="column"
                    justifyContent="center"
                    alignItems="center"
                    py={6}
                    sx={{ backgroundColor: '#f8fafc', borderRadius: 1 }}
                  >
                    <Typography color="text.secondary" mb={1}>
                      No videos match the selected filters
                    </Typography>
                    <Chip
                      label="Clear Filters"
                      size="small"
                      onClick={clearAllFilters}
                      sx={{
                        backgroundColor: '#ef4444',
                        color: 'white',
                        '&:hover': { backgroundColor: '#dc2626' }
                      }}
                    />
                  </Box>
                ) : (
                  <Grid container spacing={2}>
                    {paginatedVideos.map((video, index) => (
                      <Grid key={video.path || index} size={{ xs: 12, sm: 6, md: 4 }}>
                        <Card
                          variant="outlined"
                          sx={{
                            transition: 'all 0.2s',
                            '&:hover': {
                              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                              borderColor: '#ef4444'
                            }
                          }}
                        >
                          <CardContent sx={{ p: 2 }}>
                            {/* Header with experiment type and success badge */}
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                              <PlayCircleOutlineIcon
                                fontSize="small"
                                sx={{ color: video.success ? '#22c55e' : '#ef4444' }}
                              />
                              <Typography
                                variant="caption"
                                sx={{
                                  textTransform: 'capitalize',
                                  fontWeight: 600,
                                  color: '#374151'
                                }}
                              >
                                {(video.experiment_type || 'unknown').replace('_', ' ')}
                              </Typography>
                              {video.success !== undefined && (
                              <Chip
                                label={video.success ? 'Success' : 'Failure'}
                                size="small"
                                color={video.success ? 'success' : 'error'}
                                sx={{ height: 18, fontSize: '9px', ml: 'auto' }}
                              />
                              )}
                            </Box>

                            {/* Metadata tags */}
                            <Box display="flex" gap={0.5} mb={1.5} flexWrap="wrap">
                              {video.suite && (
                              <Chip
                                label={`Suite: ${video.suite.replace('_', ' ')}`}
                                size="small"
                                variant="outlined"
                                sx={{
                                  height: 18,
                                  fontSize: '9px',
                                  textTransform: 'capitalize'
                                }}
                              />
                              )}
                              {video.task !== undefined && (
                              <Chip
                                label={`Task ${video.task}`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 18, fontSize: '9px' }}
                              />
                              )}
                              {video.concept !== 'baseline' && (
                                <Chip
                                  label={video.concept}
                                  size="small"
                                  sx={{
                                    height: 18,
                                    fontSize: '9px',
                                    backgroundColor: '#fef2f2',
                                    color: '#dc2626',
                                    textTransform: 'capitalize'
                                  }}
                                />
                              )}
                              {video.model && (
                                <Chip
                                  label={video.model}
                                  size="small"
                                  variant="outlined"
                                  sx={{
                                    height: 18,
                                    fontSize: '9px',
                                    borderColor: '#6366f1',
                                    color: '#6366f1'
                                  }}
                                />
                              )}
                            </Box>

                            {/* Video placeholder/player */}
                            <Box
                              sx={{
                                width: '100%',
                                aspectRatio: '4/3',
                                backgroundColor: '#0a1628',
                                borderRadius: 1,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                position: 'relative',
                                overflow: 'hidden'
                              }}
                            >
                              {/* Video element - fetches from backend */}
                              <video
                                src={getVideoUrl(video.path, currentModel)}
                                controls
                                muted
                                style={{
                                  width: '100%',
                                  height: '100%',
                                  objectFit: 'contain',
                                }}
                                onError={(e) => {
                                  // Hide video element on error, show placeholder
                                  (e.target as HTMLVideoElement).style.display = 'none';
                                }}
                              />
                              {/* Fallback placeholder shown if video fails to load */}
                              <Box
                                sx={{
                                  position: 'absolute',
                                  top: 0,
                                  left: 0,
                                  right: 0,
                                  bottom: 0,
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  pointerEvents: 'none',
                                  zIndex: -1
                                }}
                              >
                                <PlayCircleOutlineIcon sx={{ color: '#ef4444', fontSize: 32, mb: 0.5 }} />
                                <Typography variant="caption" sx={{ color: '#94a3b8', fontSize: '10px' }}>
                                  Video Placeholder
                                </Typography>
                              </Box>
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Box>
            )}
          </Box>
        )}
      </Box>
    </Paper>
  );
}
