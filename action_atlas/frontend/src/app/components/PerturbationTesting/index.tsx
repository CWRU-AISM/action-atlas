"use client";
import React, { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  Paper,
  Typography,
  Box,
  Slider,
  Button,
  CircularProgress,
  Chip,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Snackbar,
  ToggleButton,
  ToggleButtonGroup,
  Card,
  CardContent,
  Autocomplete,
  TextField,
} from "@mui/material";
import Grid from "@mui/material/Grid2";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import { API_BASE_URL } from "@/config/api";
import { useAppSelector } from "@/redux/hooks";
import { DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
import GridAblationDemo from "./GridAblationDemo";
import CounterfactualDemo from "./CounterfactualDemo";
import InjectionDemo from "./InjectionDemo";

// Convert a static /videos/... path to an API-served URL.
// Static paths rely on Next.js public/ serving (broken in Docker where symlinks are invalid).
// API paths go through the Flask backend which resolves symlinks and Docker volume mounts.
const toVideoApiUrl = (staticPath: string): string => {
  if (!staticPath) return '';
  // Already a full URL (e.g., http://...) — return as-is
  if (staticPath.startsWith('http://') || staticPath.startsWith('https://')) return staticPath;
  // Already a fully-qualified API URL with base — return as-is
  if (API_BASE_URL && staticPath.startsWith(API_BASE_URL)) return staticPath;
  // Already an API path — prepend base URL
  if (staticPath.startsWith('/api/')) return `${API_BASE_URL}${staticPath}`;
  // Strip leading /videos/ prefix and serve through the video API endpoint
  const cleaned = staticPath.replace(/^\/videos\//, '');
  return `${API_BASE_URL}/api/vla/video/${cleaned}`;
};

// Model display names
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  pi05: 'Pi0.5',
  openvla: 'OpenVLA-OFT',
  xvla: 'X-VLA',
  smolvla: 'SmolVLA',
  groot: 'GR00T',
  act: 'ACT-ALOHA',
};

const getModelDisplayName = (model: string): string =>
  MODEL_DISPLAY_NAMES[model] || model.toUpperCase();

// Model chip colors
const MODEL_CHIP_COLORS: Record<string, string> = {
  pi05: '#3b82f6',
  openvla: '#8b5cf6',
  xvla: '#f59e0b',
  smolvla: '#10b981',
  groot: '#ef4444',
  act: '#64748b',
};

const getModelChipColor = (model: string): string =>
  MODEL_CHIP_COLORS[model] || '#8b5cf6';

// Mapping from perturbation type to experiment type
const PERTURBATION_TO_EXPERIMENT: Record<string, string> = {
  // Vision perturbations
  noise: 'vision_perturbation',
  blur: 'vision_perturbation',
  crop: 'vision_perturbation',
  h_flip: 'vision_perturbation',
  v_flip: 'vision_perturbation',
  rotate: 'vision_perturbation',
  grayscale: 'vision_perturbation',
  invert: 'vision_perturbation',
  brightness: 'vision_perturbation',
  contrast: 'vision_perturbation',
  saturation: 'vision_perturbation',
  // Counterfactual
  prompt_null: 'counterfactual',
  prompt_swap: 'counterfactual',
  // Injection — each maps to its own experiment type in the baked index
  cross_task: 'cross_task',
  cross_scene: 'cross_scene_injection',
  temporal_inject: 'temporal_injection',
};

// Category to experiment type mapping (supports multiple types per category)
const CATEGORY_TO_EXPERIMENT: Record<string, string[]> = {
  vision: ['vision_perturbation'],
  counterfactual: ['counterfactual'],
  injection: ['cross_task', 'cross_scene_injection', 'temporal_injection'],
};

// Default perturbation types organized by category
const DEFAULT_PERTURBATION_TYPES: PerturbationTypeConfig[] = [
  // Vision perturbations
  { id: "noise", label: "Noise", hasStrength: true, icon: "~", category: "vision" },
  { id: "blur", label: "Blur", hasStrength: true, icon: "B", category: "vision" },
  { id: "crop", label: "Crop", hasStrength: true, icon: "C", category: "vision" },
  { id: "h_flip", label: "H-Flip", hasStrength: false, icon: "<>", category: "vision" },
  { id: "v_flip", label: "V-Flip", hasStrength: false, icon: "^v", category: "vision" },
  { id: "rotate", label: "Rotate", hasStrength: true, icon: "R", category: "vision" },
  { id: "grayscale", label: "Grayscale", hasStrength: false, icon: "G", category: "vision" },
  { id: "invert", label: "Invert", hasStrength: false, icon: "I", category: "vision" },
  { id: "brightness", label: "Brightness", hasStrength: true, icon: "L", category: "vision" },
  { id: "contrast", label: "Contrast", hasStrength: true, icon: "K", category: "vision" },
  { id: "saturation", label: "Saturation", hasStrength: true, icon: "S", category: "vision" },
  // Object perturbations
  { id: "object_remove", label: "Remove Object", hasStrength: false, icon: "✕", category: "object" },
  { id: "object_swap", label: "Swap Object", hasStrength: false, icon: "⇄", category: "object" },
  { id: "object_color", label: "Object Color", hasStrength: true, icon: "🎨", category: "object" },
  { id: "object_size", label: "Object Size", hasStrength: true, icon: "⊡", category: "object" },
  { id: "object_position", label: "Object Position", hasStrength: true, icon: "↔", category: "object" },
  // Counterfactual / Injection experiments
  { id: "prompt_null", label: "Null Prompt", hasStrength: false, icon: "∅", category: "counterfactual" },
  { id: "prompt_swap", label: "Swap Prompt", hasStrength: false, icon: "⟳", category: "counterfactual" },
  { id: "cross_task", label: "Cross-Task Inject", hasStrength: true, icon: "⤨", category: "injection" },
  { id: "cross_scene", label: "Cross-Scene Inject", hasStrength: true, icon: "⊕", category: "injection" },
  { id: "temporal_inject", label: "Temporal Inject", hasStrength: true, icon: "⏱", category: "injection" },
];

// Type definitions for API responses
interface PerturbationTypeConfig {
  id: string;
  label: string;
  hasStrength: boolean;
  icon: string;
  category?: "vision" | "object" | "counterfactual" | "injection";
}

interface PerturbationTypesResponse {
  perturbation_types: PerturbationTypeConfig[];
}

interface PerturbResponse {
  success: boolean;
  perturbed_image: string; // base64 encoded image
  error?: string;
}

interface ExtractFrameResponse {
  success: boolean;
  frame: string; // base64 encoded image
  error?: string;
}

type PerturbationType = string;

interface VPExperimentResult {
  perturbation: string;
  success_rate: number;
  baseline_success_rate: number | null;
  delta_success_rate: number | null;
  avg_n_steps: number | null;
  baseline_avg_n_steps: number | null;
  delta_n_steps: number | null;
  n_episodes: number;
  video_path: string | null;
  baseline_video_path: string | null;
}

// Icon components for perturbation types
const PerturbationIcon = ({ type }: { type: PerturbationType }) => {
  const iconMap: Record<string, React.ReactNode> = {
    noise: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="2" cy="4" r="1" />
        <circle cx="8" cy="2" r="1" />
        <circle cx="14" cy="5" r="1" />
        <circle cx="4" cy="8" r="1" />
        <circle cx="10" cy="7" r="1" />
        <circle cx="6" cy="12" r="1" />
        <circle cx="12" cy="11" r="1" />
        <circle cx="3" cy="14" r="1" />
      </svg>
    ),
    blur: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" opacity="0.5">
        <circle cx="8" cy="8" r="6" />
      </svg>
    ),
    crop: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 1v11h11" />
        <path d="M1 4h11v11" />
      </svg>
    ),
    h_flip: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <path d="M1 8l4-4v8z" />
        <path d="M15 8l-4-4v8z" />
        <rect x="7" y="2" width="2" height="12" />
      </svg>
    ),
    v_flip: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 1l-4 4h8z" />
        <path d="M8 15l-4-4h8z" />
        <rect x="2" y="7" width="12" height="2" />
      </svg>
    ),
    rotate: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M14 8A6 6 0 1 1 8 2" />
        <path d="M14 2v6h-6" fill="currentColor" />
      </svg>
    ),
    grayscale: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <rect x="1" y="1" width="6" height="14" fill="#333" />
        <rect x="9" y="1" width="6" height="14" fill="#aaa" />
      </svg>
    ),
    invert: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 1a7 7 0 0 1 0 14z" />
      </svg>
    ),
    brightness: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="8" cy="8" r="3" />
        <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.5 1.5M11.5 11.5L13 13M3 13l1.5-1.5M11.5 4.5L13 3" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    ),
    contrast: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1" />
        <path d="M8 1v14" stroke="currentColor" strokeWidth="1" />
        <path d="M8 1a7 7 0 0 1 0 14" fill="currentColor" />
      </svg>
    ),
    saturation: (
      <svg width="16" height="16" viewBox="0 0 16 16">
        <circle cx="6" cy="6" r="4" fill="#ef4444" opacity="0.8" />
        <circle cx="10" cy="6" r="4" fill="#22c55e" opacity="0.8" />
        <circle cx="8" cy="10" r="4" fill="#3b82f6" opacity="0.8" />
      </svg>
    ),
    // Object perturbations
    object_remove: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="10" height="10" rx="1" />
        <path d="M6 8h4" strokeLinecap="round" />
      </svg>
    ),
    object_swap: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="1" y="5" width="5" height="6" rx="1" />
        <rect x="10" y="5" width="5" height="6" rx="1" />
        <path d="M6 6l2-2 2 2M6 10l2 2 2-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    object_color: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <rect x="2" y="6" width="6" height="8" rx="1" fill="#ef4444" />
        <circle cx="12" cy="6" r="4" fill="#3b82f6" />
      </svg>
    ),
    object_size: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <rect x="2" y="4" width="4" height="4" rx="0.5" />
        <rect x="8" y="2" width="6" height="6" rx="0.5" opacity="0.6" />
        <path d="M1 12h14M4 10v4M12 10v4" stroke="currentColor" strokeWidth="1" />
      </svg>
    ),
    object_position: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="2" width="4" height="4" rx="0.5" fill="currentColor" />
        <path d="M8 4h6M11 1v6" strokeLinecap="round" />
        <rect x="10" y="10" width="4" height="4" rx="0.5" strokeDasharray="2 1" />
      </svg>
    ),
    // Counterfactual experiments
    prompt_null: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="8" cy="8" r="6" />
        <path d="M4 4l8 8" strokeLinecap="round" />
      </svg>
    ),
    prompt_swap: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M2 4h12M2 8h12M2 12h8" strokeLinecap="round" />
        <path d="M12 10l2 2-2 2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    // Injection experiments
    cross_task: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="1" y="1" width="6" height="6" rx="1" />
        <rect x="9" y="9" width="6" height="6" rx="1" />
        <path d="M7 4h2M4 7v2M9 12h-2M12 9v-2" strokeLinecap="round" />
      </svg>
    ),
    cross_scene: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="1" y="1" width="14" height="10" rx="1" />
        <path d="M1 8l4-3 3 2 4-4 3 3" />
        <circle cx="12" cy="4" r="1.5" fill="currentColor" />
      </svg>
    ),
    temporal_inject: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="8" cy="8" r="6" />
        <path d="M8 4v4l3 2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M12 2l2-1v3h-3" fill="currentColor" />
      </svg>
    ),
  };
  return <>{iconMap[type] || <span style={{ fontSize: 12 }}>{type.charAt(0).toUpperCase()}</span>}</>;
};

// Placeholder image component
const PlaceholderImage = ({ label }: { label: string }) => (
  <Box
    sx={{
      width: "100%",
      height: "100%",
      minHeight: 200,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      bgcolor: "#1a1a2e",
      borderRadius: 1,
      border: "2px dashed #444",
    }}
  >
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="#666" strokeWidth="2">
      <rect x="4" y="8" width="40" height="32" rx="2" />
      <circle cx="16" cy="20" r="4" />
      <path d="M4 36l12-12 8 8 8-10 12 14" />
    </svg>
    <Typography variant="caption" sx={{ color: "#666", mt: 1 }}>
      {label}
    </Typography>
  </Box>
);

// Category tabs for organizing perturbation types
type PerturbationCategory = "vision" | "object" | "counterfactual" | "injection";
const CATEGORY_LABELS: Record<PerturbationCategory, { label: string; description: string }> = {
  vision: { label: "Vision", description: "Image-level perturbations (noise, blur, etc.)" },
  object: { label: "Objects", description: "Object-level manipulations (remove, swap, resize)" },
  counterfactual: { label: "Counterfactual", description: "Prompt and language perturbations" },
  injection: { label: "Injections", description: "Cross-task/scene activation injections" },
};

// Color mapping for perturbation subtypes displayed as chips/badges
const PERTURBATION_CHIP_COLORS: Record<string, { bg: string; text: string }> = {
  // Blur variants
  blur: { bg: '#1e40af', text: '#93c5fd' },
  blur_heavy: { bg: '#1e3a8a', text: '#93c5fd' },
  blur_light: { bg: '#2563eb', text: '#bfdbfe' },
  blur_medium: { bg: '#1d4ed8', text: '#93c5fd' },
  // Noise variants
  noise: { bg: '#7c2d12', text: '#fed7aa' },
  noise_high: { bg: '#7c2d12', text: '#fed7aa' },
  noise_low: { bg: '#9a3412', text: '#fdba74' },
  noise_medium: { bg: '#c2410c', text: '#fed7aa' },
  // Crop variants
  crop: { bg: '#065f46', text: '#6ee7b7' },
  crop_50: { bg: '#065f46', text: '#6ee7b7' },
  crop_70: { bg: '#047857', text: '#6ee7b7' },
  crop_90: { bg: '#059669', text: '#a7f3d0' },
  crop_left_half: { bg: '#065f46', text: '#6ee7b7' },
  crop_right_half: { bg: '#047857', text: '#6ee7b7' },
  crop_top_half: { bg: '#059669', text: '#a7f3d0' },
  crop_bottom_half: { bg: '#065f46', text: '#6ee7b7' },
  // Flip variants
  h_flip: { bg: '#581c87', text: '#d8b4fe' },
  v_flip: { bg: '#6b21a8', text: '#d8b4fe' },
  // Rotate variants
  rotate: { bg: '#164e63', text: '#67e8f9' },
  rotate_15: { bg: '#164e63', text: '#67e8f9' },
  rotate_45: { bg: '#155e75', text: '#67e8f9' },
  // Color perturbations
  grayscale: { bg: '#374151', text: '#d1d5db' },
  invert: { bg: '#1f2937', text: '#e5e7eb' },
  brightness: { bg: '#713f12', text: '#fde68a' },
  bright_up: { bg: '#713f12', text: '#fde68a' },
  bright_down: { bg: '#78350f', text: '#fcd34d' },
  contrast: { bg: '#44403c', text: '#e7e5e4' },
  contrast_up: { bg: '#44403c', text: '#e7e5e4' },
  contrast_down: { bg: '#57534e', text: '#d6d3d1' },
  saturation: { bg: '#831843', text: '#f9a8d4' },
  hue_shift: { bg: '#701a75', text: '#f0abfc' },
  // Occlusion
  occlusion_small: { bg: '#991b1b', text: '#fca5a5' },
  occlusion_large: { bg: '#7f1d1d', text: '#fca5a5' },
  // Baseline
  baseline: { bg: '#166534', text: '#86efac' },
};

const getPerturbationChipColor = (subtype: string): { bg: string; text: string } => {
  const normalized = subtype.toLowerCase().replace(/\s+/g, '_');
  if (PERTURBATION_CHIP_COLORS[normalized]) return PERTURBATION_CHIP_COLORS[normalized];
  // Fallback: try to match the prefix (e.g., "blur_foo" -> "blur")
  const prefix = normalized.split('_')[0];
  if (PERTURBATION_CHIP_COLORS[prefix]) return PERTURBATION_CHIP_COLORS[prefix];
  return { bg: '#334155', text: '#94a3b8' };
};

// Format a subtype string for display: "blur_heavy" -> "Blur Heavy"
const formatPerturbationLabel = (subtype: string): string => {
  return subtype
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

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

// Get the task description for a given suite and task number
const getTaskDescriptionForSuiteTask = (suite: string, task: number): string => {
  return LIBERO_TASK_DESCRIPTIONS[suite]?.[task] || '';
};

// Available baseline videos by model/suite/task
const BASELINE_VIDEOS: Record<string, Record<string, Record<number, string>>> = {
  pi05: {
    libero_goal: Object.fromEntries(Array.from({length: 10}, (_, i) => [i, `pi05/baseline/libero_goal/task_${i}_seed42.mp4`])),
    libero_spatial: Object.fromEntries(Array.from({length: 10}, (_, i) => [i, `pi05/baseline/libero_spatial/task_${i}_seed42.mp4`])),
    libero_object: Object.fromEntries(Array.from({length: 10}, (_, i) => [i, `pi05/baseline/libero_object/task_${i}_seed42.mp4`])),
    libero_10: Object.fromEntries(Array.from({length: 10}, (_, i) => [i, `pi05/baseline/libero_10/task_${i}_seed42.mp4`])),
  },
  openvla: {
    libero_goal: {
      0: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task0_baseline.mp4',
      1: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task1_baseline.mp4',
      2: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task2_baseline.mp4',
      3: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task3_baseline.mp4',
      4: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task4_baseline.mp4',
      5: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task5_baseline.mp4',
      6: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task6_baseline.mp4',
      7: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task7_baseline.mp4',
      8: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task8_baseline.mp4',
      9: '/videos/openvla/openvla_oft/libero_goal/20260130_214236/null_injection/task9_baseline.mp4',
    },
    libero_object: {
      0: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task0_baseline.mp4',
      1: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task1_baseline.mp4',
      2: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task2_baseline.mp4',
      3: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task3_baseline.mp4',
      4: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task4_baseline.mp4',
      5: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task5_baseline.mp4',
      6: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task6_baseline.mp4',
      7: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task7_baseline.mp4',
      8: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task8_baseline.mp4',
      9: '/videos/openvla/openvla_oft/libero_object/20260130_215041/null_injection/task9_baseline.mp4',
    },
    libero_spatial: {
      0: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task0_baseline.mp4',
      1: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task1_baseline.mp4',
      2: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task2_baseline.mp4',
      3: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task3_baseline.mp4',
      4: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task4_baseline.mp4',
      5: '/videos/openvla/openvla_oft/libero_spatial/20260130_220941/null_injection/task5_baseline.mp4',
    },
    libero_10: {
      0: '/videos/openvla/openvla_oft/libero_10/20260130_220749/null_injection/task0_baseline.mp4',
      1: '/videos/openvla/openvla_oft/libero_10/20260130_220749/null_injection/task1_baseline.mp4',
      2: '/videos/openvla/openvla_oft/libero_10/20260130_220749/null_injection/task2_baseline.mp4',
      3: '/videos/openvla/openvla_oft/libero_10/20260130_220749/null_injection/task3_baseline.mp4',
    },
  },
};

// Removal video interface (fetched dynamically from API)
interface RemovalVideoData {
  path: string;
  concept: string;
  task?: number;
  suite?: string;
  success?: boolean;
  layer?: number;
  concept_type?: string;
}

export default function PerturbationTesting() {
  // Get model and dataset from Redux
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);

  // Main tab state: 0 = Baseline vs Removed, 1 = Perturbation Testing, 2 = Grid Ablation, 3 = Counterfactual, 4 = Injection
  const [mainTab, setMainTab] = useState(0);

  // State for Baseline vs Removed comparison
  const [selectedSuite, setSelectedSuite] = useState<string>('libero_goal');
  const [selectedTask, setSelectedTask] = useState<number>(0);
  const [selectedConcept, setSelectedConcept] = useState<string>('');

  // Dynamic removal/ablation videos from API
  const [removalVideos, setRemovalVideos] = useState<RemovalVideoData[]>([]);
  const [removalConcepts, setRemovalConcepts] = useState<string[]>([]);
  const [ablationSummaryNote, setAblationSummaryNote] = useState<string>('');
  const [isLoadingRemovalVideos, setIsLoadingRemovalVideos] = useState(false);

  // State for Perturbation Testing
  const [selectedPerturbation, setSelectedPerturbation] = useState<PerturbationType>("noise");
  const [selectedCategory, setSelectedCategory] = useState<PerturbationCategory>("vision");
  const [strength, setStrength] = useState<number>(50);
  const [sourceTab, setSourceTab] = useState(0); // 0: Upload, 1: From Video
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [perturbedImage, setPerturbedImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<VPExperimentResult | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<string>("");
  const [selectedFrame, setSelectedFrame] = useState<number>(0);
  const [availableVideos, setAvailableVideos] = useState<{id: string; label: string; experiment_type?: string; suite?: string; success?: boolean; steps?: number}[]>([]);
  const [availablePrompts, setAvailablePrompts] = useState<string[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);
  const [isLoadingPrompts, setIsLoadingPrompts] = useState(false);
  const [availableExperiments, setAvailableExperiments] = useState<Set<string>>(new Set());
  const [isLoadingVideos, setIsLoadingVideos] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // API-related state
  const [perturbationTypes, setPerturbationTypes] = useState<PerturbationTypeConfig[]>(DEFAULT_PERTURBATION_TYPES);
  const [isLoadingTypes, setIsLoadingTypes] = useState(true);
  const [isApplyingPerturbation, setIsApplyingPerturbation] = useState(false);
  const [isExtractingFrame, setIsExtractingFrame] = useState(false);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [showError, setShowError] = useState(false);

  // State for perturbation rollout videos
  const [perturbRolloutVideos, setPerturbRolloutVideos] = useState<{path: string; suite: string; subtype: string; task: string; seed: string}[]>([]);
  const [isLoadingRolloutVideos, setIsLoadingRolloutVideos] = useState(false);
  const [rolloutSubtypeFilter, setRolloutSubtypeFilter] = useState<string>("all");

  // Helper: parse concepts from the /api/concepts/list response.
  // Response format: { data: { motion: [...], object: [...], spatial: [...] }, metadata: {...} }
  const parseConceptsApiResponse = (conceptsData: Record<string, unknown>): string[] => {
    const dataObj = (conceptsData.data || conceptsData) as Record<string, string[]>;
    const result: string[] = [];
    for (const [category, names] of Object.entries(dataObj)) {
      if (Array.isArray(names)) {
        for (const name of names) {
          result.push(`${category}/${name}`);
        }
      }
    }
    return result;
  };

  // Helper: extract concept name from a video path when the concept field is missing.
  // Handles paths like "smolvla/ablation/concept_ablation/.../ablation_L00_motion/close/ep00.mp4"
  // and "groot/ablation/sae_feature_ablation/.../ablation_dit_L00_disc_libero_goal_task0_task0_ep0.mp4"
  const extractConceptFromPath = (videoPath: string): string => {
    // SmolVLA pattern: .../ablation_L{N}_{concept_type}/{concept_name}/...
    const smolvlaMatch = videoPath.match(/ablation_L\d+_(\w+)\/(\w+)\//);
    if (smolvlaMatch) {
      return `${smolvlaMatch[1]}/${smolvlaMatch[2]}`;
    }
    // GR00T pattern: ablation_{component}_L{N}_disc_{suite}_task{N}_{...}
    // or concept_ablation paths with concept info in directory structure
    const grootDirMatch = videoPath.match(/concept_ablation\/(\w+)\/(\w+)\//);
    if (grootDirMatch) {
      return `${grootDirMatch[1]}/${grootDirMatch[2]}`;
    }
    // Generic fallback: try to find concept_type/concept_name in path segments
    const segments = videoPath.split('/');
    for (let i = 0; i < segments.length - 1; i++) {
      const seg = segments[i];
      // Match ablation_L{N}_{type} pattern
      const ablMatch = seg.match(/ablation_(?:L|dit_L|eagle_L|vlsa_L|vlm_layer|expert_layer)\d+_(\w+)/);
      if (ablMatch) {
        const conceptType = ablMatch[1];
        // Next segment might be the concept name
        const nextSeg = segments[i + 1];
        if (nextSeg && !nextSeg.endsWith('.mp4') && !nextSeg.startsWith('task') && !nextSeg.startsWith('ep')) {
          return `${conceptType}/${nextSeg}`;
        }
        return conceptType;
      }
    }
    return '';
  };

  // Fetch ablation/removal videos from API when model changes
  useEffect(() => {
    const fetchRemovalVideos = async () => {
      setIsLoadingRemovalVideos(true);
      try {
        // Fetch suites belonging to the current dataset
        const datasetSuites = DATASET_SUITES[currentDataset as DatasetType] || [];
        const suites = datasetSuites.map(s => s.value);

        // Use ablation/videos endpoint for ALL models (backend now supports all via baked index)
        const usePerSuiteFetch = suites.length > 0 && (currentModel === 'openvla');

        const videosFetches = usePerSuiteFetch
          ? suites.map(s => fetch(`${API_BASE_URL}/api/ablation/videos?model=${currentModel}&suite=${s}&limit=5000`))
          : [fetch(`${API_BASE_URL}/api/ablation/videos?model=${currentModel}&limit=5000`)];

        const summaryFetch = fetch(`${API_BASE_URL}/api/ablation/summary?model=${currentModel}`);

        const [summaryRes, ...videosResults] = await Promise.all([
          summaryFetch,
          ...videosFetches,
        ]);

        // Collect all videos from all suite fetches
        const allVideos: RemovalVideoData[] = [];
        for (const videosRes of videosResults) {
          if (videosRes.ok) {
            const vData = await videosRes.json();
            const videos: RemovalVideoData[] = vData.data?.videos || vData.videos || [];
            allVideos.push(...videos);
          }
        }

        // Also try the baked video index as a fallback source (works for all models)
        try {
          const indexRes = await fetch(`${API_BASE_URL}/api/vla/videos?model=${currentModel}&experiment_type=concept_ablation&limit=5000`);
          if (indexRes.ok) {
            const indexData = await indexRes.json();
            const indexVideos = indexData.videos || [];
            // Filter by current dataset suites
            const datasetSuiteValues = suites.length > 0 ? new Set(suites) : null;
            // Merge in any videos from the index that we don't already have (by path)
            const existingPaths = new Set(allVideos.map(v => v.path));
            for (const v of indexVideos) {
              if (v.path && !existingPaths.has(v.path)) {
                // Filter by dataset suites if available
                if (datasetSuiteValues && v.suite && !datasetSuiteValues.has(v.suite)) {
                  continue;
                }
                // Extract concept from path if not provided
                const concept = v.concept || extractConceptFromPath(v.path || '');
                allVideos.push({
                  path: v.path,
                  concept: concept,
                  task: v.task,
                  suite: v.suite,
                  success: v.success,
                  layer: v.layer,
                  concept_type: v.concept_type,
                });
              }
            }
          }
        } catch {
          // Ignore fallback failures
        }

        // Post-process: extract concepts from paths for any video missing the concept field
        for (const v of allVideos) {
          if (!v.concept && v.path) {
            v.concept = extractConceptFromPath(v.path);
          }
        }

        setRemovalVideos(allVideos);
        // Extract unique concepts from all suites
        const concepts = [...new Set(allVideos.map(v => v.concept).filter(Boolean))] as string[];
        let foundConcepts = concepts.length > 0;
        if (foundConcepts) {
          setRemovalConcepts(concepts.sort());
          // Reset selected concept if current one is not available
          if (!concepts.includes(selectedConcept)) {
            setSelectedConcept(concepts[0]);
          }
        }

        if (summaryRes.ok) {
          const sData = await summaryRes.json();
          const summaryData = sData.data;
          if (summaryData?.note) {
            setAblationSummaryNote(summaryData.note);
          }
          // If no videos but we have summary concepts, use those
          if (summaryData?.summary && allVideos.length === 0) {
            const summaryConcepts = (summaryData.summary as { concept: string }[]).map(s => s.concept);
            if (summaryConcepts.length > 0) {
              setRemovalConcepts(summaryConcepts.sort());
              foundConcepts = true;
              if (!summaryConcepts.includes(selectedConcept)) {
                setSelectedConcept(summaryConcepts[0]);
              }
            }
          }
        }

        // Always fetch from concepts list API as a reliable source of concept names.
        // This supplements concepts found in ablation videos/summary, and provides
        // concepts for models that don't have ablation video data (SmolVLA, GR00T, etc.)
        try {
          const conceptsRes = await fetch(`${API_BASE_URL}/api/concepts/list?model=${currentModel}`);
          if (conceptsRes.ok) {
            const conceptsData = await conceptsRes.json();
            const apiConcepts = parseConceptsApiResponse(conceptsData);
            if (apiConcepts.length > 0) {
              // Merge with any concepts already found from ablation videos
              const merged = foundConcepts
                ? [...new Set([...concepts, ...apiConcepts])].sort()
                : apiConcepts.sort();
              setRemovalConcepts(merged);
              if (!merged.includes(selectedConcept)) {
                setSelectedConcept(merged[0]);
              }
            }
          }
        } catch {
          // Ignore concepts API failures — we may still have concepts from ablation videos
        }
      } catch (err) {
        console.error('Error fetching removal videos:', err);
      } finally {
        setIsLoadingRemovalVideos(false);
      }
    };
    fetchRemovalVideos();
  }, [currentModel, currentDataset]);

  // Reset suite/task selection when model or dataset changes to avoid stale selections
  useEffect(() => {
    // Derive suites from ablation data first, then baseline videos as fallback
    const ablationSuites = [...new Set(removalVideos.map(v => v.suite).filter(Boolean))] as string[];
    const baselineSuites = Object.keys(BASELINE_VIDEOS[currentModel] || {});
    const allSuites = ablationSuites.length > 0 ? ablationSuites : baselineSuites;
    // Also filter by current dataset suites
    const datasetSuiteValues = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
    const filteredSuites = datasetSuiteValues.length > 0
      ? allSuites.filter(s => datasetSuiteValues.includes(s))
      : allSuites;
    const suitesToUse = filteredSuites.length > 0 ? filteredSuites : allSuites;
    if (suitesToUse.length > 0) {
      const newSuite = suitesToUse.includes(selectedSuite) ? selectedSuite : suitesToUse[0];
      setSelectedSuite(newSuite);
    }
  }, [currentModel, currentDataset, removalVideos]);

  // Get available suites from ablation data (primary) and BASELINE_VIDEOS (fallback), filtered by dataset
  const availableSuites = useMemo(() => {
    const ablationSuites = [...new Set(removalVideos.map(v => v.suite).filter(Boolean))] as string[];
    const baselineSuites = Object.keys(BASELINE_VIDEOS[currentModel] || {});
    const datasetSuiteValues = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
    // Merge ablation suites and baseline suites, then filter by dataset suites
    const combinedSuites = [...new Set([...ablationSuites, ...baselineSuites])].sort();
    if (datasetSuiteValues.length > 0) {
      // Always include all dataset suites (even if no ablation/baseline data yet)
      const merged = [...new Set([...combinedSuites.filter(s => datasetSuiteValues.includes(s)), ...datasetSuiteValues])];
      return merged.sort();
    }
    return combinedSuites.length > 0 ? combinedSuites : datasetSuiteValues;
  }, [currentModel, currentDataset, removalVideos]);

  // Get available tasks from ablation data for selected suite (primary) and BASELINE_VIDEOS (fallback)
  const availableTasks = useMemo(() => {
    // Get tasks from ablation data for the selected suite
    const ablationTasks = [...new Set(
      removalVideos
        .filter(v => v.suite === selectedSuite && v.task != null)
        .map(v => v.task as number)
    )].sort((a, b) => a - b);
    if (ablationTasks.length > 0) return ablationTasks;
    // Fallback to BASELINE_VIDEOS
    const modelBaselines = BASELINE_VIDEOS[currentModel] || {};
    const suiteBaselines = modelBaselines[selectedSuite] || {};
    return Object.keys(suiteBaselines).map(Number).sort((a, b) => a - b);
  }, [currentModel, selectedSuite, removalVideos]);

  // Reset selectedTask when availableTasks changes and current selection is invalid
  useEffect(() => {
    if (availableTasks.length > 0 && !availableTasks.includes(selectedTask)) {
      setSelectedTask(availableTasks[0]);
    }
  }, [availableTasks, selectedTask]);

  // Get current baseline video path — try hardcoded BASELINE_VIDEOS first, then
  // search for any baseline in the main video index via the API
  const [dynamicBaselinePath, setDynamicBaselinePath] = useState<string>('');
  const baselineVideoPath = useMemo(() => {
    // First try the hardcoded baselines (exact suite + task match)
    const modelBaselines = BASELINE_VIDEOS[currentModel] || {};
    const suiteBaselines = modelBaselines[selectedSuite] || {};
    if (suiteBaselines[selectedTask]) return suiteBaselines[selectedTask];
    // If dynamic baseline was fetched for this task, use it
    if (dynamicBaselinePath) return dynamicBaselinePath;
    // Don't fall back to a different task's video — show nothing instead
    return '';
  }, [currentModel, selectedSuite, selectedTask, dynamicBaselinePath]);

  // Fetch a baseline video dynamically when hardcoded one is missing
  useEffect(() => {
    setDynamicBaselinePath('');
    const modelBaselines = BASELINE_VIDEOS[currentModel] || {};
    const suiteBaselines = modelBaselines[selectedSuite] || {};
    if (suiteBaselines[selectedTask]) return; // Already have hardcoded baseline

    // Try to find a baseline from the video index
    const fetchBaseline = async () => {
      try {
        // The index uses shortened suite names for Pi0.5 (goal, spatial, object)
        // but full names for concept_ablation (libero_goal, etc.)
        const suiteVariants = [selectedSuite, selectedSuite.replace('libero_', '')];
        for (const suiteVar of suiteVariants) {
          const res = await fetch(
            `${API_BASE_URL}/api/vla/videos?model=${currentModel}&experiment_type=counterfactual&suite=${suiteVar}&limit=200`
          );
          if (!res.ok) continue;
          const data = await res.json();
          const videos = data.videos || [];
          // Find a baseline for this task
          // Check both subtype field AND filename (GR00T/SmolVLA have null subtype)
          const isBaselineVideo = (v: { subtype?: string; filename?: string; path?: string }) =>
            v.subtype === 'baseline' || (v.filename || v.path?.split('/').pop() || '').startsWith('baseline');
          // Only match baselines for the exact selected task — never fall back to a different task
          const baseline = videos.find(
            (v: { subtype?: string; filename?: string; path?: string; task?: number; success?: boolean }) =>
              isBaselineVideo(v) && v.task === selectedTask && v.success === true
          ) || videos.find(
            (v: { subtype?: string; filename?: string; path?: string; task?: number }) =>
              isBaselineVideo(v) && v.task === selectedTask
          );
          if (baseline?.path) {
            // Paths from the index are relative to the model's video dir.
            // Prepend model prefix so the video API resolves them correctly.
            const modelPrefix = `${currentModel}/`;
            const videoPath = baseline.path.startsWith(modelPrefix) || baseline.path.startsWith(`${currentModel}_`)
              ? baseline.path
              : `${modelPrefix}${baseline.path}`;
            setDynamicBaselinePath(videoPath);
            return;
          }
        }
        // Also try vision_perturbation baselines as another source
        for (const suiteVar of suiteVariants) {
          const res = await fetch(
            `${API_BASE_URL}/api/vla/videos?model=${currentModel}&experiment_type=vision_perturbation&suite=${suiteVar}&limit=200`
          );
          if (!res.ok) continue;
          const data = await res.json();
          const videos = data.videos || [];
          const isBaselineVP = (v: { subtype?: string; filename?: string; path?: string }) =>
            v.subtype === 'baseline' || (v.filename || v.path?.split('/').pop() || '').startsWith('baseline');
          // Only match baselines for the exact selected task
          const baseline = videos.find(
            (v: { subtype?: string; filename?: string; path?: string; task?: number }) =>
              isBaselineVP(v) && v.task === selectedTask
          );
          if (baseline?.path) {
            const modelPrefix = `${currentModel}/`;
            const videoPath = baseline.path.startsWith(modelPrefix) || baseline.path.startsWith(`${currentModel}_`)
              ? baseline.path
              : `${modelPrefix}${baseline.path}`;
            setDynamicBaselinePath(videoPath);
            return;
          }
        }
        // Also try baseline experiment type directly (SmolVLA has these)
        for (const suiteVar of suiteVariants) {
          const res = await fetch(
            `${API_BASE_URL}/api/vla/videos?model=${currentModel}&experiment_type=baseline&suite=${suiteVar}&limit=200`
          );
          if (!res.ok) continue;
          const data = await res.json();
          const videos = data.videos || [];
          // Only match baselines for the exact selected task
          const baseline = videos.find(
            (v: { task?: number }) => v.task === selectedTask
          );
          if (baseline?.path) {
            const modelPrefix = `${currentModel}/`;
            const videoPath = baseline.path.startsWith(modelPrefix) || baseline.path.startsWith(`${currentModel}_`)
              ? baseline.path
              : `${modelPrefix}${baseline.path}`;
            setDynamicBaselinePath(videoPath);
            return;
          }
        }
      } catch {
        // Ignore errors
      }
    };
    fetchBaseline();
  }, [currentModel, selectedSuite, selectedTask]);

  // Get current removal video from dynamic data — filter by suite, task, AND concept
  const removalVideo = useMemo(() => {
    const conceptLower = selectedConcept.toLowerCase();
    // Use exact concept match (equality), not loose .includes()
    const matchesConcept = (v: RemovalVideoData) =>
      v.concept?.toLowerCase() === conceptLower;
    // Best match: exact suite + task + concept
    const exactMatch = removalVideos.find(v =>
      matchesConcept(v) &&
      v.suite === selectedSuite &&
      v.task === selectedTask
    );
    if (exactMatch) {
      return {
        path: toVideoApiUrl(exactMatch.path),
        task: exactMatch.task ?? selectedTask,
        suite: exactMatch.suite ?? selectedSuite,
        success: exactMatch.success ?? false,
      };
    }
    // Fallback: suite + concept (any task)
    const suiteMatch = removalVideos.find(v =>
      matchesConcept(v) &&
      v.suite === selectedSuite
    );
    if (suiteMatch) {
      return {
        path: toVideoApiUrl(suiteMatch.path),
        task: suiteMatch.task ?? selectedTask,
        suite: suiteMatch.suite ?? selectedSuite,
        success: suiteMatch.success ?? false,
      };
    }
    // Last resort: concept only
    const conceptMatch = removalVideos.find(v => matchesConcept(v));
    if (conceptMatch) {
      return {
        path: toVideoApiUrl(conceptMatch.path),
        task: conceptMatch.task ?? selectedTask,
        suite: conceptMatch.suite ?? selectedSuite,
        success: conceptMatch.success ?? false,
      };
    }
    return null;
  }, [selectedConcept, removalVideos, selectedSuite, selectedTask]);

  // Fetch perturbation types on mount
  useEffect(() => {
    const fetchPerturbationTypes = async () => {
      try {
        setIsLoadingTypes(true);
        const response = await fetch(`${API_BASE_URL}/api/vla/perturbation_types`);

        if (!response.ok) {
          throw new Error(`Failed to fetch perturbation types: ${response.statusText}`);
        }

        const data: PerturbationTypesResponse = await response.json();

        if (data.perturbation_types && data.perturbation_types.length > 0) {
          // Merge API types (vision) with default non-vision types (object, counterfactual, injection)
          // API only returns vision perturbation types; keep defaults for other categories
          const apiTypeIds = new Set(data.perturbation_types.map(t => t.id));
          const nonVisionDefaults = DEFAULT_PERTURBATION_TYPES.filter(
            t => t.category !== 'vision' && !apiTypeIds.has(t.id)
          );
          setPerturbationTypes([...data.perturbation_types, ...nonVisionDefaults]);
          // Set default selection to first type if current selection is not in the list
          setSelectedPerturbation((currentSelection) => {
            const allTypeIds = [...data.perturbation_types, ...nonVisionDefaults].map(t => t.id);
            if (!allTypeIds.includes(currentSelection)) {
              return data.perturbation_types[0].id;
            }
            return currentSelection;
          });
        }
      } catch (err) {
        console.error("Error fetching perturbation types:", err);
        setError("Failed to load perturbation types from server. Using default types.");
        setShowError(true);
        // Keep using default types
      } finally {
        setIsLoadingTypes(false);
      }
    };

    fetchPerturbationTypes();
  }, []);

  // Fetch available videos and experiment types
  useEffect(() => {
    const fetchVideos = async () => {
      setIsLoadingVideos(true);
      try {
        // Fetch experiment types first to know what's available, then fetch
        // a manageable number of videos per experiment type so we don't miss any.
        // Without experiment_type filter, we'd get 10K+ videos and only slice 200.
        const knownExperimentTypes = ['vision_perturbation', 'counterfactual', 'cross_task', 'concept_ablation', 'cross_scene_injection', 'temporal_injection'];
        const experimentTypes = new Set<string>();
        const allVideos: {id: string; label: string; experiment_type?: string; suite?: string; success?: boolean; steps?: number}[] = [];

        // Fetch videos for each experiment type (up to 1000 each)
        const fetches = knownExperimentTypes.map(async (expType) => {
          try {
            const resp = await fetch(`${API_BASE_URL}/api/vla/videos?model=${currentModel}&experiment_type=${expType}&limit=1000`);
            if (resp.ok) {
              const data = await resp.json();
              const videos = (data.videos || []).slice(0, 1000);
              if (videos.length > 0) {
                experimentTypes.add(expType);
              }
              return videos;
            }
          } catch {
            // ignore individual failures
          }
          return [];
        });

        const results = await Promise.all(fetches);
        // Get valid suites for the current dataset to filter videos.
        // Use prefix matching: a video with suite "metaworld" matches dataset suites
        // like "metaworld_easy", "metaworld_medium", etc.
        const datasetSuiteValues = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
        const suiteMatchesDataset = (videoSuite: string): boolean => {
          if (datasetSuiteValues.length === 0) return true;
          if (!videoSuite) return true;
          // Exact match
          if (datasetSuiteValues.includes(videoSuite)) return true;
          // Video suite is a prefix of a dataset suite (e.g., "metaworld" matches "metaworld_easy")
          if (datasetSuiteValues.some(ds => ds.startsWith(videoSuite + '_') || ds === videoSuite)) return true;
          // Dataset suite is a prefix of the video suite
          if (datasetSuiteValues.some(ds => videoSuite.startsWith(ds + '_') || videoSuite.startsWith(ds))) return true;
          return false;
        };
        for (const videos of results) {
          for (const v of videos) {
            const suite = v.suite || '';
            // Filter by dataset suites if available
            if (!suiteMatchesDataset(suite)) {
              continue;
            }
            const filename = (v.path || v.filename || '').split('/').pop() || 'Unknown';
            const expType = v.experiment_type || '';
            const subtype = v.subtype || '';
            if (expType) experimentTypes.add(expType);
            const label = subtype
              ? `[${expType}/${suite}] ${subtype} - ${filename}`
              : `[${expType}/${suite}] ${filename}`;
            allVideos.push({
              id: v.path || v.id || '',
              label: label.length > 60 ? label.substring(0, 57) + '...' : label,
              experiment_type: expType,
              suite: suite,
              success: v.success,
              steps: v.steps,
            });
          }
        }

        setAvailableVideos(allVideos);
        setAvailableExperiments(experimentTypes);
        // Always select the first matching video when model changes or no video selected.
        // The old selectedVideo may belong to a different model.
        if (allVideos.length > 0) {
          const currentExpTypes = CATEGORY_TO_EXPERIMENT[selectedCategory] || [];
          const matchingVideos = currentExpTypes.length > 0
            ? allVideos.filter(v => v.experiment_type && currentExpTypes.includes(v.experiment_type))
            : allVideos;
          const firstMatch = matchingVideos.length > 0 ? matchingVideos[0] : allVideos[0];
          setSelectedVideo(firstMatch.id);
        }
      } catch (err) {
        console.error("Error fetching videos:", err);
        setError("Failed to load videos. Ensure the backend server is running (python run_backend.py --port 6006).");
        setShowError(true);
      } finally {
        setIsLoadingVideos(false);
      }
    };
    fetchVideos();
  }, [currentModel, currentDataset]);

  // Fetch available prompts when model changes
  useEffect(() => {
    const fetchPrompts = async () => {
      setIsLoadingPrompts(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/vla/prompts?model=${currentModel}`);
        if (response.ok) {
          const data = await response.json();
          setAvailablePrompts(data.prompts || data || []);
          // Reset selected prompt when model changes
          setSelectedPrompt(null);
        }
      } catch (err) {
        console.error("Error fetching prompts:", err);
        setAvailablePrompts([]);
      } finally {
        setIsLoadingPrompts(false);
      }
    };
    fetchPrompts();
  }, [currentModel]);

  // Fetch perturbation rollout videos when perturbation type changes
  useEffect(() => {
    if (mainTab !== 1) return;
    setRolloutSubtypeFilter("all"); // Reset subtype filter when perturbation type changes
    const fetchRolloutVideos = async () => {
      setIsLoadingRolloutVideos(true);
      try {
        const modelKey = currentModel;
        const res = await fetch(`${API_BASE_URL}/api/vla/perturbation_videos?type=${selectedPerturbation}&model=${modelKey}&limit=24`);
        if (res.ok) {
          const data = await res.json();
          setPerturbRolloutVideos(data.videos || []);
        } else {
          setPerturbRolloutVideos([]);
        }
      } catch { setPerturbRolloutVideos([]); }
      setIsLoadingRolloutVideos(false);
    };
    fetchRolloutVideos();
  }, [selectedPerturbation, currentModel, mainTab]);

  // Compute unique subtypes from rollout videos for filter chips
  const rolloutSubtypes = useMemo(() => {
    const subtypes = [...new Set(perturbRolloutVideos.map(v => v.subtype).filter(Boolean))].sort();
    return subtypes;
  }, [perturbRolloutVideos]);

  // Filter rollout videos by selected subtype
  const filteredRolloutVideos = useMemo(() => {
    if (rolloutSubtypeFilter === "all") return perturbRolloutVideos;
    return perturbRolloutVideos.filter(v => v.subtype === rolloutSubtypeFilter);
  }, [perturbRolloutVideos, rolloutSubtypeFilter]);

  // Filter perturbation types by category
  const filteredPerturbations = perturbationTypes.filter(p => p.category === selectedCategory || !p.category);

  // Get current perturbation config
  const currentPerturbation = perturbationTypes.find((p) => p.id === selectedPerturbation) || perturbationTypes[0];

  // Get the experiment types for current category (may be multiple)
  const currentExperimentTypes = CATEGORY_TO_EXPERIMENT[selectedCategory] || [];
  // Display label: use the specific perturbation's experiment type if available, else first category type
  const currentExperimentType = PERTURBATION_TO_EXPERIMENT[selectedPerturbation] || currentExperimentTypes[0] || '';

  // Filter videos by current category's experiment types
  const filteredVideos = useMemo(() => {
    if (currentExperimentTypes.length === 0) return availableVideos;
    return availableVideos.filter(v => v.experiment_type && currentExperimentTypes.includes(v.experiment_type));
  }, [availableVideos, currentExperimentTypes]);

  // Baseline-only videos for the dropdown (filter to baseline subtype/filename)
  const baselineVideos = useMemo(() => {
    return availableVideos.filter(v => {
      const filename = (v.id || '').split('/').pop() || '';
      const label = (v.label || '').toLowerCase();
      return v.experiment_type === 'baseline' ||
        label.includes('baseline') ||
        filename.startsWith('baseline') ||
        filename.includes('_baseline');
    });
  }, [availableVideos]);

  // Extract task number from the currently selected baseline video
  const selectedBaselineTask = useMemo((): number | null => {
    if (!selectedVideo) return null;
    const filename = selectedVideo.split('/').pop() || '';
    // Match patterns like task_3_seed42.mp4, task3_baseline.mp4, etc.
    const taskMatch = filename.match(/task[_]?(\d+)/);
    if (taskMatch) return parseInt(taskMatch[1], 10);
    // Also check the video metadata
    const vid = availableVideos.find(v => v.id === selectedVideo);
    if (vid?.label) {
      const labelMatch = vid.label.match(/task[_]?(\d+)/i);
      if (labelMatch) return parseInt(labelMatch[1], 10);
    }
    return null;
  }, [selectedVideo, availableVideos]);

  // Extract suite from the currently selected baseline video
  const selectedBaselineSuite = useMemo((): string => {
    const vid = availableVideos.find(v => v.id === selectedVideo);
    return vid?.suite || '';
  }, [selectedVideo, availableVideos]);

  // Count videos per experiment type
  const videoCountByExperiment = useMemo(() => {
    const counts: Record<string, number> = {};
    availableVideos.forEach(v => {
      if (v.experiment_type) {
        counts[v.experiment_type] = (counts[v.experiment_type] || 0) + 1;
      }
    });
    return counts;
  }, [availableVideos]);

  // Check if a perturbation type has available data for the selected baseline task
  const isPerturbationAvailable = useCallback((perturbationId: string) => {
    const experimentType = PERTURBATION_TO_EXPERIMENT[perturbationId];
    if (!experimentType) return true;
    // Must have the experiment type globally
    if (!availableExperiments.has(experimentType)) return false;
    // If we have a selected baseline task, check if this experiment type has data for that task
    if (selectedBaselineTask !== null) {
      return availableVideos.some(v =>
        v.experiment_type === experimentType &&
        ((v.id || '').includes(`task${selectedBaselineTask}`) ||
         (v.id || '').includes(`task_${selectedBaselineTask}`))
      );
    }
    return true;
  }, [availableExperiments, selectedBaselineTask, availableVideos]);

  // Reset selected video when category changes — always prefer a baseline
  useEffect(() => {
    if (baselineVideos.length > 0 && !baselineVideos.find(v => v.id === selectedVideo)) {
      setSelectedVideo(baselineVideos[0].id);
    }
  }, [selectedCategory, baselineVideos, selectedVideo]);

  // Handle image upload
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setPerturbedImage(null);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  // Handle perturbation selection
  const handlePerturbationSelect = useCallback((type: PerturbationType) => {
    setSelectedPerturbation(type);
    setPerturbedImage(null);
    setResult(null);
  }, []);

  // Apply perturbation via API
  const handleApplyPerturbation = useCallback(async () => {
    if (!uploadedImage) return;

    setIsApplyingPerturbation(true);
    setPerturbedImage(null);
    setError(null);

    try {
      // Extract base64 data from data URL (remove prefix like "data:image/png;base64,")
      const base64Data = uploadedImage.includes(",")
        ? uploadedImage.split(",")[1]
        : uploadedImage;

      const response = await fetch(`${API_BASE_URL}/api/vla/perturb`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Data,
          perturbation_type: selectedPerturbation,
          strength: strength,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const data: PerturbResponse = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Failed to apply perturbation");
      }

      // Set the perturbed image (add data URL prefix if not present)
      const perturbedImageData = data.perturbed_image.startsWith("data:")
        ? data.perturbed_image
        : `data:image/png;base64,${data.perturbed_image}`;

      setPerturbedImage(perturbedImageData);
    } catch (err) {
      console.error("Error applying perturbation:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to apply perturbation";
      setError(errorMessage);
      setShowError(true);
    } finally {
      setIsApplyingPerturbation(false);
    }
  }, [uploadedImage, selectedPerturbation, strength]);

  // Fetch real VP experiment results from backend (data-driven, no GPU needed)
  const handleRunInference = useCallback(async () => {
    setIsRunning(true);
    setResult(null);

    try {
      // Determine the perturbation type from the selected video or selected perturbation
      const selectedVideoData = filteredVideos.find(v => v.id === selectedVideo);
      const modelParam = currentModel;

      // Determine suite from selected video or default
      let suiteParam = '';
      if (selectedVideoData?.suite) {
        const s = selectedVideoData.suite;
        // Normalize: OFT uses libero_X, Pi0.5 uses short names
        suiteParam = s.startsWith('libero_') ? s : `libero_${s}`;
      }

      // Determine perturbation from the video subtype/filename or selected perturbation
      let pertParam = selectedPerturbation;
      if (selectedVideoData) {
        // Extract perturbation from path filename
        const fname = selectedVideoData.id.split('/').pop()?.replace('.mp4', '').replace('_combined', '') || '';
        // Pi0.5 format: libero_goal_task0_s42_blur_heavy_after_100
        const pi05Match = fname.match(/libero_\w+_task\d+_s\d+_(.*)/);
        // OFT format: task0_blur_light
        const oftMatch = fname.match(/task\d+_(.*)/);
        if (pi05Match) pertParam = pi05Match[1];
        else if (oftMatch) pertParam = oftMatch[1];
      }

      const url = `${API_BASE_URL}/api/vla/vp_experiment_results?model=${modelParam}&suite=${encodeURIComponent(suiteParam)}&perturbation=${encodeURIComponent(pertParam)}`;
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        const r = data.result;
        if (r) {
          setResult({
            perturbation: r.perturbation || pertParam,
            success_rate: r.success_rate ?? 0,
            baseline_success_rate: r.baseline_success_rate ?? null,
            delta_success_rate: r.delta_success_rate ?? null,
            avg_n_steps: r.avg_n_steps ?? null,
            baseline_avg_n_steps: r.baseline_avg_n_steps ?? null,
            delta_n_steps: r.delta_n_steps ?? null,
            n_episodes: r.n_episodes ?? 0,
            video_path: r.video_path ?? null,
            baseline_video_path: r.baseline_video_path ?? null,
          });
        } else {
          // No matching data — show a zero-result with available alternatives inline
          const available = (data.available_perturbations || []) as string[];
          setResult({
            perturbation: pertParam,
            success_rate: -1,
            baseline_success_rate: null,
            delta_success_rate: null,
            avg_n_steps: null,
            baseline_avg_n_steps: null,
            delta_n_steps: null,
            n_episodes: 0,
            video_path: null,
            baseline_video_path: null,
          });
          if (available.length > 0) {
            setError(`No data for "${pertParam}" in this suite. Available perturbations: ${available.slice(0, 8).join(', ')}${available.length > 8 ? '...' : ''}`);
          } else {
            setError(`No VP data for "${pertParam}" in suite "${suiteParam}".`);
          }
          // Show inline, not as a toast
        }
      } else {
        const errData = await resp.json().catch(() => ({}));
        const errMsg = errData.error || `No vision perturbation results available for ${getModelDisplayName(currentModel)}`;
        setError(errMsg);
        // Show inline, not as a toast
      }
    } catch (err) {
      console.error('VP results fetch error:', err);
      setError('Error fetching VP experiment results');
      setShowError(true);
    } finally {
      setIsRunning(false);
    }
  }, [selectedVideo, filteredVideos, selectedPerturbation, currentModel]);

  // Build the video URL for the hidden video element used for frame extraction
  const videoSrcUrl = useMemo(() => {
    if (!selectedVideo) return '';
    return toVideoApiUrl(selectedVideo.startsWith('/videos/') ? selectedVideo : `/videos/${selectedVideo}`);
  }, [selectedVideo]);

  // Handle video metadata loaded — set slider range dynamically
  const handleVideoMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video && isFinite(video.duration) && video.duration > 0) {
      setVideoDuration(video.duration);
      // Reset frame position if beyond duration
      if (selectedFrame > video.duration) {
        setSelectedFrame(0);
      }
    }
  }, [selectedFrame]);

  // Extract frame client-side using canvas (works with Tigris-served videos)
  const handleExtractFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !selectedVideo) return;

    setIsExtractingFrame(true);
    setError(null);

    try {
      // Seek the video to the selected time
      video.currentTime = selectedFrame;

      // Wait for the seek to complete
      await new Promise<void>((resolve, reject) => {
        const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve(); };
        const onError = () => { video.removeEventListener('error', onError); reject(new Error('Video seek failed')); };
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', onError, { once: true });
        // Timeout after 5 seconds
        setTimeout(() => { video.removeEventListener('seeked', onSeeked); reject(new Error('Video seek timed out')); }, 5000);
      });

      // Draw the frame to canvas
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Failed to get canvas context');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Export as data URL
      const frameData = canvas.toDataURL('image/png');
      setUploadedImage(frameData);
      setPerturbedImage(null);
      setResult(null);
    } catch (err) {
      console.error("Error extracting frame:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to extract frame from video";
      setError(errorMessage);
      setShowError(true);
    } finally {
      setIsExtractingFrame(false);
    }
  }, [selectedVideo, selectedFrame]);

  // Clear all
  const handleClear = useCallback(() => {
    setUploadedImage(null);
    setPerturbedImage(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  // Close error snackbar
  const handleCloseError = useCallback(() => {
    setShowError(false);
  }, []);

  return (
    <Paper className="h-full flex flex-col rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className="h-10 flex items-center px-4 bg-[#0a1628] rounded-t-lg border-b border-slate-700">
        <Typography variant="subtitle2" sx={{ color: "white", fontWeight: 600 }}>
          Pen Testing
        </Typography>
        <Chip
          label={currentModel.toUpperCase()}
          size="small"
          sx={{
            ml: 2,
            height: 18,
            fontSize: "9px",
            bgcolor: getModelChipColor(currentModel),
            color: "white",
          }}
        />
        <Box sx={{ ml: "auto", display: "flex", gap: 1 }}>
          <Button
            size="small"
            onClick={handleClear}
            sx={{
              fontSize: "10px",
              color: "#94a3b8",
              "&:hover": { color: "white", bgcolor: "rgba(255,255,255,0.1)" },
            }}
          >
            Clear
          </Button>
        </Box>
      </div>

      {/* Content */}
      <Box className="flex-1 overflow-auto bg-slate-900 p-4">
        {/* Main Tabs: Baseline vs Removed | Perturbation Testing */}
        <Tabs
          value={mainTab}
          onChange={(_, v) => setMainTab(v)}
          sx={{
            mb: 3,
            minHeight: 36,
            '& .MuiTab-root': {
              minHeight: 36,
              fontSize: '12px',
              fontWeight: 600,
              color: '#64748b',
              '&.Mui-selected': { color: '#ef4444' },
            },
            '& .MuiTabs-indicator': { backgroundColor: '#ef4444' },
          }}
        >
          <Tab label="Baseline vs Removed" />
          <Tab label="Perturbation Testing" />
          <Tab label="Grid Ablation" />
          <Tab label="Counterfactual" />
          <Tab label="Injection" />
        </Tabs>

        {mainTab === 0 && (
          /* Baseline vs Removed Comparison Tab */
          <Box>
            <Typography variant="body2" sx={{ color: '#94a3b8', mb: 2 }}>
              Compare baseline rollouts with concept-ablated (feature removal) rollouts. Select a concept to see
              how removing its SAE features affects task completion.
              {isLoadingRemovalVideos && ' Loading ablation data...'}
              {!isLoadingRemovalVideos && removalVideos.length > 0 && (
                <span style={{ color: '#64748b' }}> ({removalVideos.length} ablation videos loaded)</span>
              )}
            </Typography>

            {/* Task Description Header */}
            {(() => {
              const taskDesc = getTaskDescriptionForSuiteTask(selectedSuite, selectedTask);
              return taskDesc ? (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: '#1e293b',
                    borderRadius: 1,
                    border: '1px solid #334155',
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="body2" sx={{ color: '#e2e8f0', fontWeight: 600 }}>
                    Task {selectedTask}: {taskDesc}
                  </Typography>
                </Box>
              ) : null;
            })()}

            {/* Controls Row */}
            <Box
              sx={{
                display: 'flex',
                gap: 2,
                mb: 3,
                p: 2,
                bgcolor: '#1e293b',
                borderRadius: 1,
                border: '1px solid #334155',
                flexWrap: 'wrap',
              }}
            >
              {/* Suite Selection */}
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel sx={{ color: '#64748b', fontSize: '12px' }}>Suite</InputLabel>
                <Select
                  value={selectedSuite}
                  label="Suite"
                  onChange={(e) => {
                    const newSuite = e.target.value;
                    setSelectedSuite(newSuite);
                    // Reset task when suite changes — derive from ablation data first
                    const ablationTasks = [...new Set(
                      removalVideos
                        .filter(v => v.suite === newSuite && v.task != null)
                        .map(v => v.task as number)
                    )].sort((a, b) => a - b);
                    if (ablationTasks.length > 0) {
                      setSelectedTask(ablationTasks[0]);
                    } else {
                      const newSuiteTasks = Object.keys(
                        (BASELINE_VIDEOS[currentModel] || BASELINE_VIDEOS.pi05)[newSuite] || {}
                      ).map(Number);
                      if (newSuiteTasks.length > 0) {
                        setSelectedTask(newSuiteTasks[0]);
                      }
                    }
                    // Reset concept to one available in the new suite
                    const suiteConcepts = [...new Set(
                      removalVideos.filter(v => v.suite === newSuite).map(v => v.concept).filter(Boolean)
                    )] as string[];
                    if (suiteConcepts.length > 0 && !suiteConcepts.includes(selectedConcept)) {
                      setSelectedConcept(suiteConcepts.sort()[0]);
                    }
                  }}
                  sx={{
                    color: '#e2e8f0',
                    fontSize: '12px',
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#475569' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                  }}
                >
                  {availableSuites.map((suite) => (
                    <MenuItem key={suite} value={suite}>
                      {suite.replace('_', ' ').replace('libero', 'Libero')}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Task Selection */}
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel sx={{ color: '#64748b', fontSize: '12px' }}>Task</InputLabel>
                <Select
                  value={availableTasks.includes(selectedTask) ? selectedTask : (availableTasks[0] ?? '')}
                  label="Task"
                  onChange={(e) => setSelectedTask(Number(e.target.value))}
                  sx={{
                    color: '#e2e8f0',
                    fontSize: '12px',
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                    '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#475569' },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                  }}
                >
                  {availableTasks.map((task) => {
                    const desc = getTaskDescriptionForSuiteTask(selectedSuite, task);
                    return (
                    <MenuItem key={task} value={task}>
                      Task {task}{desc ? `: ${desc}` : ''}
                    </MenuItem>
                    );
                  })}
                </Select>
              </FormControl>

              {/* Concept Selection */}
              {(() => {
                // Check if per-concept videos exist for this model
                const hasConceptVideoMatches = removalVideos.some(v =>
                  v.concept && !v.concept.startsWith('disc_') && v.concept.includes('/')
                );
                const isPerConceptModel = currentModel === 'pi05' || currentModel === 'openvla' || hasConceptVideoMatches;

                if (!isPerConceptModel && !isLoadingRemovalVideos) {
                  return (
                    <Box sx={{ display: 'flex', alignItems: 'center', px: 1 }}>
                      <Typography variant="caption" sx={{ color: '#64748b', fontSize: '11px', fontStyle: 'italic' }}>
                        Per-concept selection: see experiment results below
                      </Typography>
                    </Box>
                  );
                }

                return (
                  <FormControl size="small" sx={{ minWidth: 150 }}>
                    <InputLabel sx={{ color: '#64748b', fontSize: '12px' }}>Removed Concept</InputLabel>
                    <Select
                      value={selectedConcept}
                      label="Removed Concept"
                      onChange={(e) => setSelectedConcept(e.target.value)}
                      sx={{
                        color: '#e2e8f0',
                        fontSize: '12px',
                        '& .MuiOutlinedInput-notchedOutline': { borderColor: '#334155' },
                        '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#475569' },
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#ef4444' },
                      }}
                    >
                      {(() => {
                        // Filter concepts to those available for the selected suite
                        const suiteConcepts = removalConcepts.filter(c =>
                          removalVideos.some(v => v.concept === c && v.suite === selectedSuite)
                        );
                        const displayConcepts = suiteConcepts.length > 0
                          ? suiteConcepts
                          : removalConcepts;
                        return displayConcepts.map((concept) => {
                          const count = removalVideos.filter(
                            v => v.concept === concept && v.suite === selectedSuite
                          ).length;
                          return (
                            <MenuItem key={concept} value={concept}>
                              {concept.replace('/', ' / ').toUpperCase()} features{count > 0 ? ` (${count})` : ''}
                            </MenuItem>
                          );
                        });
                      })()}
                    </Select>
                  </FormControl>
                );
              })()}
            </Box>

            {/* Video Comparison Grid — per-concept comparison for Pi0.5/OFT/SmolVLA, fallback for others */}
            {(() => {
              // Determine if per-concept video comparison is available for this model.
              // Pi0.5 and OFT have per-concept ablation videos with concept info in filenames.
              // SmolVLA has concept_ablation videos with concepts extractable from path structure.
              // X-VLA has NO concept_ablation videos. GR00T has SAE-level (not concept-level) videos.
              const hasConceptVideos = removalVideo?.path != null;
              const modelHasConceptAblationVideos = removalVideos.some(v =>
                v.concept && !v.concept.startsWith('disc_') && v.concept.includes('/')
              );
              const showPerConceptComparison = hasConceptVideos || modelHasConceptAblationVideos || (currentModel === 'pi05' || currentModel === 'openvla');

              if (showPerConceptComparison) {
                // Standard per-concept video comparison (Pi0.5, OFT, SmolVLA with matching concept)
                return (
                  <>
                    <Grid container spacing={3}>
                      {/* Baseline Video */}
                      <Grid size={{ xs: 12, md: 6 }}>
                        <Card sx={{ bgcolor: '#0f172a', border: '1px solid #334155' }}>
                          <CardContent sx={{ p: 2 }}>
                            <Box display="flex" alignItems="center" gap={1} mb={2}>
                              <PlayCircleOutlineIcon sx={{ color: '#22c55e', fontSize: 20 }} />
                              <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
                                Baseline (No Ablation)
                              </Typography>
                              <Chip
                                label="Success"
                                size="small"
                                sx={{
                                  height: 18,
                                  fontSize: '9px',
                                  bgcolor: '#166534',
                                  color: '#86efac',
                                  ml: 'auto',
                                }}
                              />
                            </Box>
                            <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mb: 1 }}>
                              {selectedSuite.replace(/_/g, ' ')} - Task {selectedTask}: {getTaskDescriptionForSuiteTask(selectedSuite, selectedTask) || 'Original policy behavior'}
                            </Typography>
                            {baselineVideoPath ? (
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
                                  key={`baseline-${selectedSuite}-${selectedTask}`}
                                  src={toVideoApiUrl(baselineVideoPath)}
                                  controls
                                  autoPlay
                                  loop
                                  muted
                                  style={{
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain',
                                  }}
                                />
                              </Box>
                            ) : (
                              <Box
                                sx={{
                                  width: '100%',
                                  aspectRatio: '4/3',
                                  bgcolor: '#1e293b',
                                  borderRadius: 1,
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                }}
                              >
                                <Typography variant="caption" sx={{ color: '#64748b' }}>
                                  No baseline video available for this task
                                </Typography>
                              </Box>
                            )}
                          </CardContent>
                        </Card>
                      </Grid>

                      {/* Removed/Ablated Video */}
                      <Grid size={{ xs: 12, md: 6 }}>
                        <Card sx={{ bgcolor: '#0f172a', border: '1px solid #334155' }}>
                          <CardContent sx={{ p: 2 }}>
                            <Box display="flex" alignItems="center" gap={1} mb={2}>
                              <PlayCircleOutlineIcon sx={{ color: '#ef4444', fontSize: 20 }} />
                              <Typography variant="subtitle2" sx={{ color: 'white', fontWeight: 600 }}>
                                {selectedConcept.toUpperCase()} Features Removed
                              </Typography>
                              <Chip
                                label={removalVideo?.success ? 'Success' : 'Failure'}
                                size="small"
                                sx={{
                                  height: 18,
                                  fontSize: '9px',
                                  bgcolor: removalVideo?.success ? '#166534' : '#991b1b',
                                  color: removalVideo?.success ? '#86efac' : '#fca5a5',
                                  ml: 'auto',
                                }}
                              />
                            </Box>
                            <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mb: 1 }}>
                              Task {removalVideo?.task || selectedTask}: {getTaskDescriptionForSuiteTask(selectedSuite, removalVideo?.task || selectedTask) || `${selectedConcept.toUpperCase()} concept features zeroed`}
                            </Typography>
                            {removalVideo?.path ? (
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
                                  key={`ablation-${selectedSuite}-${selectedTask}-${selectedConcept}`}
                                  src={removalVideo.path}
                                  controls
                                  autoPlay
                                  loop
                                  muted
                                  style={{
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain',
                                  }}
                                />
                              </Box>
                            ) : (
                              <Box
                                sx={{
                                  width: '100%',
                                  aspectRatio: '4/3',
                                  bgcolor: '#1e293b',
                                  borderRadius: 1,
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                }}
                              >
                                {isLoadingRemovalVideos ? (
                                  <CircularProgress size={20} sx={{ color: '#ef4444' }} />
                                ) : (
                                  <Typography variant="caption" sx={{ color: '#64748b' }}>
                                    No ablation video available for {selectedConcept.toUpperCase()} ({getModelDisplayName(currentModel)})
                                  </Typography>
                                )}
                              </Box>
                            )}
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>

                    {/* Key Finding Box */}
                    <Box
                      sx={{
                        mt: 3,
                        p: 2,
                        bgcolor: '#1e293b',
                        borderRadius: 1,
                        border: '1px solid #334155',
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ color: '#ef4444', fontWeight: 600, mb: 1 }}>
                        Ablation Effect — {getModelDisplayName(currentModel)}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        {currentModel === 'openvla'
                          ? `Removing ${selectedConcept.toUpperCase()} concept features shows sparse effects: 91.6% of task-concept pairs have zero impact. The 4096-dim hidden space distributes representations redundantly.`
                          : currentModel === 'pi05'
                          ? `Removing ${selectedConcept.toUpperCase()} concept features (top 30/64 by activation) ${
                              removalVideo?.success === false
                                ? 'causes catastrophic task failure (-60 to -100pp). The 1024-dim hidden space concentrates critical information.'
                                : 'may affect task performance. Compare the videos to see behavioral differences.'
                            }`
                          : `Removing ${selectedConcept.toUpperCase()} concept features. Compare the baseline and ablated rollouts to see how feature removal affects task completion for ${getModelDisplayName(currentModel)}.`
                        }
                      </Typography>
                      {ablationSummaryNote && (
                        <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mt: 1, fontStyle: 'italic' }}>
                          {ablationSummaryNote}
                        </Typography>
                      )}
                    </Box>
                  </>
                );
              }

              // ---- Fallback UI for models without per-concept ablation videos ----
              // (X-VLA, GR00T, or any model where concept video matching fails)
              return (
                <>
                  {/* Baseline video (if available) */}
                  {baselineVideoPath && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="subtitle2" sx={{ color: '#e2e8f0', fontWeight: 600, mb: 1 }}>
                        Baseline Rollout
                      </Typography>
                      <Card sx={{ bgcolor: '#0f172a', border: '1px solid #334155', maxWidth: 500 }}>
                        <CardContent sx={{ p: 2 }}>
                          <Box display="flex" alignItems="center" gap={1} mb={1}>
                            <PlayCircleOutlineIcon sx={{ color: '#22c55e', fontSize: 18 }} />
                            <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                              {selectedSuite.replace(/_/g, ' ')} - Task {selectedTask}: {getTaskDescriptionForSuiteTask(selectedSuite, selectedTask) || 'Baseline behavior'}
                            </Typography>
                          </Box>
                          <Box sx={{ width: '100%', aspectRatio: '4/3', bgcolor: '#000', borderRadius: 1, overflow: 'hidden' }}>
                            <video
                              key={`baseline-fallback-${selectedSuite}-${selectedTask}`}
                              src={toVideoApiUrl(baselineVideoPath)}
                              controls
                              autoPlay
                              loop
                              muted
                              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    </Box>
                  )}

                  {/* Info message directing to other tabs */}
                  <Alert
                    severity="info"
                    sx={{
                      bgcolor: '#1e293b',
                      color: '#94a3b8',
                      border: '1px solid #334155',
                      '& .MuiAlert-icon': { color: '#3b82f6' },
                    }}
                  >
                    Per-concept ablation video comparison is available for models with ablation videos (Pi0.5, OpenVLA-OFT).
                    Use the <strong>Grid Ablation</strong>, <strong>Counterfactual</strong>, or <strong>Injection</strong> tabs
                    for {getModelDisplayName(currentModel)} analysis.
                  </Alert>
                </>
              );
            })()}
          </Box>
        )}

        {mainTab === 1 && (
          /* Perturbation Testing Tab */
          <Box>
            {/* Usage Guide */}
            <Box
              sx={{
                mb: 3,
                p: 2,
                bgcolor: '#1e293b',
                borderRadius: 1,
                border: '1px solid #334155',
              }}
            >
              <Typography variant="subtitle2" sx={{ color: '#ef4444', mb: 1, fontWeight: 600 }}>
                How to Use Perturbation Testing
              </Typography>
              <Box component="ol" sx={{ color: '#94a3b8', fontSize: '12px', pl: 2, m: 0 }}>
                <li style={{ marginBottom: '4px' }}><strong>Select Image Source:</strong> Upload an image directly or extract a frame from a demo video</li>
                <li style={{ marginBottom: '4px' }}><strong>Choose Perturbation:</strong> Select a perturbation type (noise, blur, crop, etc.) and adjust strength</li>
                <li style={{ marginBottom: '4px' }}><strong>Apply &amp; Compare:</strong> Click &quot;Apply Perturbation&quot; to see the original vs perturbed image side-by-side</li>
                <li><strong>Experiment Results:</strong> Click &quot;Show Experiment Results&quot; to see real success rates and metrics from robot rollout experiments</li>
              </Box>
            </Box>

        <Grid container spacing={3}>
          {/* Left Column: Image Source */}
          <Grid size={{ xs: 12, md: 4 }}>
            <Box sx={{ mb: 3 }}>
              <Typography
                variant="subtitle2"
                sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
              >
                Image Source
              </Typography>
              <Tabs
                value={sourceTab}
                onChange={(_, v) => setSourceTab(v)}
                sx={{
                  minHeight: 32,
                  mb: 2,
                  "& .MuiTab-root": {
                    minHeight: 32,
                    fontSize: "11px",
                    color: "#64748b",
                    "&.Mui-selected": { color: "#ef4444" },
                  },
                  "& .MuiTabs-indicator": { bgcolor: "#ef4444" },
                }}
              >
                <Tab label="Upload Image" />
                <Tab label="From Video" />
              </Tabs>

              {sourceTab === 0 ? (
                <Box>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    style={{ display: "none" }}
                  />
                  <Button
                    variant="outlined"
                    fullWidth
                    onClick={() => fileInputRef.current?.click()}
                    sx={{
                      borderColor: "#334155",
                      color: "#94a3b8",
                      borderStyle: "dashed",
                      py: 2,
                      "&:hover": { borderColor: "#ef4444", color: "#ef4444" },
                    }}
                  >
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                        <polyline points="17,8 12,3 7,8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                      <Typography variant="caption" sx={{ mt: 1 }}>
                        Click to upload image
                      </Typography>
                    </Box>
                  </Button>
                </Box>
              ) : (
                <Box>
                  {/* Baseline video count badge */}
                  <Box sx={{ mb: 1.5, display: "flex", alignItems: "center", gap: 1 }}>
                    <Chip
                      label="Baseline Videos"
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: "10px",
                        bgcolor: baselineVideos.length > 0 ? "#166534" : "#374151",
                        color: baselineVideos.length > 0 ? "#86efac" : "#9ca3af",
                      }}
                    />
                    <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px" }}>
                      {baselineVideos.length} baselines{selectedBaselineTask !== null ? ` | Task ${selectedBaselineTask} selected` : ''}
                    </Typography>
                  </Box>
                  <Autocomplete
                    value={baselineVideos.find(v => v.id === selectedVideo) || null}
                    onChange={(_, newValue) => setSelectedVideo(newValue?.id || "")}
                    options={baselineVideos}
                    getOptionLabel={(option) => option.label}
                    isOptionEqualToValue={(option, value) => option.id === value.id}
                    loading={isLoadingVideos}
                    size="small"
                    sx={{ mb: 2 }}
                    noOptionsText="No baseline videos available"
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Select Video"
                        placeholder="Search videos..."
                        sx={{
                          "& .MuiInputLabel-root": { color: "#64748b", fontSize: "12px" },
                          "& .MuiOutlinedInput-root": {
                            color: "#e2e8f0",
                            fontSize: "12px",
                            "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
                            "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
                            "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
                          },
                        }}
                      />
                    )}
                    ListboxProps={{
                      sx: {
                        bgcolor: "#1e293b",
                        "& .MuiAutocomplete-option": {
                          color: "#e2e8f0",
                          fontSize: "12px",
                          "&:hover": { bgcolor: "#334155" },
                          "&[aria-selected='true']": { bgcolor: "#475569" },
                        },
                      },
                    }}
                  />
                  {/* Hidden video + canvas for client-side frame extraction */}
                  <video
                    ref={videoRef}
                    src={videoSrcUrl}
                    crossOrigin="anonymous"
                    preload="metadata"
                    onLoadedMetadata={handleVideoMetadata}
                    onError={() => setVideoDuration(-1)}
                    style={{ display: 'none' }}
                  />
                  <canvas ref={canvasRef} style={{ display: 'none' }} />
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 1 }}>
                      Time: {selectedFrame.toFixed(1)}s {videoDuration > 0 ? `/ ${videoDuration.toFixed(1)}s` : videoDuration === -1 ? '(video unavailable)' : '(loading...)'}
                    </Typography>
                    <Slider
                      value={selectedFrame}
                      onChange={(_, v) => setSelectedFrame(v as number)}
                      min={0}
                      max={videoDuration > 0 ? videoDuration : 10}
                      step={0.1}
                      disabled={videoDuration === 0}
                      sx={{
                        color: "#ef4444",
                        "& .MuiSlider-thumb": { width: 12, height: 12 },
                      }}
                    />
                  </Box>
                  <Button
                    variant="contained"
                    fullWidth
                    disabled={!selectedVideo || isExtractingFrame}
                    onClick={handleExtractFrame}
                    sx={{
                      bgcolor: "#1e293b",
                      color: "#e2e8f0",
                      "&:hover": { bgcolor: "#334155" },
                      "&.Mui-disabled": { bgcolor: "#0f172a", color: "#475569" },
                    }}
                  >
                    {isExtractingFrame ? (
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <CircularProgress size={16} sx={{ color: "#94a3b8" }} />
                        Extracting...
                      </Box>
                    ) : (
                      "Extract Frame"
                    )}
                  </Button>
                </Box>
              )}
            </Box>

            {/* Category Tabs */}
            <Box sx={{ mb: 2 }}>
              <Tabs
                value={selectedCategory}
                onChange={(_, v) => setSelectedCategory(v as PerturbationCategory)}
                variant="fullWidth"
                sx={{
                  minHeight: 32,
                  mb: 1,
                  "& .MuiTab-root": {
                    minHeight: 32,
                    py: 0.5,
                    fontSize: "10px",
                    fontWeight: 600,
                    color: "#64748b",
                    "&.Mui-selected": { color: "#ef4444" },
                  },
                  "& .MuiTabs-indicator": { backgroundColor: "#ef4444" },
                }}
              >
                {(Object.keys(CATEGORY_LABELS) as PerturbationCategory[]).map((cat) => (
                  <Tab
                    key={cat}
                    value={cat}
                    label={CATEGORY_LABELS[cat].label}
                    sx={{ textTransform: "none" }}
                  />
                ))}
              </Tabs>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px" }}>
                {CATEGORY_LABELS[selectedCategory].description}
              </Typography>
            </Box>

            {/* Perturbation Types Grid */}
            <Box sx={{ mb: 3 }}>
              <Typography
                variant="subtitle2"
                sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
              >
                Perturbation Type
                {isLoadingTypes && (
                  <CircularProgress size={10} sx={{ ml: 1, color: "#94a3b8" }} />
                )}
              </Typography>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: 1,
                }}
              >
                {filteredPerturbations.map((pType) => {
                  const isAvailable = isPerturbationAvailable(pType.id);
                  return (
                    <Button
                      key={pType.id}
                      variant={selectedPerturbation === pType.id ? "contained" : "outlined"}
                      onClick={() => handlePerturbationSelect(pType.id)}
                      disabled={!isAvailable}
                      sx={{
                        flexDirection: "column",
                        py: 1.5,
                        px: 1,
                        minWidth: 0,
                        borderColor: selectedPerturbation === pType.id ? "#ef4444" : "#334155",
                        bgcolor: selectedPerturbation === pType.id ? "#ef4444" : "transparent",
                        color: selectedPerturbation === pType.id ? "white" : "#94a3b8",
                        opacity: isAvailable ? 1 : 0.4,
                        "&:hover": {
                          borderColor: "#ef4444",
                          bgcolor: selectedPerturbation === pType.id ? "#dc2626" : "rgba(239, 68, 68, 0.1)",
                        },
                        "&.Mui-disabled": {
                          borderColor: "#1e293b",
                          color: "#475569",
                        },
                      }}
                    >
                      <PerturbationIcon type={pType.id} />
                      <Typography variant="caption" sx={{ mt: 0.5, fontSize: "9px" }}>
                        {pType.label}
                      </Typography>
                    </Button>
                  );
                })}
              </Box>
            </Box>

            {/* Prompt/Task Selector for Counterfactual and Injection categories */}
            {(selectedCategory === "counterfactual" || selectedCategory === "injection") && (
              <Box sx={{ mb: 3 }}>
                <Typography
                  variant="subtitle2"
                  sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
                >
                  {selectedCategory === "injection" ? "Injection Target Prompt" : "Target Prompt"}
                  {isLoadingPrompts && (
                    <CircularProgress size={10} sx={{ ml: 1, color: "#94a3b8" }} />
                  )}
                </Typography>
                <Autocomplete
                  value={selectedPrompt}
                  onChange={(_, newValue) => setSelectedPrompt(newValue)}
                  options={availablePrompts}
                  loading={isLoadingPrompts}
                  size="small"
                  freeSolo
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      placeholder="Search or type a prompt..."
                      sx={{
                        "& .MuiInputLabel-root": { color: "#64748b", fontSize: "12px" },
                        "& .MuiOutlinedInput-root": {
                          color: "#e2e8f0",
                          fontSize: "11px",
                          "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
                          "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
                          "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
                        },
                      }}
                      InputProps={{
                        ...params.InputProps,
                        endAdornment: (
                          <>
                            {isLoadingPrompts ? <CircularProgress color="inherit" size={16} /> : null}
                            {params.InputProps.endAdornment}
                          </>
                        ),
                      }}
                    />
                  )}
                  ListboxProps={{
                    sx: {
                      bgcolor: "#1e293b",
                      maxHeight: 200,
                      "& .MuiAutocomplete-option": {
                        color: "#e2e8f0",
                        fontSize: "11px",
                        py: 1,
                        "&:hover": { bgcolor: "#334155" },
                        "&[aria-selected='true']": { bgcolor: "#475569" },
                      },
                    },
                  }}
                />
                <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px", mt: 0.5, display: "block" }}>
                  {selectedCategory === "injection"
                    ? "Select a prompt to inject activations from"
                    : "Select a prompt for counterfactual comparison"}
                </Typography>
              </Box>
            )}

            {/* Strength Slider */}
            {currentPerturbation.hasStrength && (
              <Box sx={{ mb: 3 }}>
                <Typography
                  variant="subtitle2"
                  sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
                >
                  Strength: {strength}%
                </Typography>
                <Slider
                  value={strength}
                  onChange={(_, v) => setStrength(v as number)}
                  min={0}
                  max={100}
                  sx={{
                    color: "#ef4444",
                    "& .MuiSlider-thumb": { width: 16, height: 16 },
                    "& .MuiSlider-track": { height: 6 },
                    "& .MuiSlider-rail": { height: 6, bgcolor: "#334155" },
                  }}
                />
              </Box>
            )}

            {/* Apply Button */}
            <Button
              variant="contained"
              fullWidth
              disabled={!uploadedImage || isApplyingPerturbation}
              onClick={handleApplyPerturbation}
              sx={{
                mb: 2,
                bgcolor: "#1e293b",
                color: "#e2e8f0",
                "&:hover": { bgcolor: "#334155" },
                "&.Mui-disabled": { bgcolor: "#0f172a", color: "#475569" },
              }}
            >
              {isApplyingPerturbation ? (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <CircularProgress size={16} sx={{ color: "#94a3b8" }} />
                  Applying {currentPerturbation?.label || selectedPerturbation}...
                </Box>
              ) : (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <PerturbationIcon type={selectedPerturbation} />
                  Apply {currentPerturbation?.label || selectedPerturbation}
                  {currentPerturbation?.hasStrength && (
                    <Chip
                      label={`${strength}%`}
                      size="small"
                      sx={{
                        height: 16,
                        fontSize: "9px",
                        fontWeight: 600,
                        bgcolor: "rgba(255,255,255,0.15)",
                        color: "inherit",
                        '& .MuiChip-label': { px: 0.5 },
                      }}
                    />
                  )}
                </Box>
              )}
            </Button>
          </Grid>

          {/* Center: Image Comparison */}
          <Grid size={{ xs: 12, md: 5 }}>
            <Typography
              variant="subtitle2"
              sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
            >
              Image Comparison
            </Typography>
            <Grid container spacing={2}>
              {/* Original Image */}
              <Grid size={{ xs: 6 }}>
                <Box
                  sx={{
                    position: "relative",
                    borderRadius: 1,
                    overflow: "hidden",
                    border: "1px solid #334155",
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      position: "absolute",
                      top: 8,
                      left: 8,
                      bgcolor: "rgba(0,0,0,0.7)",
                      color: "#94a3b8",
                      px: 1,
                      py: 0.25,
                      borderRadius: 0.5,
                      fontSize: "10px",
                      zIndex: 1,
                    }}
                  >
                    Original
                  </Typography>
                  {uploadedImage ? (
                    <img
                      src={uploadedImage}
                      alt="Original"
                      style={{ width: "100%", height: "auto", display: "block" }}
                    />
                  ) : (
                    <PlaceholderImage label="Upload an image" />
                  )}
                </Box>
              </Grid>

              {/* Perturbed Image */}
              <Grid size={{ xs: 6 }}>
                <Box
                  sx={{
                    position: "relative",
                    borderRadius: 1,
                    overflow: "hidden",
                    border: "1px solid #334155",
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      position: "absolute",
                      top: 8,
                      left: 8,
                      bgcolor: "rgba(239, 68, 68, 0.8)",
                      color: "white",
                      px: 1,
                      py: 0.25,
                      borderRadius: 0.5,
                      fontSize: "10px",
                      zIndex: 1,
                    }}
                  >
                    {currentPerturbation?.label || selectedPerturbation} ({strength}%)
                  </Typography>
                  {isApplyingPerturbation ? (
                    <Box
                      sx={{
                        width: "100%",
                        minHeight: 200,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        bgcolor: "#1a1a2e",
                      }}
                    >
                      <CircularProgress size={32} sx={{ color: "#ef4444" }} />
                      <Typography variant="caption" sx={{ color: "#94a3b8", mt: 2 }}>
                        Applying perturbation...
                      </Typography>
                    </Box>
                  ) : perturbedImage ? (
                    <img
                      src={perturbedImage}
                      alt="Perturbed"
                      style={{ width: "100%", height: "auto", display: "block" }}
                    />
                  ) : (
                    <PlaceholderImage label="Apply perturbation" />
                  )}
                </Box>
              </Grid>
            </Grid>

            {/* VP Experiment Results Button */}
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                fullWidth
                onClick={handleRunInference}
                disabled={isRunning}
                sx={{
                  py: 1.5,
                  bgcolor: "#ef4444",
                  color: "white",
                  fontWeight: 600,
                  "&:hover": { bgcolor: "#dc2626" },
                  "&.Mui-disabled": { bgcolor: "#7f1d1d", color: "#fca5a5" },
                }}
              >
                {isRunning ? (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <CircularProgress size={16} sx={{ color: "white" }} />
                    Loading Results...
                  </Box>
                ) : (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M4 2l10 6-10 6V2z" />
                    </svg>
                    Show Experiment Results
                    {(perturbedImage || selectedVideo) && (
                      <Chip
                        label={currentPerturbation?.label || selectedPerturbation}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: "9px",
                          fontWeight: 600,
                          bgcolor: "rgba(255,255,255,0.2)",
                          color: "white",
                          '& .MuiChip-label': { px: 0.75 },
                        }}
                      />
                    )}
                  </Box>
                )}
              </Button>
              <Typography variant="caption" sx={{ color: "#22c55e", mt: 0.5, display: "block", textAlign: "center", fontSize: "10px", fontWeight: 600 }}>
                Results from real robot rollout experiments
              </Typography>
            </Box>
          </Grid>

          {/* Right: Experiment Results */}
          <Grid size={{ xs: 12, md: 3 }}>
            <Typography
              variant="subtitle2"
              sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}
            >
              Experiment Results
            </Typography>
            <Box
              sx={{
                bgcolor: "#0f172a",
                borderRadius: 1,
                border: "1px solid #334155",
                p: 2,
                minHeight: 300,
              }}
            >
              {result ? (
                <Box>
                  {/* Perturbation Label */}
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 1,
                      mb: 1.5,
                      p: 1,
                      bgcolor: "#1e293b",
                      borderRadius: 1,
                      border: "1px solid #334155",
                    }}
                  >
                    <PerturbationIcon type={selectedPerturbation} />
                    <Typography variant="caption" sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "11px" }}>
                      {result.perturbation.replace(/_/g, ' ')}
                    </Typography>
                    <Chip
                      label={result.success_rate === -1 ? 'no data' : `${result.n_episodes} episodes`}
                      size="small"
                      sx={{
                        height: 18,
                        fontSize: "9px",
                        fontWeight: 600,
                        bgcolor: result.success_rate === -1 ? "#92400e" : "#374151",
                        color: result.success_rate === -1 ? "#fbbf24" : "#9ca3af",
                        '& .MuiChip-label': { px: 0.75 },
                      }}
                    />
                  </Box>

                  {/* Not-available message when success_rate is -1 */}
                  {result.success_rate === -1 && (
                    <Box sx={{ p: 2, bgcolor: '#1c1917', borderRadius: 1, border: '1px solid #78350f', mb: 2 }}>
                      <Typography variant="body2" sx={{ color: '#fbbf24', fontWeight: 600, mb: 0.5 }}>
                        No experiment data available
                      </Typography>
                      {error && (
                        <Typography variant="caption" sx={{ color: '#a8a29e', display: 'block' }}>
                          {error}
                        </Typography>
                      )}
                    </Box>
                  )}

                  {/* Success Rate with Delta (hidden when no data) */}
                  {result.success_rate >= 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 0.5, fontWeight: 600 }}>
                      Success Rate
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Box
                        sx={{
                          flex: 1,
                          height: 10,
                          bgcolor: "#1e293b",
                          borderRadius: 1,
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          sx={{
                            width: `${result.success_rate}%`,
                            height: "100%",
                            bgcolor: result.success_rate > 80 ? "#22c55e" : result.success_rate > 50 ? "#f59e0b" : "#ef4444",
                          }}
                        />
                      </Box>
                      <Typography variant="caption" sx={{ color: "#e2e8f0", fontWeight: 700, fontSize: "13px", minWidth: 48, textAlign: "right" }}>
                        {result.success_rate.toFixed(1)}%
                      </Typography>
                    </Box>
                    {result.baseline_success_rate != null && (
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                        <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px" }}>
                          Baseline: {result.baseline_success_rate.toFixed(1)}%
                        </Typography>
                        {result.delta_success_rate != null && (
                          <Chip
                            label={`${result.delta_success_rate >= 0 ? '+' : ''}${result.delta_success_rate.toFixed(1)}pp`}
                            size="small"
                            sx={{
                              height: 18,
                              fontSize: "10px",
                              fontWeight: 700,
                              bgcolor: result.delta_success_rate > 0 ? "rgba(34,197,94,0.2)" : result.delta_success_rate < -10 ? "rgba(239,68,68,0.3)" : "rgba(245,158,11,0.2)",
                              color: result.delta_success_rate > 0 ? "#86efac" : result.delta_success_rate < -10 ? "#fca5a5" : "#fbbf24",
                              '& .MuiChip-label': { px: 0.75 },
                            }}
                          />
                        )}
                      </Box>
                    )}
                  </Box>
                  )}

                  {/* Average Steps */}
                  {result.avg_n_steps != null && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 0.5, fontWeight: 600 }}>
                        Avg Steps
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Typography
                          variant="body2"
                          sx={{ color: "#e2e8f0", fontFamily: "monospace", fontSize: "13px", fontWeight: 600 }}
                        >
                          {result.avg_n_steps.toFixed(0)}
                        </Typography>
                        {result.baseline_avg_n_steps != null && (
                          <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px" }}>
                            (baseline: {result.baseline_avg_n_steps.toFixed(0)})
                          </Typography>
                        )}
                        {result.delta_n_steps != null && (
                          <Chip
                            label={`${result.delta_n_steps >= 0 ? '+' : ''}${result.delta_n_steps.toFixed(0)}`}
                            size="small"
                            sx={{
                              height: 18,
                              fontSize: "10px",
                              fontWeight: 600,
                              bgcolor: Math.abs(result.delta_n_steps) > 30 ? "rgba(239,68,68,0.2)" : "#1e293b",
                              color: Math.abs(result.delta_n_steps) > 30 ? "#fca5a5" : "#94a3b8",
                              '& .MuiChip-label': { px: 0.75 },
                            }}
                          />
                        )}
                      </Box>
                    </Box>
                  )}

                  {/* Side-by-side Video Comparison */}
                  {(result.baseline_video_path || result.video_path) && (
                    <Box sx={{ mt: 2, pt: 2, borderTop: "1px solid #334155" }}>
                      <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 1, fontWeight: 600 }}>
                        Rollout Videos
                      </Typography>
                      <Box sx={{ display: "flex", gap: 1 }}>
                        {result.baseline_video_path && (
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "9px", display: "block", mb: 0.5, textAlign: "center" }}>
                              Baseline
                            </Typography>
                            <video
                              src={toVideoApiUrl(`/videos/${currentModel}/${result.baseline_video_path}`)}
                              controls
                              muted
                              loop
                              playsInline
                              style={{ width: "100%", borderRadius: 4, border: "1px solid #334155" }}
                            />
                          </Box>
                        )}
                        {result.video_path && (
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="caption" sx={{ color: "#fca5a5", fontSize: "9px", display: "block", mb: 0.5, textAlign: "center" }}>
                              Perturbed
                            </Typography>
                            <video
                              src={toVideoApiUrl(`/videos/${currentModel}/${result.video_path}`)}
                              controls
                              muted
                              loop
                              playsInline
                              style={{ width: "100%", borderRadius: 4, border: "1px solid #ef4444" }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Box>
                  )}

                  {/* Data source note */}
                  <Box sx={{ mt: 2, pt: 1, borderTop: "1px solid #1e293b" }}>
                    <Typography variant="caption" sx={{ color: "#475569", display: "block", fontSize: "9px" }}>
                      From {result.n_episodes} real robot rollouts. No simulated data.
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Box
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#475569",
                  }}
                >
                  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="24" cy="24" r="20" />
                    <path d="M24 14v10l7 7" />
                  </svg>
                  <Typography variant="caption" sx={{ mt: 2, textAlign: "center" }}>
                    Click &quot;Show Experiment Results&quot; to see real VP data
                  </Typography>
                  <Typography variant="caption" sx={{ mt: 0.5, textAlign: "center", color: "#22c55e", fontSize: "9px" }}>
                    Click to query real experiment data
                  </Typography>
                </Box>
              )}
            </Box>
          </Grid>
        </Grid>

            {/* Related Rollout Videos */}
            {perturbRolloutVideos.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}>
                  Rollout Videos — {currentPerturbation?.label || selectedPerturbation} perturbation ({filteredRolloutVideos.length}{rolloutSubtypeFilter !== "all" ? ` of ${perturbRolloutVideos.length}` : ""} results)
                </Typography>
                <Typography variant="caption" sx={{ color: "#475569", display: "block", mb: 1, fontSize: "10px" }}>
                  Pre-recorded robot rollouts under this perturbation condition. Videos show actual task execution with the perturbation applied throughout.
                </Typography>

                {/* Subtype filter chips */}
                {rolloutSubtypes.length > 1 && (
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 1.5 }}>
                    <Chip
                      label={`All (${perturbRolloutVideos.length})`}
                      size="small"
                      onClick={() => setRolloutSubtypeFilter("all")}
                      sx={{
                        height: 22,
                        fontSize: "10px",
                        fontWeight: rolloutSubtypeFilter === "all" ? 700 : 500,
                        bgcolor: rolloutSubtypeFilter === "all" ? "#ef4444" : "#1e293b",
                        color: rolloutSubtypeFilter === "all" ? "white" : "#94a3b8",
                        border: `1px solid ${rolloutSubtypeFilter === "all" ? "#ef4444" : "#334155"}`,
                        cursor: "pointer",
                        '&:hover': { bgcolor: rolloutSubtypeFilter === "all" ? "#dc2626" : "#334155" },
                        '& .MuiChip-label': { px: 1 },
                      }}
                    />
                    {rolloutSubtypes.map((st) => {
                      const count = perturbRolloutVideos.filter(v => v.subtype === st).length;
                      const chipColor = getPerturbationChipColor(st);
                      const isActive = rolloutSubtypeFilter === st;
                      return (
                        <Chip
                          key={st}
                          label={`${formatPerturbationLabel(st)} (${count})`}
                          size="small"
                          onClick={() => setRolloutSubtypeFilter(isActive ? "all" : st)}
                          sx={{
                            height: 22,
                            fontSize: "10px",
                            fontWeight: isActive ? 700 : 500,
                            bgcolor: isActive ? chipColor.bg : "#1e293b",
                            color: isActive ? chipColor.text : "#94a3b8",
                            border: `1px solid ${isActive ? chipColor.text + '66' : "#334155"}`,
                            cursor: "pointer",
                            '&:hover': { bgcolor: isActive ? chipColor.bg : "#334155" },
                            '& .MuiChip-label': { px: 1 },
                          }}
                        />
                      );
                    })}
                  </Box>
                )}

                <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 1.5 }}>
                  {filteredRolloutVideos.map((v, i) => {
                    const chipColor = getPerturbationChipColor(v.subtype);
                    return (
                      <Box key={i} sx={{ bgcolor: "#0f172a", borderRadius: 1, overflow: "hidden", border: "1px solid #1e293b", position: "relative" }}>
                        {/* Perturbation type badge overlaid on video */}
                        <Chip
                          label={formatPerturbationLabel(v.subtype)}
                          size="small"
                          sx={{
                            position: "absolute",
                            top: 6,
                            left: 6,
                            zIndex: 2,
                            height: 20,
                            fontSize: "9px",
                            fontWeight: 700,
                            bgcolor: chipColor.bg,
                            color: chipColor.text,
                            border: `1px solid ${chipColor.text}33`,
                            backdropFilter: "blur(4px)",
                            '& .MuiChip-label': { px: 1 },
                          }}
                        />
                        <video
                          src={toVideoApiUrl(v.path)}
                          controls muted
                          style={{ width: "100%", height: 140, objectFit: "cover" }}
                        />
                        <Box sx={{ p: 1 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.25 }}>
                            <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 600, fontSize: "10px" }}>
                              Task {v.task}
                            </Typography>
                            <Typography variant="caption" sx={{ color: "#475569", fontSize: "9px" }}>
                              · {v.suite} · seed {v.seed}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                    );
                  })}
                </Box>
              </Box>
            )}
            {isLoadingRolloutVideos && (
              <Box sx={{ mt: 2, display: "flex", alignItems: "center", gap: 1 }}>
                <CircularProgress size={14} sx={{ color: "#ef4444" }} />
                <Typography variant="caption" sx={{ color: "#64748b" }}>Loading rollout videos...</Typography>
              </Box>
            )}
          </Box>
        )}

        {mainTab === 2 && (
          <Box sx={{ mt: 2 }}>
            <GridAblationDemo />
          </Box>
        )}

        {mainTab === 3 && (
          <Box sx={{ mt: 2 }}>
            <CounterfactualDemo />
          </Box>
        )}

        {mainTab === 4 && (
          <Box sx={{ mt: 2 }}>
            <InjectionDemo />
          </Box>
        )}
      </Box>

      {/* Error Snackbar */}
      <Snackbar
        open={showError}
        autoHideDuration={6000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={handleCloseError}
          severity="error"
          sx={{
            bgcolor: "#7f1d1d",
            color: "#fca5a5",
            "& .MuiAlert-icon": { color: "#fca5a5" },
          }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Paper>
  );
}
