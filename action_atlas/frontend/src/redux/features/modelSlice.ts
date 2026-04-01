import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { ModelData, VisibleRange } from "@/types/types";

// Available VLA models - defined early so it can be used in generateModelData
export type VLAModelType = 'pi05' | 'openvla' | 'xvla' | 'smolvla' | 'groot' | 'act';

export type DatasetType = 'libero' | 'simplerenv' | 'metaworld' | 'aloha';

export const DATASET_INFO: Record<DatasetType, { name: string; description: string }> = {
  libero: { name: 'LIBERO', description: '4 suites: Goal, Object, Spatial, LIBERO-10' },
  simplerenv: { name: 'SimplerEnv', description: 'WidowX + Google Robot environments' },
  metaworld: { name: 'MetaWorld', description: '4 difficulties: Easy, Medium, Hard, Very Hard' },
  aloha: { name: 'ALOHA', description: 'Bimanual manipulation (Insertion, TransferCube)' },
};

// Suites available per dataset
export const DATASET_SUITES: Record<DatasetType, { display: string; value: string }[]> = {
  libero: [
    { display: 'Goal', value: 'libero_goal' },
    { display: 'Spatial', value: 'libero_spatial' },
    { display: 'Object', value: 'libero_object' },
    { display: 'LIBERO-10', value: 'libero_10' },
  ],
  simplerenv: [
    { display: 'WidowX', value: 'simplerenv_widowx' },
    { display: 'Google Robot', value: 'simplerenv_google_robot' },
  ],
  metaworld: [
    { display: 'MetaWorld (All)', value: 'metaworld' },
    { display: 'MetaWorld Easy', value: 'metaworld_easy' },
    { display: 'MetaWorld Medium', value: 'metaworld_medium' },
    { display: 'MetaWorld Hard', value: 'metaworld_hard' },
    { display: 'MetaWorld V.Hard', value: 'metaworld_very_hard' },
  ],
  aloha: [
    { display: 'Insertion', value: 'aloha_insertion' },
    { display: 'TransferCube', value: 'aloha_transfercube' },
  ],
};

export const VLA_MODELS = {
  pi05: { id: 'pi05', name: 'Pi0.5', layers: 18, status: 'available', params: '3B', actionGen: 'Flow Matching', environments: ['libero'] as DatasetType[] },
  openvla: { id: 'openvla', name: 'OpenVLA-OFT', layers: 32, status: 'available', params: '7B', actionGen: 'L1 Regression', environments: ['libero'] as DatasetType[] },
  xvla: { id: 'xvla', name: 'X-VLA', layers: 24, status: 'available', params: '1B', actionGen: 'Flow Matching', environments: ['libero', 'simplerenv'] as DatasetType[] },
  smolvla: { id: 'smolvla', name: 'SmolVLA', layers: 32, status: 'available', params: '450M', actionGen: 'Flow Matching', environments: ['libero', 'metaworld'] as DatasetType[] },
  groot: { id: 'groot', name: 'GR00T N1.5', layers: 32, status: 'available', params: '3B', actionGen: 'Diffusion', environments: ['libero'] as DatasetType[] },
  act: { id: 'act', name: 'ACT-ALOHA', layers: 0, status: 'available', params: '80M', actionGen: 'CVAE', environments: ['aloha'] as DatasetType[] },
} as const;

// Define metric groups for VLA analysis
export const metricGroups = {
  motion: ["put_features", "open_features", "push_features", "pick_features"],
  object: ["bowl_features", "plate_features", "stove_features", "cabinet_features", "drawer_features", "wine_bottle_features", "rack_features"],
  spatial: ["on_features", "in_features", "top_features", "front_features", "middle_features"],
  actionPhase: ["approach_features", "grasp_features", "lift_features", "transport_features", "lower_features", "release_features", "retract_features"],
  totals: ["total_motion", "total_object", "total_spatial", "total_action_phase"]
};

// Layer type definitions for multi-pathway models
const LAYER_CONFIGS: Record<VLAModelType, { prefix: string; segments?: { name: string; prefix: string; count: number }[] }> = {
  pi05: { prefix: 'action_expert_layer' },
  openvla: { prefix: 'layer' },
  xvla: { prefix: 'layer' },
  smolvla: { prefix: '', segments: [
    { name: 'VLM', prefix: 'vlm_layer', count: 32 },
    { name: 'Expert', prefix: 'expert_layer', count: 32 },
  ]},
  groot: { prefix: '', segments: [
    { name: 'DiT', prefix: 'dit_layer', count: 16 },
    { name: 'Eagle', prefix: 'eagle_layer', count: 12 },
    { name: 'VL-SA', prefix: 'vlsa_layer', count: 4 },
  ]},
  act: { prefix: '' },
};

// Helper function to generate VLA model data for a given number of layers
const generateModelData = (numLayers: number = 18, modelType: VLAModelType = 'pi05'): ModelData[] => {
  // ACT has no layers/SAE data
  if (modelType === 'act') return [];
  const data: ModelData[] = [];
  const config = LAYER_CONFIGS[modelType];

  if (config.segments) {
    // Multi-pathway models (SmolVLA, GR00T)
    let globalIdx = 0;
    for (const seg of config.segments) {
      for (let i = 0; i < seg.count; i++) {
        const id = `${seg.prefix}_${i}-concepts`;
        data.push({
          id,
          type: seg.name,
          layer: globalIdx,
          top_10_score: { value: 0, rank: numLayers },
          top_100_score: { value: 0, rank: numLayers },
          top_1000_score: { value: 0, rank: numLayers },
        });
        globalIdx++;
      }
    }
  } else {
    // Single-pathway models (Pi0.5, OpenVLA, X-VLA)
    for (let layer = 0; layer < numLayers; layer++) {
      const id = `${config.prefix}_${layer}-concepts`;
      data.push({
        id,
        type: "RES",
        layer: layer,
        top_10_score: { value: 0, rank: numLayers },
        top_100_score: { value: 0, rank: numLayers },
        top_1000_score: { value: 0, rank: numLayers },
      });
    }
  }
  return data;
};

interface ModelState {
  currentModel: VLAModelType;
  currentDataset: DatasetType;
  modelData: ModelData[];
  selectedModel: ModelData | null;
  visibleRange: VisibleRange;
  selectedAttrs: string[];
  metricGroups: {
    [key: string]: string[];
  };
}

const initialState: ModelState = {
  currentModel: 'pi05',
  currentDataset: 'libero',
  modelData: generateModelData(),
  selectedModel: null,
  visibleRange: { start: 0, end: 3 },
  selectedAttrs: ["put_features", "open_features", "push_features", "pick_features"],
  metricGroups,
};

export const modelSlice = createSlice({
  name: "model",
  initialState,
  reducers: {
    setCurrentModel: (state, action: PayloadAction<VLAModelType>) => {
      state.currentModel = action.payload;
      // Reset dataset to the first available environment for this model
      const envs = VLA_MODELS[action.payload].environments;
      state.currentDataset = envs[0] as DatasetType;
      // Regenerate model data with the correct number of layers for the new model
      const numLayers = VLA_MODELS[action.payload].layers;
      state.modelData = generateModelData(numLayers, action.payload);
      state.selectedModel = null;
      state.visibleRange = { start: 0, end: numLayers > 0 ? Math.min(3, numLayers - 1) : 0 };
    },
    setCurrentDataset: (state, action: PayloadAction<DatasetType>) => {
      state.currentDataset = action.payload;
    },
    setModelData: (state, action: PayloadAction<ModelData[]>) => {
      state.modelData = action.payload;
    },
    setSelectedModel: (state, action: PayloadAction<ModelData | null>) => {
      state.selectedModel = action.payload;
    },
    setVisibleRange: (state, action: PayloadAction<VisibleRange>) => {
      state.visibleRange = action.payload;
    },
    setSelectedAttrs: (state, action: PayloadAction<string[]>) => {
      state.selectedAttrs = action.payload;
    },
    toggleMetricGroup: (state, action: PayloadAction<{ group: string; selected: boolean }>) => {
      const { group, selected } = action.payload;
      const groupMetrics = state.metricGroups[group] || [];
      
      if (selected) {
        // Add all metrics from the group that aren't already selected
        const newAttrs = [...state.selectedAttrs];
        groupMetrics.forEach(metric => {
          if (!newAttrs.includes(metric)) {
            newAttrs.push(metric);
          }
        });
        state.selectedAttrs = newAttrs;
      } else {
        // Remove all metrics from the group
        state.selectedAttrs = state.selectedAttrs.filter(
          attr => !groupMetrics.includes(attr)
        );
        
        if (state.selectedAttrs.length === 0) {
          state.selectedAttrs = ["put_features", "open_features", "push_features", "pick_features"];
        }
      }
    },
  },
});

export const {
  setCurrentModel,
  setCurrentDataset,
  setModelData,
  setSelectedModel,
  setVisibleRange,
  setSelectedAttrs,
  toggleMetricGroup,
} = modelSlice.actions;

export default modelSlice.reducer; 