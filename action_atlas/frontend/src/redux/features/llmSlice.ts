// src/redux/features/llmSlice.ts
// Updated for VLA Feature Explorer - manages layer and suite selection
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface VLAState {
  selectedLLM: string;  // Keep for compatibility - now stores "layer-suite" format
  selectedLayer: string;
  selectedSuite: string;
  selectedStrength: 'all' | 'strong' | 'very_strong';  // Strength filter for selectivity
  availableLLMs: { display: string; value: string }[];  // Keep for compatibility
  availableLayers: { display: string; value: string }[];
  availableSuites: { display: string; value: string }[];
}

const initialState: VLAState = {
  selectedLLM: 'action_expert_layer_12-concepts',  // Default VLA SAE ID
  selectedLayer: 'action_expert_layer_12',
  selectedSuite: 'concepts',
  selectedStrength: 'all',  // Default: show all features
  availableLLMs: [
    // VLA layers as dropdown options
    { display: 'Layer 0', value: 'action_expert_layer_0' },
    { display: 'Layer 1', value: 'action_expert_layer_1' },
    { display: 'Layer 2', value: 'action_expert_layer_2' },
    { display: 'Layer 3', value: 'action_expert_layer_3' },
    { display: 'Layer 4', value: 'action_expert_layer_4' },
    { display: 'Layer 5', value: 'action_expert_layer_5' },
    { display: 'Layer 6', value: 'action_expert_layer_6' },
    { display: 'Layer 7', value: 'action_expert_layer_7' },
    { display: 'Layer 8', value: 'action_expert_layer_8' },
    { display: 'Layer 9', value: 'action_expert_layer_9' },
    { display: 'Layer 10', value: 'action_expert_layer_10' },
    { display: 'Layer 11', value: 'action_expert_layer_11' },
    { display: 'Layer 12', value: 'action_expert_layer_12' },
    { display: 'Layer 13', value: 'action_expert_layer_13' },
    { display: 'Layer 14', value: 'action_expert_layer_14' },
    { display: 'Layer 15', value: 'action_expert_layer_15' },
    { display: 'Layer 16', value: 'action_expert_layer_16' },
    { display: 'Layer 17', value: 'action_expert_layer_17' },
    { display: 'Input Proj', value: 'action_in_proj' },
    { display: 'Output Proj', value: 'action_out_proj_input' },
  ],
  availableLayers: [
    // All layers option for aggregate analysis
    { display: 'All', value: 'all_layers' },
    // 18 action expert layers (1024D each)
    { display: 'L0', value: 'action_expert_layer_0' },
    { display: 'L1', value: 'action_expert_layer_1' },
    { display: 'L2', value: 'action_expert_layer_2' },
    { display: 'L3', value: 'action_expert_layer_3' },
    { display: 'L4', value: 'action_expert_layer_4' },
    { display: 'L5', value: 'action_expert_layer_5' },
    { display: 'L6', value: 'action_expert_layer_6' },
    { display: 'L7', value: 'action_expert_layer_7' },
    { display: 'L8', value: 'action_expert_layer_8' },
    { display: 'L9', value: 'action_expert_layer_9' },
    { display: 'L10', value: 'action_expert_layer_10' },
    { display: 'L11', value: 'action_expert_layer_11' },
    { display: 'L12', value: 'action_expert_layer_12' },
    { display: 'L13', value: 'action_expert_layer_13' },
    { display: 'L14', value: 'action_expert_layer_14' },
    { display: 'L15', value: 'action_expert_layer_15' },
    { display: 'L16', value: 'action_expert_layer_16' },
    { display: 'L17', value: 'action_expert_layer_17' },
    // 2 projection layers (1024D each)
    { display: 'InProj', value: 'action_in_proj' },
    { display: 'OutProjIn', value: 'action_out_proj_input' },
    // Note: action_out_proj_output is 32D, too small for SAE
  ],
  availableSuites: [
    { display: 'Goal', value: 'concepts' },
    { display: 'Object', value: 'object' },
    { display: 'Spatial', value: 'spatial' },
    { display: 'LIBERO-10', value: 'libero_10' },
  ]
};

export const LLMSlice = createSlice({
  name: 'llm',
  initialState,
  reducers: {
    setSelectedLLM: (state, action: PayloadAction<string>) => {
      state.selectedLLM = action.payload;
      // Parse the sae_id to extract layer and suite (format: "layer-suite")
      const lastDashIndex = action.payload.lastIndexOf('-');
      if (lastDashIndex > 0) {
        const layer = action.payload.substring(0, lastDashIndex);
        const suite = action.payload.substring(lastDashIndex + 1);
        state.selectedLayer = layer;
        state.selectedSuite = suite;
      }
    },
    setSelectedLayer: (state, action: PayloadAction<string>) => {
      state.selectedLayer = action.payload;
      state.selectedLLM = `${action.payload}-${state.selectedSuite}`;
    },
    setSelectedSuite: (state, action: PayloadAction<string>) => {
      state.selectedSuite = action.payload;
      state.selectedLLM = `${state.selectedLayer}-${action.payload}`;
    },
    setSelectedStrength: (state, action: PayloadAction<'all' | 'strong' | 'very_strong'>) => {
      state.selectedStrength = action.payload;
    },
    addLLM: (state, action: PayloadAction<{ display: string; value: string }>) => {
      if (!state.availableLLMs.find(llm => llm.value === action.payload.value)) {
        state.availableLLMs.push(action.payload);
      }
    },
    setAvailableLayers: (state, action: PayloadAction<{ display: string; value: string }[]>) => {
      state.availableLayers = action.payload;
    },
    setAvailableSuites: (state, action: PayloadAction<{ display: string; value: string }[]>) => {
      state.availableSuites = action.payload;
    }
  }
});

export const {
  setSelectedLLM,
  setSelectedLayer,
  setSelectedSuite,
  setSelectedStrength,
  addLLM,
  setAvailableLayers,
  setAvailableSuites
} = LLMSlice.actions;
export default LLMSlice.reducer;
