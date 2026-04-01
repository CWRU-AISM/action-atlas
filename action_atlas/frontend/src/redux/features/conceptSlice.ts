import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface ConceptFeatureInfo {
  total_features: number;
  concept_features: number;
  strong_features: number;
  very_strong_features: number;
  feature_indices: number[];
  strong_feature_indices?: number[];  // Features with selectivity > 3x
  very_strong_feature_indices?: number[];  // Features with selectivity > 5x
}

interface ConceptState {
  selectedConcept: string | null;
  selectedType: string | null;
  conceptLayers: Record<string, ConceptFeatureInfo> | null;
  highlightedFeatureIds: number[];
}

const initialState: ConceptState = {
  selectedConcept: null,
  selectedType: null,
  conceptLayers: null,
  highlightedFeatureIds: [],
};

const conceptSlice = createSlice({
  name: 'concept',
  initialState,
  reducers: {
    setSelectedConcept: (
      state,
      action: PayloadAction<{
        concept: string | null;
        type: string | null;
        layers: Record<string, ConceptFeatureInfo> | null;
        strength?: 'all' | 'strong' | 'very_strong';
      }>
    ) => {
      state.selectedConcept = action.payload.concept;
      state.selectedType = action.payload.type;
      state.conceptLayers = action.payload.layers;

      const strength = action.payload.strength || 'all';

      // Extract feature IDs based on strength filter
      if (action.payload.layers) {
        const allFeatureIds: number[] = [];
        Object.values(action.payload.layers).forEach((layer) => {
          // Choose feature indices based on strength filter
          let indices: number[];
          if (strength === 'very_strong' && layer.very_strong_feature_indices) {
            indices = layer.very_strong_feature_indices;
          } else if (strength === 'strong' && layer.strong_feature_indices) {
            indices = layer.strong_feature_indices;
          } else if (strength === 'strong' && !layer.strong_feature_indices) {
            // Fallback: use top N features where N = strong_features count
            indices = layer.feature_indices.slice(0, layer.strong_features);
          } else if (strength === 'very_strong' && !layer.very_strong_feature_indices) {
            // Fallback: use top N features where N = very_strong_features count
            indices = layer.feature_indices.slice(0, layer.very_strong_features);
          } else {
            indices = layer.feature_indices;
          }
          allFeatureIds.push(...indices);
        });
        state.highlightedFeatureIds = [...new Set(allFeatureIds)]; // Deduplicate
      } else {
        state.highlightedFeatureIds = [];
      }
    },
    updateHighlightedFeaturesByStrength: (
      state,
      action: PayloadAction<'all' | 'strong' | 'very_strong'>
    ) => {
      // Re-filter highlighted features based on new strength setting
      if (state.conceptLayers) {
        const strength = action.payload;
        const allFeatureIds: number[] = [];
        Object.values(state.conceptLayers).forEach((layer) => {
          let indices: number[];
          if (strength === 'very_strong' && layer.very_strong_feature_indices) {
            indices = layer.very_strong_feature_indices;
          } else if (strength === 'strong' && layer.strong_feature_indices) {
            indices = layer.strong_feature_indices;
          } else if (strength === 'strong') {
            indices = layer.feature_indices.slice(0, layer.strong_features);
          } else if (strength === 'very_strong') {
            indices = layer.feature_indices.slice(0, layer.very_strong_features);
          } else {
            indices = layer.feature_indices;
          }
          allFeatureIds.push(...indices);
        });
        state.highlightedFeatureIds = [...new Set(allFeatureIds)];
      }
    },
    clearSelectedConcept: (state) => {
      state.selectedConcept = null;
      state.selectedType = null;
      state.conceptLayers = null;
      state.highlightedFeatureIds = [];
    },
  },
});

export const { setSelectedConcept, clearSelectedConcept, updateHighlightedFeaturesByStrength } = conceptSlice.actions;
export default conceptSlice.reducer;
