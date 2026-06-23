import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { FeatureDetailResponse } from "@/types/feature";

export interface FeatureInfo {
  data: FeatureDetailResponse;
}

// Updated SelectedToken type definition
export interface SelectedToken {
  prompt: string; // the prompt the token belongs to
  token_index: number; // index of the token within the prompt
  token: string; // text content of the token
  activation_value: number; // activation value of the token
}

interface FeatureState {
  selectedFeature: FeatureInfo | null;
  validateFeatureId: string | null;
  selectedTokens: SelectedToken[];
  isLoading: boolean;
}

const initialState: FeatureState = {
  selectedFeature: null,
  validateFeatureId: null,
  selectedTokens: [],
  isLoading: false,
};

export const featureSlice = createSlice({
  name: "feature",
  initialState,
  reducers: {
    setSelectedFeature: (state, action: PayloadAction<FeatureInfo | null>) => {
      state.selectedFeature = action.payload;
    },
    setValidateFeatureId: (state, action: PayloadAction<string | null>) => {
      state.validateFeatureId = action.payload;
    },
    setSelectedTokens: (state, action: PayloadAction<SelectedToken[]>) => {
      state.selectedTokens = action.payload;
    },
    addSelectedToken: (state, action: PayloadAction<SelectedToken>) => {
      // Add the token if it is not already selected (uniqueness by prompt and index)
      const exists = state.selectedTokens.some(
        (t) =>
          t.prompt === action.payload.prompt &&
          t.token_index === action.payload.token_index
      );
      if (!exists) {
        state.selectedTokens.push(action.payload);
      }
    },
    removeSelectedToken: (
      state,
      action: PayloadAction<{ prompt: string; token_index: number }>
    ) => {
      state.selectedTokens = state.selectedTokens.filter(
        (t) =>
          !(
            t.prompt === action.payload.prompt &&
            t.token_index === action.payload.token_index
          )
      );
    },
    setFeatureLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
});

export const {
  setSelectedFeature,
  setValidateFeatureId,
  setSelectedTokens,
  addSelectedToken,
  removeSelectedToken,
  setFeatureLoading,
} = featureSlice.actions;
export default featureSlice.reducer;
