import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { FeatureDetailResponse } from "@/types/feature";

export interface FeatureInfo {
  data: FeatureDetailResponse;
}

// 更新 SelectedToken 类型定义
export interface SelectedToken {
  prompt: string; // 保存token所在的prompt
  token_index: number; // token在prompt中的索引
  token: string; // token的文本内容
  activation_value: number; // token的激活值
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
      // 如果该token还未被选中（通过prompt和索引判断唯一性），则添加
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
