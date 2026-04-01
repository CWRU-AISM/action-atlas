import {createSlice, PayloadAction} from "@reduxjs/toolkit";
import {QueryResult} from "@/types/types";

interface QueryState {
  query: string;
  optimizedQuery: string | null;
  currentLLM: string; // 新增这行
  queryResult: QueryResult;
  optimizedQueryResult: QueryResult | null;
  layerTypeData: {
    features: Record<string, any>;
    saeDistributions: Record<string, any>;
  };
}

const initialState: QueryState = {
  query: "",
  optimizedQuery: null,
  currentLLM: "pi05", // VLA model
  queryResult: {
    bins: Array.from({length: 10}, (_, i) => (i + 1) * 0.1),
    counts: Array.from({length: 10}, () => 10),
  },
  optimizedQueryResult: null,
  layerTypeData: {
    features: {},
    saeDistributions: {},
  },
};

export const querySlice = createSlice({
  name: "query",
  initialState,
  reducers: {
    submitQuery: (state, action: PayloadAction<{query: string, llm: string}>) => {
      state.query = action.payload.query;
      state.currentLLM = action.payload.llm;
    },
    setQueryResult: (state, action: PayloadAction<QueryResult>) => {
      state.queryResult = action.payload;
    },
    setOptimizedQuery: (state, action: PayloadAction<string>) => {
      state.optimizedQuery = action.payload;
    },
    setOptimizedQueryResult: (state, action: PayloadAction<QueryResult>) => {
      state.optimizedQueryResult = action.payload;
    },
    // 添加新的 reducer：
    setCurrentLLM: (state, action: PayloadAction<string>) => {
      state.currentLLM = action.payload;
    },
    setLayerTypeData: (state, action: PayloadAction<{
      features: Record<string, any>;
      saeDistributions: Record<string, any>;
    }>) => {
      state.layerTypeData = action.payload;
    },
  },
});

export const {
  submitQuery, 
  setQueryResult, 
  setOptimizedQuery, 
  setOptimizedQueryResult,
  setCurrentLLM,
  setLayerTypeData
} = querySlice.actions;

export default querySlice.reducer;
