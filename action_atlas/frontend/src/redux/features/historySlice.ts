import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface HistoryItem {
  feature_id: string;
  sae_id: string;
  timestamp: string;
}

interface HistoryState {
  items: HistoryItem[];
}

const initialState: HistoryState = {
  items: [],
};

export const historySlice = createSlice({
  name: "history",
  initialState,
  reducers: {
    addHistoryItem: (state, action: PayloadAction<HistoryItem>) => {
      state.items = [action.payload, ...state.items];
    },
    setHistory: (state, action: PayloadAction<HistoryItem[]>) => {
      state.items = action.payload;
    },
    clearHistory: (state) => {
      state.items = [];
    },
  },
});

export const { addHistoryItem, setHistory, clearHistory } = historySlice.actions;
export default historySlice.reducer;

