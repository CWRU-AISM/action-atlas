import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface SAEState {
  sae: string;
}

const initialState: SAEState = {
  sae: "",
};

export const saeSlice = createSlice({
  name: "sae",
  initialState,
  reducers: {
    submitSAE: (state, action: PayloadAction<string>) => {
      state.sae = action.payload;
    },
  },
});

export const { submitSAE } = saeSlice.actions;
export default saeSlice.reducer;
