import { configureStore } from "@reduxjs/toolkit";
import queryReducer from "./features/querySlice";
import saeReducer from "./features/saeSlice";
import featureReducer from "./features/featureSlice";
import historyReducer from "./features/historySlice";
import modelReducer from "./features/modelSlice";
import llmReducer from './features/llmSlice';
import conceptReducer from './features/conceptSlice';

export const store = configureStore({
  reducer: {
    query: queryReducer,
    sae: saeReducer,
    feature: featureReducer,
    history: historyReducer,
    model: modelReducer,
    llm: llmReducer,
    concept: conceptReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
