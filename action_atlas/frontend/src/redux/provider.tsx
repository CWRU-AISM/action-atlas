"use client";

import { Provider } from "react-redux";
import { store } from "./store";
import { ReactNode } from "react";

export function ReduxProvider({ children }: { children: ReactNode }) {
  // Expose store on window for screenshot automation (Puppeteer)
  if (typeof window !== "undefined") {
    (window as any).__REDUX_STORE__ = store;
  }
  return <Provider store={store}>{children}</Provider>;
}
