"use client";
import {ModelRadarPanelProps} from "@/types/types";
import {Paper} from "@mui/material";
import React from "react";

const ModelRadarPanel: React.FC<ModelRadarPanelProps> = ({
  selectedModel,
}) => {
  if (!selectedModel) return null;
  return (
    <Paper elevation={1} className="p-2 w-full h-full" sx={{ mt: 3 }}>
    </Paper>
  );
};

export default ModelRadarPanel;
