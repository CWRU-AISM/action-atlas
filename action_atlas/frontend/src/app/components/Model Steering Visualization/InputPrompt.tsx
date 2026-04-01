"use client";
import React from "react";
import { Button, CircularProgress, TextField, Typography } from "@mui/material";
import { PlayCircle } from "lucide-react";
import { InputPromptProps } from "@/types/types";

const InputPrompt: React.FC<InputPromptProps> = ({
  prompt,
  setPrompt,
  handleSteer,
  loading,
}) => {
  return (
    <div className="flex gap-1 items-center w-full mb-2">
      <TextField
        placeholder="Enter prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        size="small"
        fullWidth
        sx={{
          "& .MuiInputBase-root": {
            height: "28px",
            fontSize: "11px",
          },
          "& .MuiInputBase-input": {
            padding: "4px 8px",
          },
        }}
      />
      <Button
        variant="contained"
        onClick={handleSteer}
        disabled={loading || !prompt}
        size="small"
        sx={{ height: "28px", minWidth: "70px", fontSize: "10px" }}
      >
        {loading ? <CircularProgress size={14} /> : "Steer"}
      </Button>
    </div>
  );
};

export default InputPrompt;
