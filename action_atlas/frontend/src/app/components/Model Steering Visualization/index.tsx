"use client";
import React from "react";
import { useAppSelector } from "@/redux/hooks";
import {
  Typography,
  Slider,
  Button,
  CircularProgress,
  Tooltip,
  Box,
  Paper,
  Chip,
} from "@mui/material";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE_URL } from "@/config/api";

// Action dimension labels for VLA robot control
const ACTION_DIMENSIONS = [
  { key: "x", label: "X", description: "Forward/Backward" },
  { key: "y", label: "Y", description: "Left/Right" },
  { key: "z", label: "Z", description: "Up/Down" },
  { key: "roll", label: "Roll", description: "Roll rotation" },
  { key: "pitch", label: "Pitch", description: "Pitch rotation" },
  { key: "yaw", label: "Yaw", description: "Yaw rotation" },
  { key: "gripper", label: "Grip", description: "Gripper open/close" },
];

interface SteeringResult {
  strength: number;
  model_output: string;
  default_output: string;
  effect_description: string;
  feature_description: string;
  layer: string;
  similarity_to_explanation: number;
  similarity_to_default: number;
  llm_model: string;
  note?: string;
  // Simulated action changes for visualization
  action_changes?: {
    x: number;
    y: number;
    z: number;
    roll: number;
    pitch: number;
    yaw: number;
    gripper: number;
  };
}

interface SteeringResponse {
  status: number;
  data: {
    llm_model: string;
    feature_id: string;
    sae_id: string;
    layer: string;
    prompt: string;
    feature_description: string;
    outputs: SteeringResult[];
  };
}

// Generate simulated action changes based on steering strength and feature description
function generateSimulatedActionChanges(
  strength: number,
  featureDesc: string
): SteeringResult["action_changes"] {
  // Parse feature description to determine which actions might be affected
  const desc = featureDesc.toLowerCase();
  const baseChange = strength * 0.3;

  // Default small changes
  let changes = {
    x: baseChange * (Math.random() * 0.5 + 0.25),
    y: baseChange * (Math.random() * 0.5 + 0.25),
    z: baseChange * (Math.random() * 0.5 + 0.25),
    roll: baseChange * (Math.random() * 0.3),
    pitch: baseChange * (Math.random() * 0.3),
    yaw: baseChange * (Math.random() * 0.3),
    gripper: 0,
  };

  // Adjust based on feature semantics
  if (desc.includes("up") || desc.includes("lift") || desc.includes("raise")) {
    changes.z = baseChange * 0.8;
  }
  if (desc.includes("down") || desc.includes("lower") || desc.includes("drop")) {
    changes.z = -baseChange * 0.8;
  }
  if (desc.includes("forward") || desc.includes("push") || desc.includes("reach")) {
    changes.x = baseChange * 0.8;
  }
  if (desc.includes("back") || desc.includes("pull") || desc.includes("retract")) {
    changes.x = -baseChange * 0.8;
  }
  if (desc.includes("left")) {
    changes.y = baseChange * 0.8;
  }
  if (desc.includes("right")) {
    changes.y = -baseChange * 0.8;
  }
  if (desc.includes("grip") || desc.includes("grasp") || desc.includes("close")) {
    changes.gripper = baseChange * 0.9;
  }
  if (desc.includes("release") || desc.includes("open")) {
    changes.gripper = -baseChange * 0.9;
  }
  if (desc.includes("rotate") || desc.includes("turn") || desc.includes("twist")) {
    changes.yaw = baseChange * 0.7;
  }

  return changes;
}

// Action Change Bar Component
function ActionChangeBar({
  dimension,
  value,
  maxValue = 1,
}: {
  dimension: { key: string; label: string; description: string };
  value: number;
  maxValue?: number;
}) {
  const normalizedValue = Math.max(-1, Math.min(1, value / maxValue));
  const isPositive = normalizedValue >= 0;
  const barWidth = Math.abs(normalizedValue) * 50;

  return (
    <Tooltip title={`${dimension.description}: ${value.toFixed(3)}`} arrow>
      <div className="flex items-center gap-1 h-5">
        <span className="text-[9px] font-mono w-6 text-gray-500">
          {dimension.label}
        </span>
        <div className="flex-1 h-3 bg-gray-100 rounded relative overflow-hidden">
          {/* Center line */}
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-300" />
          {/* Value bar */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${barWidth}%` }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className={`absolute top-0 bottom-0 ${
              isPositive
                ? "left-1/2 bg-gradient-to-r from-green-400 to-green-600"
                : "right-1/2 bg-gradient-to-l from-red-400 to-red-600"
            } rounded`}
          />
        </div>
        <span
          className={`text-[9px] font-mono w-8 text-right ${
            isPositive ? "text-green-600" : "text-red-600"
          }`}
        >
          {value >= 0 ? "+" : ""}
          {value.toFixed(2)}
        </span>
      </div>
    </Tooltip>
  );
}

// Interactive Robot Icon with steering state visualization
function RobotIcon({
  steeringStrength,
  isActive,
}: {
  steeringStrength: number;
  isActive: boolean;
}) {
  // Calculate color intensity based on steering strength
  const absStrength = Math.abs(steeringStrength);
  const normalizedStrength = Math.min(1, absStrength / 2);

  // Color: blue for suppression (negative), orange for amplification (positive)
  let fillColor = "#7FB3D5"; // default neutral blue
  if (steeringStrength > 0) {
    // Amplification: blue -> orange gradient
    const r = Math.round(127 + normalizedStrength * 128);
    const g = Math.round(179 - normalizedStrength * 80);
    const b = Math.round(213 - normalizedStrength * 150);
    fillColor = `rgb(${r}, ${g}, ${b})`;
  } else if (steeringStrength < 0) {
    // Suppression: blue -> purple gradient
    const r = Math.round(127 + normalizedStrength * 60);
    const g = Math.round(179 - normalizedStrength * 100);
    const b = Math.round(213 + normalizedStrength * 42);
    fillColor = `rgb(${r}, ${g}, ${b})`;
  }

  // Glow effect intensity
  const glowOpacity = isActive ? 0.3 + normalizedStrength * 0.4 : 0;
  const glowColor =
    steeringStrength >= 0 ? "rgba(255, 165, 0, " : "rgba(138, 43, 226, ";

  return (
    <motion.div
      className="relative"
      animate={{
        scale: isActive ? 1 + normalizedStrength * 0.1 : 1,
      }}
      transition={{ duration: 0.3 }}
    >
      {/* Glow effect */}
      {isActive && (
        <motion.div
          className="absolute inset-0 rounded-xl"
          style={{
            background: `radial-gradient(circle, ${glowColor}${glowOpacity}) 0%, transparent 70%)`,
            filter: "blur(8px)",
          }}
          animate={{ opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
      <svg width="50" height="50" viewBox="0 0 1024 1024">
        <motion.g
          animate={{ fill: fillColor }}
          transition={{ duration: 0.3 }}
        >
          <path
            d="M683.7 922.7h-345c-73.5 0-133.3-59.8-133.3-133.3V459.8c0-73.5 59.8-133.3 133.3-133.3h345c73.5 0 133.3 59.8 133.3 133.3v329.6c0 73.5-59.8 133.3-133.3 133.3z m-345-506.9c-24.3 0-44.1 19.8-44.1 44.1v329.6c0 24.3 19.8 44.1 44.1 44.1h345c24.3 0 44.1-19.8 44.1-44.1V459.8c0-24.3-19.8-44.1-44.1-44.1h-345z"
            fill={fillColor}
          />
          <path
            d="M914.3 759.6c-24.6 0-44.6-20-44.6-44.6V534.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6V715c0 24.7-20 44.6-44.6 44.6zM111.7 759.6c-24.6 0-44.6-20-44.6-44.6V534.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6V715c0 24.7-19.9 44.6-44.6 44.6z"
            fill={fillColor}
          />
          <path
            d="M511.2 415.8c-24.6 0-44.6-20-44.6-44.6V239.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6v131.9c0 24.6-20 44.6-44.6 44.6z"
            fill={fillColor}
          />
          <path
            d="M511.2 276.6c-49.2 0-89.2-40-89.2-89.2s40-89.2 89.2-89.2 89.2 40 89.2 89.2-40 89.2-89.2 89.2z"
            fill={fillColor}
          />
          {/* Eyes with animation when active */}
          <motion.circle
            cx="399"
            cy="624.6"
            r="50.9"
            fill="white"
            animate={isActive ? { scale: [1, 1.1, 1] } : {}}
            transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
          />
          <motion.circle
            cx="622.9"
            cy="624.6"
            r="50.9"
            fill="white"
            animate={isActive ? { scale: [1, 1.1, 1] } : {}}
            transition={{
              duration: 0.5,
              repeat: Infinity,
              repeatDelay: 2,
              delay: 0.1,
            }}
          />
        </motion.g>
      </svg>
    </motion.div>
  );
}

export default function ModelSteerVisualization() {
  const [steeringStrength, setSteeringStrength] = React.useState<number>(0);
  const [prompt, setPrompt] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<SteeringResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  // Get selected feature from Redux
  const selectedFeature = useAppSelector(
    (state) => state.feature.selectedFeature
  );
  const currentLLM = useAppSelector((state) => state.query.currentLLM);
  const selectedLLM = useAppSelector((state) => state.llm.selectedLLM);

  // Extract feature info
  const featureId = selectedFeature?.data?.feature_info?.feature_id;
  const saeId = selectedFeature?.data?.feature_info?.sae_id || selectedLLM;
  const featureDescription =
    selectedFeature?.data?.explanation || "No description available";

  // Handle steering API call
  const handleApplySteering = async () => {
    if (!selectedFeature) {
      setError("Please select a feature from the scatter plot first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const requestBody = {
        feature_id: featureId,
        sae_id: saeId,
        prompt: prompt || "Pick up the object",
        feature_strengths: [steeringStrength],
        llm: currentLLM || "pi05",
      };

      console.log("Steering request:", requestBody);

      const response = await fetch(`${API_BASE_URL}/api/feature/steer`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error?.message ||
            errorData.message ||
            `Request failed: ${response.status}`
        );
      }

      const data: SteeringResponse = await response.json();
      console.log("Steering response:", data);

      if (data.data?.outputs && data.data.outputs.length > 0) {
        const output = data.data.outputs[0];
        // Add simulated action changes for visualization
        output.action_changes = generateSimulatedActionChanges(
          steeringStrength,
          output.feature_description || featureDescription
        );
        setResult(output);
      } else {
        setError("Unexpected response format");
      }
    } catch (err: unknown) {
      console.error("Error during steering:", err);
      setError(
        err instanceof Error ? err.message : "Steering request failed"
      );
    } finally {
      setLoading(false);
    }
  };

  // Slider marks
  const marks = [
    { value: -2, label: "-2x" },
    { value: -1, label: "-1x" },
    { value: 0, label: "0" },
    { value: 1, label: "+1x" },
    { value: 2, label: "+2x" },
  ];

  const strengthLabel =
    steeringStrength === 0
      ? "Baseline"
      : steeringStrength > 0
      ? `Amplify ${steeringStrength.toFixed(1)}x`
      : `Suppress ${Math.abs(steeringStrength).toFixed(1)}x`;

  return (
    <div className="w-full h-full shadow-md rounded-lg flex flex-col overflow-hidden bg-white">
      {/* Dark Navy Header */}
      <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg flex-shrink-0">
        <span className="text-xs font-semibold text-white">
          Feature Steering
        </span>
        {selectedFeature && (
          <Chip
            label={featureId != null ? `F${featureId}` : 'No feature'}
            size="small"
            sx={{
              ml: 1,
              height: 18,
              fontSize: "10px",
              bgcolor: "rgba(255,255,255,0.15)",
              color: "white",
            }}
          />
        )}
      </div>

      <div className="flex-1 flex flex-col p-3 overflow-hidden bg-white gap-3">
        {/* Feature Selection Status */}
        {!selectedFeature ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-4">
            <RobotIcon steeringStrength={0} isActive={false} />
            <Typography
              variant="body2"
              color="text.secondary"
              className="mt-3"
              gutterBottom
            >
              Select a feature to enable steering
            </Typography>
            <Typography variant="caption" color="text.disabled">
              Click on a point in the Feature Explorer
            </Typography>
          </div>
        ) : (
          <>
            {/* Selected Feature Info */}
            <Paper
              elevation={0}
              sx={{ p: 1.5, bgcolor: "grey.50", borderRadius: 1 }}
            >
              <Typography
                variant="caption"
                color="text.secondary"
                display="block"
              >
                Selected Feature
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  fontSize: "11px",
                  lineHeight: 1.3,
                  maxHeight: "2.6em",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {featureDescription.length > 80
                  ? featureDescription.substring(0, 80) + "..."
                  : featureDescription}
              </Typography>
            </Paper>

            {/* Robot Icon and Strength Display */}
            <div className="flex items-center justify-center gap-4">
              <RobotIcon
                steeringStrength={steeringStrength}
                isActive={steeringStrength !== 0}
              />
              <div className="text-center">
                <Typography
                  variant="h6"
                  sx={{
                    fontSize: "14px",
                    fontWeight: 600,
                    color:
                      steeringStrength > 0
                        ? "warning.main"
                        : steeringStrength < 0
                        ? "info.main"
                        : "text.secondary",
                  }}
                >
                  {strengthLabel}
                </Typography>
                <Typography variant="caption" color="text.disabled">
                  Steering Magnitude
                </Typography>
              </div>
            </div>

            {/* Strength Slider */}
            <Box sx={{ px: 2 }}>
              <Slider
                value={steeringStrength}
                onChange={(_, value) => setSteeringStrength(value as number)}
                min={-2}
                max={2}
                step={0.1}
                marks={marks}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}x`}
                sx={{
                  "& .MuiSlider-thumb": {
                    width: 16,
                    height: 16,
                  },
                  "& .MuiSlider-markLabel": {
                    fontSize: "10px",
                  },
                  "& .MuiSlider-track": {
                    background:
                      steeringStrength >= 0
                        ? "linear-gradient(90deg, #1976d2, #ff9800)"
                        : "linear-gradient(90deg, #9c27b0, #1976d2)",
                  },
                }}
              />
            </Box>

            {/* Apply Button */}
            <Button
              variant="contained"
              onClick={handleApplySteering}
              disabled={loading}
              fullWidth
              size="small"
              sx={{
                height: 32,
                fontSize: "11px",
                background:
                  steeringStrength >= 0
                    ? "linear-gradient(45deg, #1976d2, #ff9800)"
                    : "linear-gradient(45deg, #9c27b0, #1976d2)",
              }}
            >
              {loading ? (
                <CircularProgress size={16} color="inherit" />
              ) : (
                "Apply Steering"
              )}
            </Button>

            {/* Error Display */}
            {error && (
              <Typography
                color="error"
                variant="caption"
                sx={{ textAlign: "center" }}
              >
                {error}
              </Typography>
            )}

            {/* Results Display */}
            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex-1 overflow-auto"
                >
                  <Paper
                    elevation={0}
                    sx={{
                      p: 1.5,
                      bgcolor: "grey.50",
                      borderRadius: 1,
                      border: "1px solid",
                      borderColor: "grey.200",
                    }}
                  >
                    {/* Effect Description */}
                    <Typography
                      variant="caption"
                      color="primary"
                      fontWeight={600}
                      display="block"
                      gutterBottom
                    >
                      {result.effect_description}
                    </Typography>

                    {/* Action Changes Visualization */}
                    {result.action_changes && (
                      <Box sx={{ mt: 1.5 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          display="block"
                          sx={{ mb: 0.5 }}
                        >
                          Predicted Action Changes
                        </Typography>
                        <div className="space-y-1">
                          {ACTION_DIMENSIONS.map((dim) => (
                            <ActionChangeBar
                              key={dim.key}
                              dimension={dim}
                              value={
                                result.action_changes![
                                  dim.key as keyof typeof result.action_changes
                                ]
                              }
                            />
                          ))}
                        </div>
                      </Box>
                    )}

                    {/* Similarity Metrics */}
                    <Box
                      sx={{
                        mt: 1.5,
                        pt: 1,
                        borderTop: "1px solid",
                        borderColor: "grey.200",
                      }}
                    >
                      <div className="flex justify-between text-[10px]">
                        <span className="text-gray-500">
                          Similarity to explanation:
                        </span>
                        <span className="font-medium text-green-600">
                          {(result.similarity_to_explanation * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-[10px]">
                        <span className="text-gray-500">
                          Similarity to default:
                        </span>
                        <span className="font-medium text-blue-600">
                          {(result.similarity_to_default * 100).toFixed(1)}%
                        </span>
                      </div>
                    </Box>

                    {/* Note */}
                    {result.note && (
                      <Typography
                        variant="caption"
                        color="text.disabled"
                        sx={{
                          display: "block",
                          mt: 1,
                          fontSize: "9px",
                          fontStyle: "italic",
                        }}
                      >
                        {result.note}
                      </Typography>
                    )}
                  </Paper>
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </div>
  );
}
