"use client";
import React, { useState } from "react";
import {
  Paper,
  Typography,
  Box,
  Chip,
  Tooltip,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import RouteIcon from "@mui/icons-material/Route";
import { useAppSelector } from "@/redux/hooks";
import { VLAModelType } from "@/redux/features/modelSlice";

// Displacement analysis data per model (placeholder data based on paper findings)
interface ObjectDisplacementData {
  baselineMeanMaxDisp: number; // meters
  injectionMeanMaxDisp: number; // meters
  reductionPct: number; // % reduction from baseline
  injectionMovedPct: number; // % of injection episodes that moved any object >2cm
  baselineEpisodes: number;
  injectionEpisodes: number;
  bySuite?: Record<string, { meanMaxDisp: number; movedPct: number; n: number }>;
}

interface DisplacementData {
  modelName: string;
  sourceOverrideRate: number; // % of episodes more similar to source than destination
  overrideRateLabel: string;
  expertVsVlmRatio: number; // how much greater expert pathway displacement is
  keyFinding: string;
  episodeCount: number;
  objectDisplacement?: ObjectDisplacementData;
  details: {
    metric: string;
    value: string;
    description: string;
  }[];
}

const DISPLACEMENT_DATA: Partial<Record<VLAModelType, DisplacementData>> = {
  pi05: {
    modelName: "Pi0.5",
    sourceOverrideRate: 99.6,
    overrideRateLabel: "99.6% episodes reproduce source behavior",
    expertVsVlmRatio: 0,
    keyFinding: "Cross-task injection in the expert pathway reproduces source task behavior in 99.6% of 1,968 episodes (cosine similarity to source: 0.97, to destination: 0.13). The flow-matching action head faithfully executes whatever trajectory the expert layers encode.",
    episodeCount: 1968,
    objectDisplacement: {
      baselineMeanMaxDisp: 0.2285,
      injectionMeanMaxDisp: 0.1661,
      reductionPct: 27.3,
      injectionMovedPct: 69.9,
      baselineEpisodes: 328,
      injectionEpisodes: 2624,
      bySuite: {
        "libero_goal": { meanMaxDisp: 0.066, movedPct: 45.9, n: 270 },
        "libero_spatial": { meanMaxDisp: 0.075, movedPct: 52.9, n: 34 },
        "libero_10": { meanMaxDisp: 0.173, movedPct: 45.8, n: 24 },
      },
    },
    details: [
      { metric: "Source Override Rate", value: "99.6%", description: "1,960 of 1,968 injection episodes reproduced source behavior" },
      { metric: "Mean Cosine Sim (Source)", value: "0.97", description: "Average cosine similarity between injected and source trajectories" },
      { metric: "Mean Cosine Sim (Dest)", value: "0.13", description: "Average cosine similarity between injected and destination trajectories" },
      { metric: "Object Displacement", value: "27.3% reduction", description: "Injection reduces max object movement by 27.3% vs baseline (robot reproduces source, not destination manipulation)" },
      { metric: "Suites", value: "3", description: "LIBERO Goal (99.6%), Spatial (99.5%), LIBERO-10 (99.3%)" },
      { metric: "Success After Injection", value: "2.6%", description: "Near-zero success on destination task after source injection" },
    ],
  },
  openvla: {
    modelName: "OpenVLA-OFT",
    sourceOverrideRate: 77.9,
    overrideRateLabel: "77.9% episodes reproduce source behavior",
    expertVsVlmRatio: 0,
    keyFinding: "Cross-task injection overrides destination behavior in 77.9% of 1,079 episodes, with mean cosine similarity to source of 0.82 vs 0.38 to destination. The single-pathway architecture shows partial resistance compared to dual-pathway models.",
    episodeCount: 1079,
    objectDisplacement: {
      baselineMeanMaxDisp: 0.2620,
      injectionMeanMaxDisp: 0.1297,
      reductionPct: 50.5,
      injectionMovedPct: 56.2,
      baselineEpisodes: 47,
      injectionEpisodes: 438,
      bySuite: {
        "libero_goal": { meanMaxDisp: 0.1437, movedPct: 90.0, n: 20 },
        "libero_object": { meanMaxDisp: 0.1607, movedPct: 43.5, n: 23 },
        "libero_spatial": { meanMaxDisp: 0.0367, movedPct: 40.0, n: 15 },
        "libero_10": { meanMaxDisp: 0.1566, movedPct: 46.7, n: 15 },
      },
    },
    details: [
      { metric: "Source Override Rate", value: "77.9%", description: "841 of 1,079 injection episodes reproduced source behavior" },
      { metric: "Mean Cosine Sim (Source)", value: "0.82", description: "Average cosine similarity between injected and source trajectories" },
      { metric: "Mean Cosine Sim (Dest)", value: "0.38", description: "Average cosine similarity between injected and destination trajectories" },
      { metric: "Object Displacement", value: "50.5% reduction", description: "Injection halves max object movement vs baseline (0.13m vs 0.26m, all-layer injection)" },
      { metric: "Suites", value: "4", description: "Goal (81.9%), LIBERO-10 (78.1%), Object (78.1%), Spatial (73.7%)" },
      { metric: "Success After Injection", value: "5.0%", description: "Near-zero success on destination task after source injection" },
    ],
  },
  xvla: {
    modelName: "X-VLA",
    sourceOverrideRate: 99.8,
    overrideRateLabel: "99.8% episodes more similar to source",
    expertVsVlmRatio: 2.0,
    keyFinding: "Cross-task injection overwhelmingly reproduces source behavior. 99.8% of episodes show higher cosine similarity to the source task trajectory than the destination task.",
    episodeCount: 3150,
    details: [
      { metric: "Source Similarity", value: "99.8%", description: "Episodes with higher cosine similarity to source than destination" },
      { metric: "Mean Cosine Sim (Source)", value: "0.94", description: "Average cosine similarity between injected and source trajectories" },
      { metric: "Mean Cosine Sim (Dest)", value: "0.31", description: "Average cosine similarity between injected and destination trajectories" },
      { metric: "Layers Tested", value: "12, 20, 23", description: "SAE layers used for cross-task injection" },
      { metric: "Suites", value: "4", description: "LIBERO Goal, Object, Spatial, LIBERO-10" },
    ],
  },
  smolvla: {
    modelName: "SmolVLA",
    sourceOverrideRate: 15.8,
    overrideRateLabel: "Expert 15.8% vs VLM 9.0% override",
    expertVsVlmRatio: 1.76,
    keyFinding: "Expert pathway injections override destination behavior in 15.8% of episodes versus 9.0% for VLM pathway injections (732 pairs, 4 difficulty levels), confirming the expert pathway as the primary action computation pathway.",
    episodeCount: 732,
    details: [
      { metric: "Expert Override Rate", value: "15.8%", description: "Expert pathway injection override rate" },
      { metric: "VLM Override Rate", value: "9.0%", description: "VLM pathway injection override rate" },
      { metric: "Expert vs VLM Ratio", value: "1.76x", description: "Expert pathway causes 1.76x greater displacement than VLM" },
      { metric: "Pairs Tested", value: "732", description: "Source-destination task pairs across 4 difficulty levels" },
      { metric: "Benchmarks", value: "LIBERO + MetaWorld", description: "Cross-benchmark transfer tested" },
    ],
  },
};

// Placeholder data for models without specific displacement results
const DEFAULT_DISPLACEMENT: DisplacementData = {
  modelName: "",
  sourceOverrideRate: 0,
  overrideRateLabel: "No displacement data available",
  expertVsVlmRatio: 0,
  keyFinding: "No displacement data available for this model.",
  episodeCount: 0,
  details: [],
};

// Summary statistics across all models
const CROSS_MODEL_STATS = {
  totalModels: 6,
  totalEpisodes: "284,000+",
  totalSAEs: "388",
  totalConcepts: "82+",
  totalBenchmarks: 4,
  benchmarkNames: ["LIBERO", "MetaWorld", "SimplerEnv", "ALOHA"],
};

type ViewMode = "overview" | "comparison" | "trajectories";

export default function DisplacementAnalysis() {
  const [viewMode, setViewMode] = useState<ViewMode>("overview");
  const currentModel = useAppSelector((state) => state.model.currentModel);

  const data = DISPLACEMENT_DATA[currentModel] || { ...DEFAULT_DISPLACEMENT, modelName: currentModel.toUpperCase() };
  const hasData = currentModel in DISPLACEMENT_DATA;

  const handleViewChange = (_: React.MouseEvent<HTMLElement>, newView: ViewMode | null) => {
    if (newView) setViewMode(newView);
  };

  const renderStatCard = (label: string, value: string, color: string, subtitle?: string) => (
    <Box
      sx={{
        bgcolor: "rgba(15, 23, 42, 0.6)",
        borderRadius: 2,
        p: 2,
        textAlign: "center",
        minWidth: 120,
        border: `1px solid ${color}30`,
        flex: 1,
      }}
    >
      <Typography
        variant="h5"
        sx={{ color, fontWeight: 700, fontSize: "1.5rem", lineHeight: 1.2 }}
      >
        {value}
      </Typography>
      <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "0.7rem", display: "block", mt: 0.5 }}>
        {label}
      </Typography>
      {subtitle && (
        <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem", display: "block" }}>
          {subtitle}
        </Typography>
      )}
    </Box>
  );

  const renderOverview = () => (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {/* Cross-Study Summary */}
      <Box>
        <Typography
          variant="caption"
          sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 1 }}
        >
          Cross-Study Summary
        </Typography>
        <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
          {renderStatCard("Models", String(CROSS_MODEL_STATS.totalModels), "#3b82f6")}
          {renderStatCard("Episodes", CROSS_MODEL_STATS.totalEpisodes, "#ef4444")}
          {renderStatCard("SAEs Trained", CROSS_MODEL_STATS.totalSAEs, "#8b5cf6")}
          {renderStatCard("Concepts", CROSS_MODEL_STATS.totalConcepts, "#10b981")}
          {renderStatCard("Benchmarks", String(CROSS_MODEL_STATS.totalBenchmarks), "#f59e0b",
            CROSS_MODEL_STATS.benchmarkNames.join(", ")
          )}
        </Box>
      </Box>

      <Divider sx={{ borderColor: "#1e293b" }} />

      {/* Current Model Displacement */}
      <Box>
        <Typography
          variant="caption"
          sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 1 }}
        >
          {data.modelName} Displacement Analysis
        </Typography>

        {hasData ? (
          <>
            {/* Main Stat */}
            <Box sx={{
              bgcolor: "rgba(239, 68, 68, 0.08)",
              border: "1px solid rgba(239, 68, 68, 0.2)",
              borderRadius: 2,
              p: 2.5,
              mb: 2,
              textAlign: "center",
            }}>
              <Typography
                variant="h3"
                sx={{ color: "#ef4444", fontWeight: 800, fontSize: "2.5rem", lineHeight: 1 }}
              >
                {data.sourceOverrideRate}%
              </Typography>
              <Typography variant="body2" sx={{ color: "#cbd5e1", fontSize: "0.8rem", mt: 1 }}>
                {data.overrideRateLabel}
              </Typography>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.65rem", display: "block", mt: 0.5 }}>
                Based on {data.episodeCount.toLocaleString()} episodes
              </Typography>
            </Box>

            {/* Key Finding */}
            <Box sx={{
              bgcolor: "rgba(59, 130, 246, 0.08)",
              border: "1px solid rgba(59, 130, 246, 0.2)",
              borderRadius: 2,
              p: 2,
              mb: 2,
            }}>
              <Typography variant="caption" sx={{ color: "#3b82f6", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 0.5 }}>
                Key Finding
              </Typography>
              <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "0.75rem", lineHeight: 1.6 }}>
                {data.keyFinding}
              </Typography>
            </Box>

            {/* Detail Metrics */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {data.details.map((detail, idx) => (
                <Box
                  key={idx}
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    bgcolor: "rgba(15, 23, 42, 0.4)",
                    borderRadius: 1,
                    px: 2,
                    py: 1,
                  }}
                >
                  <Box>
                    <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "0.7rem", fontWeight: 600 }}>
                      {detail.metric}
                    </Typography>
                    <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem" }}>
                      {detail.description}
                    </Typography>
                  </Box>
                  <Chip
                    label={detail.value}
                    size="small"
                    sx={{
                      height: 22,
                      fontSize: "0.7rem",
                      fontWeight: 700,
                      bgcolor: "rgba(239, 68, 68, 0.15)",
                      color: "#ef4444",
                      border: "1px solid rgba(239, 68, 68, 0.3)",
                    }}
                  />
                </Box>
              ))}
            </Box>

            {/* Object Displacement Section */}
            {data.objectDisplacement && (
              <Box sx={{
                bgcolor: "rgba(16, 185, 129, 0.08)",
                border: "1px solid rgba(16, 185, 129, 0.2)",
                borderRadius: 2,
                p: 2,
                mt: 2,
              }}>
                <Typography variant="caption" sx={{ color: "#10b981", fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 1 }}>
                  Object Displacement (Physical)
                </Typography>

                {/* Baseline vs Injection comparison bar */}
                <Box sx={{ display: "flex", gap: 1.5, mb: 1.5 }}>
                  <Box sx={{ flex: 1, textAlign: "center", bgcolor: "rgba(15, 23, 42, 0.4)", borderRadius: 1, p: 1 }}>
                    <Typography variant="h6" sx={{ color: "#94a3b8", fontWeight: 700, fontSize: "1rem" }}>
                      {(data.objectDisplacement.baselineMeanMaxDisp * 100).toFixed(1)}cm
                    </Typography>
                    <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem", display: "block" }}>
                      Baseline
                    </Typography>
                    <Typography variant="caption" sx={{ color: "#475569", fontSize: "0.5rem" }}>
                      {data.objectDisplacement.baselineEpisodes} eps
                    </Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center" }}>
                    <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 700, fontSize: "0.7rem" }}>
                      {data.objectDisplacement.reductionPct > 0 ? "\u2193" : "\u2191"}{Math.abs(data.objectDisplacement.reductionPct)}%
                    </Typography>
                  </Box>
                  <Box sx={{ flex: 1, textAlign: "center", bgcolor: "rgba(15, 23, 42, 0.4)", borderRadius: 1, p: 1 }}>
                    <Typography variant="h6" sx={{ color: "#10b981", fontWeight: 700, fontSize: "1rem" }}>
                      {(data.objectDisplacement.injectionMeanMaxDisp * 100).toFixed(1)}cm
                    </Typography>
                    <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem", display: "block" }}>
                      After Injection
                    </Typography>
                    <Typography variant="caption" sx={{ color: "#475569", fontSize: "0.5rem" }}>
                      {data.objectDisplacement.injectionEpisodes} eps
                    </Typography>
                  </Box>
                </Box>

                {/* Per-suite breakdown */}
                {data.objectDisplacement.bySuite && (
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                    {Object.entries(data.objectDisplacement.bySuite).map(([suite, suiteData]) => (
                      <Tooltip key={suite} title={`${suiteData.n} episodes, ${suiteData.movedPct}% moved any object >2cm`} arrow>
                        <Chip
                          label={`${suite.replace("libero_", "")}: ${(suiteData.meanMaxDisp * 100).toFixed(1)}cm`}
                          size="small"
                          sx={{
                            height: 18,
                            fontSize: "0.55rem",
                            bgcolor: "rgba(16, 185, 129, 0.12)",
                            color: "#10b981",
                            border: "1px solid rgba(16, 185, 129, 0.25)",
                          }}
                        />
                      </Tooltip>
                    ))}
                  </Box>
                )}

                <Typography variant="caption" sx={{ color: "#cbd5e1", fontSize: "0.6rem", display: "block", mt: 1 }}>
                  Mean max object displacement per episode. Injection causes robot to reproduce source behavior, reducing interaction with destination objects.
                </Typography>
              </Box>
            )}
          </>
        ) : (
          <Box sx={{
            bgcolor: "rgba(245, 158, 11, 0.08)",
            border: "1px solid rgba(245, 158, 11, 0.2)",
            borderRadius: 2,
            p: 3,
            textAlign: "center",
          }}>
            <Typography variant="body2" sx={{ color: "#f59e0b", fontSize: "0.8rem" }}>
              {data.keyFinding}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );

  const renderComparison = () => (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography
        variant="caption"
        sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em" }}
      >
        Cross-Model Override Comparison
      </Typography>

      {/* Comparison bars */}
      {Object.entries(DISPLACEMENT_DATA).map(([modelId, modelData]) => {
        if (!modelData) return null;
        const isCurrentModel = modelId === currentModel;
        return (
          <Box
            key={modelId}
            sx={{
              bgcolor: isCurrentModel ? "rgba(239, 68, 68, 0.08)" : "rgba(15, 23, 42, 0.4)",
              border: isCurrentModel ? "1px solid rgba(239, 68, 68, 0.3)" : "1px solid rgba(30, 41, 59, 0.5)",
              borderRadius: 2,
              p: 2,
            }}
          >
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" sx={{ color: "#f1f5f9", fontWeight: 600, fontSize: "0.8rem" }}>
                  {modelData.modelName}
                </Typography>
                {isCurrentModel && (
                  <Chip label="Selected" size="small" sx={{ height: 16, fontSize: "0.5rem", bgcolor: "#ef4444", color: "white" }} />
                )}
              </Box>
              <Typography variant="body2" sx={{ color: "#ef4444", fontWeight: 700, fontSize: "0.9rem" }}>
                {modelData.sourceOverrideRate}%
              </Typography>
            </Box>

            {/* Bar */}
            <Box sx={{ width: "100%", height: 8, bgcolor: "rgba(15, 23, 42, 0.6)", borderRadius: 4, overflow: "hidden" }}>
              <Box
                sx={{
                  width: `${modelData.sourceOverrideRate}%`,
                  height: "100%",
                  bgcolor: isCurrentModel ? "#ef4444" : "#3b82f6",
                  borderRadius: 4,
                  transition: "width 0.5s ease",
                }}
              />
            </Box>

            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mt: 0.5 }}>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem" }}>
                {modelData.overrideRateLabel} ({modelData.episodeCount.toLocaleString()} episodes)
              </Typography>
              {modelData.objectDisplacement && (
                <Tooltip title={`Object displacement reduced ${modelData.objectDisplacement.reductionPct}% vs baseline (${(modelData.objectDisplacement.baselineMeanMaxDisp * 100).toFixed(1)}cm → ${(modelData.objectDisplacement.injectionMeanMaxDisp * 100).toFixed(1)}cm)`} arrow>
                  <Chip
                    label={`obj: ${modelData.objectDisplacement.reductionPct > 0 ? "\u2193" : "\u2191"}${Math.abs(modelData.objectDisplacement.reductionPct)}%`}
                    size="small"
                    sx={{
                      height: 16,
                      fontSize: "0.5rem",
                      bgcolor: "rgba(16, 185, 129, 0.12)",
                      color: "#10b981",
                      border: "1px solid rgba(16, 185, 129, 0.25)",
                    }}
                  />
                </Tooltip>
              )}
            </Box>
          </Box>
        );
      })}

      {/* Expert vs VLM comparison */}
      <Box sx={{
        bgcolor: "rgba(139, 92, 246, 0.08)",
        border: "1px solid rgba(139, 92, 246, 0.2)",
        borderRadius: 2,
        p: 2,
      }}>
        <Typography variant="caption" sx={{ color: "#8b5cf6", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 1 }}>
          Expert vs VLM Pathway Displacement
        </Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Box sx={{ flex: 1, textAlign: "center" }}>
            <Typography variant="h5" sx={{ color: "#ef4444", fontWeight: 700 }}>2x</Typography>
            <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "0.6rem" }}>Expert Pathway</Typography>
          </Box>
          <Divider orientation="vertical" flexItem sx={{ borderColor: "#1e293b" }} />
          <Box sx={{ flex: 1, textAlign: "center" }}>
            <Typography variant="h5" sx={{ color: "#64748b", fontWeight: 700 }}>1x</Typography>
            <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "0.6rem" }}>VLM Pathway</Typography>
          </Box>
        </Box>
        <Typography variant="caption" sx={{ color: "#cbd5e1", fontSize: "0.6rem", display: "block", mt: 1, textAlign: "center" }}>
          Expert pathways cause 2x greater behavioral displacement than VLM pathways
        </Typography>
      </Box>
    </Box>
  );

  const renderTrajectories = () => (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography
        variant="caption"
        sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em" }}
      >
        Trajectory Displacement Visualization
      </Typography>

      {/* Schematic diagram of injection effect */}
      <Box sx={{
        bgcolor: "rgba(15, 23, 42, 0.6)",
        borderRadius: 2,
        p: 3,
        border: "1px solid rgba(30, 41, 59, 0.5)",
      }}>
        {/* Source Task */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
          <Box sx={{ width: 12, height: 12, borderRadius: "50%", bgcolor: "#22c55e", flexShrink: 0 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" sx={{ color: "#22c55e", fontWeight: 600, fontSize: "0.75rem" }}>
              Source Task Trajectory
            </Typography>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem" }}>
              Original behavior pattern of the source task
            </Typography>
          </Box>
          <Box sx={{
            height: 4,
            flex: 2,
            background: "linear-gradient(90deg, #22c55e, #22c55e80)",
            borderRadius: 2,
          }} />
        </Box>

        {/* Injected Result */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
          <Box sx={{ width: 12, height: 12, borderRadius: "50%", bgcolor: "#ef4444", flexShrink: 0 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" sx={{ color: "#ef4444", fontWeight: 600, fontSize: "0.75rem" }}>
              Injected Trajectory
            </Typography>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem" }}>
              Behavior after cross-task feature injection
            </Typography>
          </Box>
          <Box sx={{
            height: 4,
            flex: 2,
            background: "linear-gradient(90deg, #ef4444, #ef444480)",
            borderRadius: 2,
          }} />
        </Box>

        {/* Destination Task */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Box sx={{ width: 12, height: 12, borderRadius: "50%", bgcolor: "#3b82f6", flexShrink: 0 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" sx={{ color: "#3b82f6", fontWeight: 600, fontSize: "0.75rem" }}>
              Destination Task Trajectory
            </Typography>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem" }}>
              Expected behavior of the destination task
            </Typography>
          </Box>
          <Box sx={{
            height: 4,
            flex: 2,
            background: "linear-gradient(90deg, #3b82f6, #3b82f680)",
            borderRadius: 2,
          }} />
        </Box>

        {/* Similarity indicator */}
        <Divider sx={{ borderColor: "#1e293b", my: 2 }} />
        <Box sx={{ display: "flex", justifyContent: "center", gap: 3 }}>
          <Tooltip title="Cosine similarity between injected and source trajectories" arrow>
            <Box sx={{ textAlign: "center" }}>
              <Typography variant="h6" sx={{ color: "#22c55e", fontWeight: 700, fontSize: "1.1rem" }}>
                {hasData ? (data.details.find(d => d.metric.includes("Cosine") && d.metric.includes("Source"))?.value.replace("%", "") || "0.94") : "0.94"}
              </Typography>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem" }}>
                cos(injected, source)
              </Typography>
            </Box>
          </Tooltip>
          <Box sx={{ display: "flex", alignItems: "center" }}>
            <CompareArrowsIcon sx={{ color: "#475569", fontSize: 20 }} />
          </Box>
          <Tooltip title="Cosine similarity between injected and destination trajectories" arrow>
            <Box sx={{ textAlign: "center" }}>
              <Typography variant="h6" sx={{ color: "#3b82f6", fontWeight: 700, fontSize: "1.1rem" }}>
                {hasData ? (data.details.find(d => d.metric.includes("Cosine") && d.metric.includes("Dest"))?.value.replace("%", "") || "0.31") : "0.31"}
              </Typography>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem" }}>
                cos(injected, dest)
              </Typography>
            </Box>
          </Tooltip>
        </Box>

        {/* Object displacement in trajectory view */}
        {data.objectDisplacement && (
          <>
            <Divider sx={{ borderColor: "#1e293b", my: 2 }} />
            <Box sx={{ display: "flex", justifyContent: "center", gap: 3 }}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h6" sx={{ color: "#94a3b8", fontWeight: 700, fontSize: "1.1rem" }}>
                  {(data.objectDisplacement.baselineMeanMaxDisp * 100).toFixed(1)}cm
                </Typography>
                <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem" }}>
                  baseline obj disp
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center" }}>
                <Typography variant="caption" sx={{ color: "#10b981", fontWeight: 700 }}>
                  {"\u2192"}
                </Typography>
              </Box>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h6" sx={{ color: "#10b981", fontWeight: 700, fontSize: "1.1rem" }}>
                  {(data.objectDisplacement.injectionMeanMaxDisp * 100).toFixed(1)}cm
                </Typography>
                <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.55rem" }}>
                  injected obj disp
                </Typography>
              </Box>
            </Box>
          </>
        )}
      </Box>

      {/* Interpretation */}
      <Box sx={{
        bgcolor: "rgba(16, 185, 129, 0.08)",
        border: "1px solid rgba(16, 185, 129, 0.2)",
        borderRadius: 2,
        p: 2,
      }}>
        <Typography variant="caption" sx={{ color: "#10b981", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 0.5 }}>
          Interpretation
        </Typography>
        <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "0.7rem", lineHeight: 1.6 }}>
          When SAE features from a source task are injected into a destination task rollout, the resulting robot trajectory closely matches the source task behavior -- not the destination. This confirms that SAE features encode <strong>causal action programs</strong>, not just correlational patterns. The features are sufficient to redirect robot behavior.
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Paper className="h-full flex flex-col overflow-hidden rounded-lg shadow-md" sx={{ bgcolor: "#0f172a" }}>
      {/* Header */}
      <Box sx={{
        height: 40,
        display: "flex",
        alignItems: "center",
        px: 2,
        bgcolor: "#0a1628",
        borderBottom: "1px solid #1e293b",
        gap: 1,
        flexShrink: 0,
      }}>
        <RouteIcon sx={{ fontSize: 16, color: "#ef4444" }} />
        <Typography variant="body2" sx={{ color: "white", fontWeight: 600, fontSize: "0.8rem" }}>
          Displacement Analysis
        </Typography>
        <Chip
          label={hasData ? data.modelName : currentModel.toUpperCase()}
          size="small"
          sx={{ height: 18, fontSize: "0.55rem", bgcolor: "#ef4444", color: "white", ml: 1 }}
        />
        <Box sx={{ ml: "auto" }}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={handleViewChange}
            size="small"
            sx={{
              "& .MuiToggleButton-root": {
                color: "#64748b",
                borderColor: "#334155",
                fontSize: "0.6rem",
                py: 0.25,
                px: 1,
                textTransform: "none",
                "&.Mui-selected": {
                  bgcolor: "rgba(239, 68, 68, 0.15)",
                  color: "#ef4444",
                  borderColor: "#ef4444",
                },
              },
            }}
          >
            <ToggleButton value="overview">
              <TrendingUpIcon sx={{ fontSize: 12, mr: 0.5 }} />
              Overview
            </ToggleButton>
            <ToggleButton value="comparison">
              <CompareArrowsIcon sx={{ fontSize: 12, mr: 0.5 }} />
              Compare
            </ToggleButton>
            <ToggleButton value="trajectories">
              <RouteIcon sx={{ fontSize: 12, mr: 0.5 }} />
              Trajectories
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Content */}
      <Box sx={{ flex: 1, overflow: "auto", p: 2 }}>
        {viewMode === "overview" && renderOverview()}
        {viewMode === "comparison" && renderComparison()}
        {viewMode === "trajectories" && renderTrajectories()}
      </Box>
    </Paper>
  );
}
