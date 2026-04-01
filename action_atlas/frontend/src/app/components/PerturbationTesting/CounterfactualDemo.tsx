"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Box, Typography, Select, MenuItem, FormControl, InputLabel, Chip, LinearProgress, Card, CardContent, Grid, Button, CircularProgress } from "@mui/material";
import { useAppSelector } from "@/redux/hooks";
import { DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
import { API_BASE_URL } from "@/config/api";

interface TaskResult {
  prompt: string;
  success: boolean;
  n_steps: number;
}

interface CounterfactualData {
  model: string;
  suite: string;
  prompt_types: string[];
  tasks: Record<string, Record<string, TaskResult>>;
  summary: Record<string, { success_rate: number; successes: number; total: number }>;
  videos: Record<string, string>;
  available_suites: string[];
  key_finding?: string;
}

interface CounterfactualVideo {
  path: string;
  experiment_type: string;
  suite: string;
  task: number | null;
  success: boolean | null;
  subtype?: string;
  [key: string]: unknown;
}

const PROMPT_LABELS: Record<string, { label: string; color: string; desc: string }> = {
  original: { label: "Original", color: "#22c55e", desc: "Exact task description" },
  generic: { label: "Generic", color: "#eab308", desc: '"complete the task"' },
  wrong: { label: "Wrong", color: "#ef4444", desc: '"do nothing and stay still"' },
  // Pi0.5 counterfactual categories
  baseline: { label: "Baseline", color: "#22c55e", desc: "Original prompt (correct)" },
  null: { label: "Null", color: "#eab308", desc: "Empty/null prompt" },
  object_swap: { label: "Object Swap", color: "#f97316", desc: "Swapped target object" },
  verb_swap: { label: "Verb Swap", color: "#ef4444", desc: "Swapped action verb" },
  motor: { label: "Motor", color: "#dc2626", desc: "Motor command prompt" },
  negation: { label: "Negation", color: "#a855f7", desc: "Negated instruction" },
  spatial_swap: { label: "Spatial Swap", color: "#3b82f6", desc: "Swapped spatial reference" },
  wrong_object: { label: "Wrong Object", color: "#ec4899", desc: "Incorrect target object" },
  conflict: { label: "Conflict", color: "#6366f1", desc: "Conflicting instructions" },
  object_only: { label: "Object Only", color: "#14b8a6", desc: "Object name only (no verb)" },
  // GR00T counterfactual categories
  null_prompt: { label: "Null Prompt", color: "#eab308", desc: "Empty/null prompt" },
  opposite: { label: "Opposite", color: "#f97316", desc: "Opposite instruction" },
  random: { label: "Random", color: "#a855f7", desc: "Random prompt" },
  wrong_task: { label: "Wrong Task", color: "#ef4444", desc: "Wrong task description" },
  cross_prompt: { label: "Cross Prompt", color: "#6366f1", desc: "Cross-task prompt" },
  // X-VLA counterfactual categories
  nonsense: { label: "Nonsense", color: "#a855f7", desc: "Random nonsense text" },
  numbers: { label: "Numbers", color: "#94a3b8", desc: "Numeric string prompt" },
  greeting: { label: "Greeting", color: "#06b6d4", desc: "Greeting text" },
  stop: { label: "Stop", color: "#dc2626", desc: '"stop" command' },
  freeze: { label: "Freeze", color: "#dc2626", desc: '"freeze" command' },
  do_nothing: { label: "Do Nothing", color: "#ef4444", desc: '"do nothing" command' },
  open_gripper: { label: "Open Gripper", color: "#f97316", desc: "Motor: open gripper" },
  close_gripper: { label: "Close Gripper", color: "#f97316", desc: "Motor: close gripper" },
  move_left: { label: "Move Left", color: "#3b82f6", desc: "Motor: move left" },
  move_right: { label: "Move Right", color: "#3b82f6", desc: "Motor: move right" },
  move_up: { label: "Move Up", color: "#3b82f6", desc: "Motor: move up" },
  move_down: { label: "Move Down", color: "#3b82f6", desc: "Motor: move down" },
  move_forward: { label: "Move Forward", color: "#3b82f6", desc: "Motor: move forward" },
  move_backward: { label: "Move Backward", color: "#3b82f6", desc: "Motor: move backward" },
};

export default function CounterfactualDemo() {
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);
  const effectiveModel = currentModel;

  // Derive available suites from the current dataset
  const datasetSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
  const defaultSuite = datasetSuites.length > 0 ? datasetSuites[0] : 'libero_goal';
  const [suite, setSuite] = useState(defaultSuite);

  // Reset suite when dataset changes — force refetch with first suite of new dataset
  useEffect(() => {
    const newSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
    if (newSuites.length > 0) {
      setSuite(newSuites[0]);
    }
  }, [currentDataset]);
  const [data, setData] = useState<CounterfactualData | null>(null);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [videos, setVideos] = useState<CounterfactualVideo[]>([]);
  const [videosLoading, setVideosLoading] = useState(false);
  const [visibleCount, setVisibleCount] = useState(24);

  // Helper: extract counterfactual condition/subtype from filename when not provided
  const extractSubtypeFromFilename = (filename: string): string => {
    if (!filename) return '';
    const stem = filename.replace(/\.mp4$/i, '');
    // Patterns: "baseline_ep0", "generic_ep0", "wrong_ep0", "null_ep0"
    // Also: "original_ep0", "negation_ep0", "motor_ep0", "cross_prompt_ep0"
    const knownConditions = ['baseline', 'generic', 'wrong', 'original', 'null', 'negation',
      'motor', 'cross_prompt', 'object_swap', 'verb_swap', 'spatial_swap', 'wrong_object',
      'conflict', 'object_only', 'cross', 'null_prompt', 'opposite', 'random', 'wrong_task'];
    for (const cond of knownConditions) {
      if (stem.startsWith(cond + '_') || stem === cond) {
        return cond;
      }
    }
    // SmolVLA pattern: "taskA_cross_taskB_seed" -> "cross"
    if (stem.includes('_cross_')) return 'cross_task';
    return '';
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/vla/counterfactual?model=${effectiveModel}&suite=${suite}`);
      const json = await res.json();
      const hasData = json.prompt_types?.length > 0 || Object.keys(json.tasks || {}).length > 0;
      if (hasData) {
        setData(json);
      } else if (!json.status || json.status !== 404) {
        // API returned a response but with empty tasks/prompts.
        // Auto-switch to an available suite if current suite has no data
        if (json.available_suites && json.available_suites.length > 0 && !json.available_suites.includes(suite)) {
          setSuite(json.available_suites[0]);
          setLoading(false);
          return;
        }
        // Build summary from counterfactual videos if available.
        const suiteVariants = [suite, suite.replace("libero_", "")];
        let cfVideos: any[] = [];
        for (const suiteVar of suiteVariants) {
          try {
            const vidRes = await fetch(
              `${API_BASE_URL}/api/vla/videos?model=${effectiveModel}&experiment_type=counterfactual&suite=${suiteVar}&limit=1000`
            );
            if (vidRes.ok) {
              const vidData = await vidRes.json();
              cfVideos = vidData.videos || [];
              if (cfVideos.length > 0) break;
            }
          } catch { /* ignore */ }
        }
        if (cfVideos.length > 0) {
          // Extract prompt types from filenames
          const promptCounts: Record<string, { total: number; successes: number }> = {};
          const taskMap: Record<string, Record<string, { prompt: string; success: boolean; n_steps: number }>> = {};
          for (const v of cfVideos) {
            const fn = (v.filename || v.path?.split('/').pop() || '').replace(/\.mp4$/i, '');
            const condition = extractSubtypeFromFilename(fn) || fn.replace(/_ep\d+$/, '');
            if (!condition) continue;
            if (!promptCounts[condition]) promptCounts[condition] = { total: 0, successes: 0 };
            promptCounts[condition].total++;
            if (v.success) promptCounts[condition].successes++;
            const taskId = String(v.task ?? 'unknown');
            if (!taskMap[taskId]) taskMap[taskId] = {};
            if (!taskMap[taskId][condition]) {
              taskMap[taskId][condition] = { prompt: condition, success: v.success ?? false, n_steps: 0 };
            }
          }
          const promptTypes = Object.keys(promptCounts).sort();
          const summary: Record<string, { success_rate: number; successes: number; total: number }> = {};
          for (const [pt, counts] of Object.entries(promptCounts)) {
            summary[pt] = {
              success_rate: counts.total > 0 ? counts.successes / counts.total : 0,
              successes: counts.successes,
              total: counts.total,
            };
          }
          setData({
            model: effectiveModel,
            suite,
            prompt_types: promptTypes,
            tasks: taskMap,
            summary,
            videos: {},
            available_suites: json.available_suites || [],
            key_finding: `${cfVideos.length} counterfactual rollout videos across ${promptTypes.length} conditions for ${effectiveModel}.`,
          });
        } else {
          setData(json);
        }
      } else {
        setData(null);
      }
    } catch { setData(null); }
    setLoading(false);
  }, [suite, effectiveModel]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const fetchVideos = useCallback(async () => {
    setVideosLoading(true);
    setVisibleCount(24);
    try {
      // Try both full suite name and shortened variant (Pi0.5 index uses shortened names)
      const suiteVariants = [suite, suite.replace("libero_", "")];
      let found: CounterfactualVideo[] = [];
      for (const suiteVar of suiteVariants) {
        const res = await fetch(
          `${API_BASE_URL}/api/vla/videos?model=${effectiveModel}&experiment_type=counterfactual&suite=${suiteVar}`
        );
        if (!res.ok) continue;
        const json = await res.json();
        const vids: CounterfactualVideo[] = json.videos || [];
        if (vids.length > 0) {
          // Post-process: extract subtype from filename when missing
          for (const v of vids) {
            if (!v.subtype) {
              const fn = (v as any).filename || v.path?.split('/').pop() || '';
              v.subtype = extractSubtypeFromFilename(fn);
            }
          }
          found = vids;
          break;
        }
      }
      // If no counterfactual videos found via experiment_type filter, also try
      // the ablation videos endpoint (GR00T counterfactual videos are indexed there)
      if (found.length === 0) {
        try {
          const ablRes = await fetch(
            `${API_BASE_URL}/api/ablation/videos?model=${effectiveModel}&limit=5000`
          );
          if (ablRes.ok) {
            const ablData = await ablRes.json();
            const ablVids = ablData.data?.videos || ablData.videos || [];
            const cfVids = ablVids.filter(
              (v: any) => v.experiment_type === 'counterfactual' &&
                (!v.suite || suiteVariants.includes(v.suite))
            );
            if (cfVids.length > 0) {
              for (const v of cfVids) {
                if (!v.subtype) {
                  const fn = v.filename || v.path?.split('/').pop() || '';
                  v.subtype = extractSubtypeFromFilename(fn);
                }
              }
              found = cfVids;
            }
          }
        } catch {
          // Ignore ablation fallback failures
        }
      }
      setVideos(found);
    } catch {
      setVideos([]);
    }
    setVideosLoading(false);
  }, [suite, effectiveModel]);

  useEffect(() => { fetchVideos(); }, [fetchVideos]);

  const getVideoUrl = (videoPath: string): string => {
    if (videoPath.startsWith("http") || videoPath.startsWith("/api/")) {
      return videoPath;
    }
    // Ensure model prefix for proper resolution
    const modelPrefix = `${effectiveModel}/`;
    const prefixedPath =
      videoPath.startsWith(modelPrefix) || videoPath.startsWith(`${effectiveModel}_`)
        ? videoPath
        : `${modelPrefix}${videoPath}`;
    return `${API_BASE_URL}/api/vla/video/${prefixedPath}`;
  };

  const visibleVideos = videos.slice(0, visibleCount);

  const taskIds = data ? Object.keys(data.tasks).sort((a, b) => parseInt(a) - parseInt(b)) : [];

  return (
    <Box>
      <Typography variant="body2" sx={{ color: "#94a3b8", mb: 2 }}>
        Test how the robot responds to different prompts. <strong>Original</strong> = exact task description,{" "}
        <strong>Generic</strong> = &quot;complete the task&quot;, <strong>Wrong</strong> = &quot;do nothing&quot;.
        See if the model actually uses language or ignores it.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel sx={{ color: "#64748b" }}>Suite</InputLabel>
          <Select value={suite} label="Suite" onChange={(e) => setSuite(e.target.value)}
            sx={{ color: "#e2e8f0", ".MuiOutlinedInput-notchedOutline": { borderColor: "#334155" } }}>
            {(datasetSuites.length > 0 ? datasetSuites : ["libero_goal", "libero_10", "libero_object", "libero_spatial"]).map(s => (
              <MenuItem key={s} value={s}>{s.replace(/_/g, ' ')}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2, "& .MuiLinearProgress-bar": { bgcolor: "#ef4444" } }} />}

      {data && (
        <>
          {/* Summary bar chart */}
          <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
            {data.prompt_types.map(pt => {
              const info = PROMPT_LABELS[pt] || { label: pt, color: "#94a3b8", desc: pt };
              const sr = data.summary[pt]?.success_rate ?? 0;
              return (
                <Box key={pt} sx={{ bgcolor: "#1e293b", borderRadius: 1, p: 2, minWidth: 150, flex: 1 }}>
                  <Typography variant="caption" sx={{ color: "#64748b", fontWeight: 600, display: "block", mb: 0.5 }}>
                    {info.label}
                  </Typography>
                  <Typography sx={{ color: info.color, fontWeight: 700, fontSize: "28px", mb: 0.5 }}>
                    {(sr * 100).toFixed(0)}%
                  </Typography>
                  <Box sx={{ width: "100%", bgcolor: "#0f172a", borderRadius: 1, height: 8, overflow: "hidden" }}>
                    <Box sx={{ width: `${sr * 100}%`, bgcolor: info.color, height: "100%", borderRadius: 1 }} />
                  </Box>
                  <Typography variant="caption" sx={{ color: "#475569", mt: 0.5, display: "block", fontSize: "9px" }}>
                    {info.desc} ({data.summary[pt]?.successes}/{data.summary[pt]?.total})
                  </Typography>
                </Box>
              );
            })}
          </Box>

          {/* Task grid */}
          <Typography variant="subtitle2" sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}>
            Per-Task Results (click to expand)
          </Typography>
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 1 }}>
            {taskIds.map(taskId => {
              const taskData = data.tasks[taskId];
              const isExpanded = selectedTask === taskId;
              return (
                <Box key={taskId}
                  onClick={() => setSelectedTask(isExpanded ? null : taskId)}
                  sx={{
                    bgcolor: isExpanded ? "#1e293b" : "#0f172a", borderRadius: 1, p: 1.5,
                    cursor: "pointer", border: isExpanded ? "1px solid #ef4444" : "1px solid #1e293b",
                    transition: "all 0.15s", "&:hover": { bgcolor: "#1e293b" }
                  }}
                >
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
                    <Typography sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "13px" }}>
                      Task {taskId}
                    </Typography>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      {data.prompt_types.map(pt => {
                        const result = taskData[pt];
                        if (!result) return null;
                        const info = PROMPT_LABELS[pt] || { label: pt, color: "#94a3b8", desc: "" };
                        return (
                          <Box key={pt} sx={{
                            width: 12, height: 12, borderRadius: "50%",
                            bgcolor: result.success ? info.color : "#dc262640",
                            border: result.success ? "none" : `1px solid ${info.color}40`,
                          }} />
                        );
                      })}
                    </Box>
                  </Box>

                  {isExpanded && (
                    <Box sx={{ mt: 1 }}>
                      {data.prompt_types.map(pt => {
                        const result = taskData[pt];
                        if (!result) return null;
                        const info = PROMPT_LABELS[pt] || { label: pt, color: "#94a3b8", desc: "" };
                        return (
                          <Box key={pt} sx={{ mb: 1, pl: 1, borderLeft: `2px solid ${info.color}` }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <Typography variant="caption" sx={{ color: info.color, fontWeight: 600 }}>
                                {info.label}
                              </Typography>
                              <Chip label={result.success ? "SUCCESS" : "FAIL"} size="small"
                                sx={{ height: 16, fontSize: "9px", fontWeight: 700,
                                  bgcolor: result.success ? "#22c55e20" : "#dc262620",
                                  color: result.success ? "#22c55e" : "#ef4444" }} />
                            </Box>
                            <Typography variant="caption" sx={{ color: "#475569", fontSize: "10px", display: "block" }}>
                              &quot;{result.prompt}&quot;
                            </Typography>
                            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px" }}>
                              {result.n_steps} steps
                            </Typography>
                          </Box>
                        );
                      })}
                    </Box>
                  )}
                </Box>
              );
            })}
          </Box>

          {/* Key finding */}
          <Box sx={{ mt: 3, p: 2, bgcolor: "#1e293b", borderRadius: 1, borderLeft: "3px solid #eab308" }}>
            <Typography variant="caption" sx={{ color: "#eab308", fontWeight: 700, display: "block", mb: 0.5 }}>
              KEY FINDING
            </Typography>
            <Typography variant="body2" sx={{ color: "#94a3b8", fontSize: "12px" }}>
              {data.key_finding || (
                <>Language sensitivity is <strong>suite-dependent</strong>: libero_goal shows strong sensitivity
                (84% → 9% with wrong prompt), while libero_object shows <strong>zero</strong> sensitivity
                (84.6% regardless of prompt). The model uses language only when visual scene alone is ambiguous.</>
              )}
            </Typography>
          </Box>
        </>
      )}

      {/* Rollout Videos section */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ color: "#94a3b8", mb: 1.5, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}>
          Rollout Videos {videos.length > 0 && `(${videos.length})`}
        </Typography>

        {videosLoading && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 3, justifyContent: "center" }}>
            <CircularProgress size={20} sx={{ color: "#ef4444" }} />
            <Typography variant="body2" sx={{ color: "#64748b" }}>
              Loading videos...
            </Typography>
          </Box>
        )}

        {!videosLoading && videos.length === 0 && (
          <Box sx={{ py: 3, textAlign: "center" }}>
            <Typography variant="body2" sx={{ color: "#475569" }}>
              No counterfactual rollout videos available for this suite.
            </Typography>
          </Box>
        )}

        {!videosLoading && videos.length > 0 && (
          <>
            <Grid container spacing={1.5}>
              {visibleVideos.map((video, idx) => (
                <Grid item xs={12} sm={6} md={4} key={`${video.path}-${idx}`}>
                  <Card sx={{
                    bgcolor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 1,
                    overflow: "hidden",
                    transition: "border-color 0.15s",
                    "&:hover": { borderColor: "#334155" },
                  }}>
                    <Box sx={{ position: "relative", bgcolor: "#000" }}>
                      <video
                        src={getVideoUrl(video.path)}
                        controls
                        muted
                        loop
                        playsInline
                        style={{
                          width: "100%",
                          height: 160,
                          objectFit: "contain",
                          display: "block",
                        }}
                      />
                      {video.success !== null && video.success !== undefined && (
                        <Chip
                          label={video.success ? "SUCCESS" : "FAIL"}
                          size="small"
                          sx={{
                            position: "absolute",
                            top: 6,
                            right: 6,
                            height: 18,
                            fontSize: "9px",
                            fontWeight: 700,
                            bgcolor: video.success ? "rgba(34, 197, 94, 0.9)" : "rgba(239, 68, 68, 0.9)",
                            color: "#fff",
                          }}
                        />
                      )}
                    </Box>
                    <CardContent sx={{ p: 1, "&:last-child": { pb: 1 } }}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <Typography variant="caption" sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "11px" }}>
                          {video.task !== null && video.task !== undefined ? `Task ${video.task}` : "Unknown Task"}
                        </Typography>
                        {video.subtype && (
                          <Chip
                            label={PROMPT_LABELS[video.subtype]?.label || video.subtype.replace(/_/g, ' ')}
                            size="small"
                            sx={{
                              height: 16,
                              fontSize: "8px",
                              bgcolor: PROMPT_LABELS[video.subtype]?.color ? `${PROMPT_LABELS[video.subtype].color}30` : "#1e293b",
                              color: PROMPT_LABELS[video.subtype]?.color || "#94a3b8",
                              fontWeight: 600,
                            }}
                          />
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>

            {videos.length > visibleCount && (
              <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setVisibleCount((prev) => prev + 24)}
                  sx={{
                    color: "#94a3b8",
                    borderColor: "#334155",
                    fontSize: "11px",
                    textTransform: "none",
                    "&:hover": { borderColor: "#ef4444", color: "#ef4444" },
                  }}
                >
                  Show More ({videos.length - visibleCount} remaining)
                </Button>
              </Box>
            )}
          </>
        )}
      </Box>
    </Box>
  );
}
