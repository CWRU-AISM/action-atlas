"use client";
import React, { useState, useEffect, useCallback, useMemo } from "react";
import { Box, Typography, Chip, Tooltip, CircularProgress } from "@mui/material";
import { API_BASE_URL } from "@/config/api";

// ─── Types ───────────────────────────────────────────────────────────────────

interface GridCell {
  row: number;
  col: number;
  success_rate: number;
  mean_reward: number;
  std_reward?: number;
  n_episodes: number;
  has_videos?: boolean;
  bbox?: number[];
}

interface GridData {
  model: string;
  task: string;
  grid_size: number;
  grid: Record<string, GridCell>;
  baseline?: { success_rate: number; mean_reward: number; n_episodes: number };
  noise?: Record<string, { success_rate: number; mean_reward: number; n_episodes?: number }>;
  critical_cell?: string;
  best_cell?: string;
}

interface InjectionCondition {
  success: boolean;
  reward: number;
  layers?: string[];
  cos_to_baseline?: number;
}

interface InjectionPair {
  seed: number;
  conditions: Record<string, InjectionCondition>;
  source?: { success: boolean; reward: number };
}

interface InjectionData {
  source_env: string;
  target_env: string;
  pairs: InjectionPair[];
}

// ─── Color helpers ───────────────────────────────────────────────────────────

function srColor(sr: number, baseline: number): string {
  if (sr <= 0.15) return "#dc2626";
  if (sr < baseline * 0.5) return "#f97316";
  if (sr < baseline * 0.8) return "#eab308";
  if (sr >= baseline) return "#22c55e";
  return "#84cc16";
}

function deltaColor(delta: number): string {
  if (delta <= -30) return "#dc2626";
  if (delta <= -10) return "#f97316";
  if (delta < 0) return "#eab308";
  if (delta === 0) return "#94a3b8";
  return "#22c55e";
}

// ─── Task selector pills ─────────────────────────────────────────────────────

const TASKS = [
  { id: "AlohaInsertion-v0", label: "Insertion", baseline: 0.8 },
  { id: "AlohaTransferCube-v0", label: "Transfer Cube", baseline: 1.0 },
];

// ─── Main component ─────────────────────────────────────────────────────────

export default function ACTLayerVisualization() {
  const [selectedTask, setSelectedTask] = useState(TASKS[0].id);
  const [gridData, setGridData] = useState<GridData | null>(null);
  const [injectionData, setInjectionData] = useState<InjectionData | null>(null);
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [injectionLoading, setInjectionLoading] = useState(true);

  // Fetch grid ablation data
  const fetchGrid = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/vla/grid_ablation?model=act&task=${selectedTask}`
      );
      if (!res.ok) {
        setGridData(null);
        setLoading(false);
        return;
      }
      const json = await res.json();
      // Backend returns grid data directly on success (no wrapping status field),
      // or {status: 404, error: ...} on failure.
      if (json.status === 404 || !json.grid) {
        setGridData(null);
      } else {
        setGridData(json);
      }
    } catch {
      setGridData(null);
    }
    setLoading(false);
  }, [selectedTask]);

  // Fetch injection data
  const fetchInjection = useCallback(async () => {
    setInjectionLoading(true);
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/vla/act_results?experiment=injection`
      );
      const json = await res.json();
      if (json.status === 200 && json.data) {
        // Load cross-task injection separately
        const crossRes = await fetch(
          `${API_BASE_URL}/api/vla/injection?model=act`
        ).catch(() => null);
        if (crossRes && crossRes.ok) {
          const crossJson = await crossRes.json();
          setInjectionData(crossJson.data || crossJson);
        }
      }
    } catch {
      // Try direct file load
    }
    setInjectionLoading(false);
  }, []);

  useEffect(() => {
    fetchGrid();
  }, [fetchGrid]);

  useEffect(() => {
    fetchInjection();
  }, [fetchInjection]);

  const baselineSR = gridData?.baseline?.success_rate ?? TASKS.find(t => t.id === selectedTask)?.baseline ?? 1.0;

  // Compute per-row and per-column averages (use -1 for "no data" so the
  // display renders "--" instead of a misleading "0%")
  const rowAvgs = useMemo(() => {
    if (!gridData?.grid || Object.keys(gridData.grid).length === 0) return [-1, -1, -1, -1];
    return [0, 1, 2, 3].map((row) => {
      const cells = [0, 1, 2, 3]
        .map((col) => gridData.grid[`${row}_${col}`])
        .filter(Boolean);
      if (cells.length === 0) return -1;
      return cells.reduce((s, c) => s + c.success_rate, 0) / cells.length;
    });
  }, [gridData]);

  const colAvgs = useMemo(() => {
    if (!gridData?.grid || Object.keys(gridData.grid).length === 0) return [-1, -1, -1, -1];
    return [0, 1, 2, 3].map((col) => {
      const cells = [0, 1, 2, 3]
        .map((row) => gridData.grid[`${row}_${col}`])
        .filter(Boolean);
      if (cells.length === 0) return -1;
      return cells.reduce((s, c) => s + c.success_rate, 0) / cells.length;
    });
  }, [gridData]);

  // Overall ablated average (-1 = no data)
  const overallAvg = useMemo(() => {
    if (!gridData?.grid) return -1;
    const cells = Object.values(gridData.grid);
    if (cells.length === 0) return -1;
    return cells.reduce((s, c) => s + c.success_rate, 0) / cells.length;
  }, [gridData]);

  // Selected cell data
  const selectedCellData = selectedCell
    ? gridData?.grid[selectedCell] || gridData?.grid[`grid_${selectedCell}`]
    : null;

  // Injection condition summary
  const injectionSummary = useMemo(() => {
    if (!injectionData?.pairs) return null;
    const conditions = new Set<string>();
    injectionData.pairs.forEach((p) => {
      Object.keys(p.conditions).forEach((c) => conditions.add(c));
    });

    const results: Record<string, { sr: number; avgCos: number; count: number }> = {};
    conditions.forEach((cond) => {
      let successes = 0;
      let cosSum = 0;
      let cosCount = 0;
      let total = 0;
      injectionData.pairs.forEach((p) => {
        const c = p.conditions[cond];
        if (c) {
          total++;
          if (c.success) successes++;
          if (c.cos_to_baseline !== undefined) {
            cosSum += c.cos_to_baseline;
            cosCount++;
          }
        }
      });
      results[cond] = {
        sr: total > 0 ? successes / total : 0,
        avgCos: cosCount > 0 ? cosSum / cosCount : 0,
        count: total,
      };
    });
    return results;
  }, [injectionData]);

  return (
    <Box
      sx={{
        height: "100%",
        overflow: "auto",
        bgcolor: "#0f172a",
        p: 3,
      }}
    >
      {/* ─── Header ─────────────────────────────────────── */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <Typography
          variant="h6"
          sx={{ color: "#e2e8f0", fontWeight: 700, letterSpacing: "-0.02em" }}
        >
          ACT-ALOHA Spatial Sensitivity
        </Typography>
        <Chip
          label="CVAE Decoder"
          size="small"
          sx={{
            height: 20,
            fontSize: "10px",
            bgcolor: "#10b981",
            color: "white",
            fontWeight: 600,
          }}
        />
        <Chip
          label="No SAE"
          size="small"
          sx={{
            height: 20,
            fontSize: "10px",
            bgcolor: "#475569",
            color: "#94a3b8",
          }}
        />
      </Box>

      <Typography
        variant="body2"
        sx={{ color: "#94a3b8", mb: 3, maxWidth: 700 }}
      >
        ACT-ALOHA uses a CVAE decoder with action chunking -- no sparse
        autoencoder layers to inspect. Instead, we probe spatial sensitivity via
        4x4 grid masking and test activation injection across tasks.
      </Typography>

      {/* ─── Task Pills ─────────────────────────────────── */}
      <Box sx={{ display: "flex", gap: 1, mb: 3 }}>
        {TASKS.map((t) => (
          <Box
            key={t.id}
            onClick={() => {
              setSelectedTask(t.id);
              setSelectedCell(null);
            }}
            sx={{
              px: 2,
              py: 0.75,
              borderRadius: 2,
              cursor: "pointer",
              fontSize: "12px",
              fontWeight: 600,
              color: selectedTask === t.id ? "white" : "#94a3b8",
              bgcolor:
                selectedTask === t.id ? "#ef4444" : "rgba(255,255,255,0.05)",
              border:
                selectedTask === t.id
                  ? "1px solid #ef4444"
                  : "1px solid #334155",
              transition: "all 0.15s",
              "&:hover": {
                bgcolor:
                  selectedTask === t.id
                    ? "#dc2626"
                    : "rgba(255,255,255,0.1)",
              },
            }}
          >
            {t.label}
            <Typography
              component="span"
              sx={{
                ml: 1,
                fontSize: "10px",
                opacity: 0.7,
              }}
            >
              (baseline {(t.baseline * 100).toFixed(0)}%)
            </Typography>
          </Box>
        ))}
      </Box>

      {loading ? (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: 300,
          }}
        >
          <CircularProgress sx={{ color: "#ef4444" }} />
        </Box>
      ) : (
        <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
          {/* ─── Left: Grid Heatmap ────────────────────────── */}
          <Box sx={{ flex: "0 0 auto" }}>
            {/* Column averages header */}
            <Box sx={{ display: "flex", pl: "44px", mb: 0.5 }}>
              {colAvgs.map((avg, i) => (
                <Box
                  key={`col-${i}`}
                  sx={{
                    width: 80,
                    mx: "2px",
                    textAlign: "center",
                    fontSize: "10px",
                    fontWeight: 600,
                    color: avg >= 0 ? deltaColor((avg - baselineSR) * 100) : "#475569",
                  }}
                >
                  {avg >= 0 ? `${(avg * 100).toFixed(0)}%` : "--"}
                </Box>
              ))}
            </Box>

            <Box sx={{ display: "flex" }}>
              {/* Grid cells */}
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "44px repeat(4, 80px)",
                  gridTemplateRows: "repeat(4, 80px)",
                  gap: "4px",
                  border: "2px solid #1e293b",
                  borderRadius: 2,
                  p: 1,
                  bgcolor: "#0a0f1e",
                }}
              >
                {[0, 1, 2, 3].map((row) => (
                  <React.Fragment key={`row-${row}`}>
                    {/* Row label + avg */}
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Typography
                        sx={{
                          fontSize: "10px",
                          color: "#64748b",
                          fontWeight: 600,
                        }}
                      >
                        R{row}
                      </Typography>
                      <Typography
                        sx={{
                          fontSize: "10px",
                          color:
                            rowAvgs[row] >= 0
                              ? deltaColor(
                                  (rowAvgs[row] - baselineSR) * 100
                                )
                              : "#475569",
                          fontWeight: 600,
                        }}
                      >
                        {rowAvgs[row] >= 0
                          ? `${(rowAvgs[row] * 100).toFixed(0)}%`
                          : "--"}
                      </Typography>
                    </Box>

                    {/* 4 cells */}
                    {[0, 1, 2, 3].map((col) => {
                      const key = `${row}_${col}`;
                      const cell =
                        gridData?.grid[key] ||
                        gridData?.grid[`grid_${key}`];
                      const sr = cell?.success_rate ?? -1;
                      const isSelected = selectedCell === key;
                      const isCritical = gridData?.critical_cell === key;
                      const isBest = gridData?.best_cell === key;
                      const delta =
                        sr >= 0 ? (sr - baselineSR) * 100 : 0;

                      return (
                        <Tooltip
                          key={key}
                          title={
                            sr >= 0
                              ? `Cell (${row},${col}): ${(sr * 100).toFixed(0)}% success | delta ${delta >= 0 ? "+" : ""}${delta.toFixed(0)}pp`
                              : "No data"
                          }
                          arrow
                          placement="top"
                        >
                          <Box
                            onClick={() => setSelectedCell(key)}
                            sx={{
                              width: 80,
                              height: 80,
                              display: "flex",
                              flexDirection: "column",
                              alignItems: "center",
                              justifyContent: "center",
                              cursor: "pointer",
                              bgcolor:
                                sr >= 0
                                  ? srColor(sr, baselineSR) + "25"
                                  : "#1e293b",
                              border: isSelected
                                ? "2px solid #ef4444"
                                : isCritical
                                ? "2px solid #dc2626"
                                : isBest
                                ? "2px solid #22c55e"
                                : "1px solid #1e293b",
                              borderRadius: 1.5,
                              transition: "all 0.15s",
                              position: "relative",
                              "&:hover": {
                                bgcolor:
                                  sr >= 0
                                    ? srColor(sr, baselineSR) + "40"
                                    : "#334155",
                                transform: "scale(1.04)",
                              },
                            }}
                          >
                            <Typography
                              sx={{
                                color:
                                  sr >= 0
                                    ? srColor(sr, baselineSR)
                                    : "#475569",
                                fontWeight: 800,
                                fontSize: "20px",
                                lineHeight: 1.1,
                              }}
                            >
                              {sr >= 0
                                ? `${(sr * 100).toFixed(0)}%`
                                : "--"}
                            </Typography>
                            <Typography
                              sx={{
                                fontSize: "10px",
                                color: deltaColor(delta),
                                fontWeight: 600,
                                mt: 0.25,
                              }}
                            >
                              {sr >= 0
                                ? `${delta >= 0 ? "+" : ""}${delta.toFixed(0)}pp`
                                : ""}
                            </Typography>
                            {isCritical && (
                              <Box
                                sx={{
                                  position: "absolute",
                                  top: 2,
                                  right: 3,
                                  fontSize: "8px",
                                  color: "#dc2626",
                                  fontWeight: 700,
                                }}
                              >
                                WORST
                              </Box>
                            )}
                            {isBest && (
                              <Box
                                sx={{
                                  position: "absolute",
                                  top: 2,
                                  right: 3,
                                  fontSize: "8px",
                                  color: "#22c55e",
                                  fontWeight: 700,
                                }}
                              >
                                BEST
                              </Box>
                            )}
                          </Box>
                        </Tooltip>
                      );
                    })}
                  </React.Fragment>
                ))}
              </Box>
            </Box>

            {/* Baseline chip */}
            <Box sx={{ display: "flex", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
              {gridData?.baseline && (
                <Chip
                  label={`Baseline: ${(gridData.baseline.success_rate * 100).toFixed(0)}% (${gridData.baseline.n_episodes} eps)`}
                  size="small"
                  sx={{
                    bgcolor: "#22c55e20",
                    color: "#22c55e",
                    fontWeight: 600,
                    fontSize: "11px",
                  }}
                />
              )}
              <Chip
                label={overallAvg >= 0 ? `Avg ablated: ${(overallAvg * 100).toFixed(0)}%` : "Avg ablated: --"}
                size="small"
                sx={{
                  bgcolor: overallAvg >= 0 ? deltaColor((overallAvg - baselineSR) * 100) + "20" : "#47556920",
                  color: overallAvg >= 0 ? deltaColor((overallAvg - baselineSR) * 100) : "#475569",
                  fontWeight: 600,
                  fontSize: "11px",
                }}
              />
            </Box>

            {/* Noise comparison */}
            {gridData?.noise &&
              Object.keys(gridData.noise).length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "#64748b",
                      fontWeight: 600,
                      display: "block",
                      mb: 0.5,
                    }}
                  >
                    Gaussian Noise Controls
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1 }}>
                    {Object.entries(gridData.noise).map(([k, v]) => (
                      <Chip
                        key={k}
                        label={`${k}: ${(v.success_rate * 100).toFixed(0)}%`}
                        size="small"
                        sx={{
                          bgcolor:
                            srColor(v.success_rate, baselineSR) + "20",
                          color: srColor(v.success_rate, baselineSR),
                          fontSize: "10px",
                          fontWeight: 600,
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              )}
          </Box>

          {/* ─── Center: Cell detail + stats ────────────────── */}
          <Box sx={{ flex: "1 1 280px", minWidth: 260 }}>
            {/* Cell detail panel */}
            <Box
              sx={{
                bgcolor: "#1e293b",
                borderRadius: 2,
                p: 2.5,
                mb: 2,
                minHeight: 180,
                border: "1px solid #334155",
              }}
            >
              {selectedCellData ? (
                <>
                  <Typography
                    variant="subtitle1"
                    sx={{ color: "#e2e8f0", fontWeight: 700, mb: 1.5 }}
                  >
                    Cell ({selectedCellData.row}, {selectedCellData.col})
                  </Typography>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: 1.5,
                    }}
                  >
                    <StatBox
                      label="Success Rate"
                      value={`${(selectedCellData.success_rate * 100).toFixed(0)}%`}
                      color={srColor(
                        selectedCellData.success_rate,
                        baselineSR
                      )}
                    />
                    <StatBox
                      label="Delta"
                      value={`${((selectedCellData.success_rate - baselineSR) * 100) >= 0 ? "+" : ""}${((selectedCellData.success_rate - baselineSR) * 100).toFixed(0)}pp`}
                      color={deltaColor(
                        (selectedCellData.success_rate - baselineSR) * 100
                      )}
                    />
                    <StatBox
                      label="Mean Reward"
                      value={
                        selectedCellData.mean_reward?.toFixed(1) ?? "--"
                      }
                      color="#e2e8f0"
                    />
                    <StatBox
                      label="Episodes"
                      value={String(selectedCellData.n_episodes || "--")}
                      color="#e2e8f0"
                    />
                  </Box>
                  {selectedCellData.bbox &&
                    selectedCellData.bbox.length === 4 && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: "#475569",
                          display: "block",
                          mt: 1.5,
                          fontSize: "10px",
                        }}
                      >
                        Image region: [
                        {selectedCellData.bbox
                          .map((v) => v.toFixed(2))
                          .join(", ")}
                        ]
                      </Typography>
                    )}
                  {gridData?.critical_cell === selectedCell && (
                    <Chip
                      label="CRITICAL -- Primary manipulation workspace"
                      size="small"
                      sx={{
                        mt: 1.5,
                        bgcolor: "#dc262620",
                        color: "#ef4444",
                        fontWeight: 600,
                        fontSize: "10px",
                      }}
                    />
                  )}
                </>
              ) : (
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    minHeight: 140,
                    color: "#475569",
                  }}
                >
                  <svg
                    width="32"
                    height="32"
                    viewBox="0 0 32 32"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  >
                    <rect x="4" y="4" width="10" height="10" rx="2" />
                    <rect x="18" y="4" width="10" height="10" rx="2" />
                    <rect x="4" y="18" width="10" height="10" rx="2" />
                    <rect x="18" y="18" width="10" height="10" rx="2" />
                  </svg>
                  <Typography
                    variant="caption"
                    sx={{ mt: 1, textAlign: "center" }}
                  >
                    Click a grid cell to see details
                  </Typography>
                </Box>
              )}
            </Box>

            {/* Key finding */}
            <Box
              sx={{
                bgcolor: "#1e293b",
                borderRadius: 2,
                p: 2,
                border: "1px solid #334155",
              }}
            >
              <Typography
                variant="subtitle2"
                sx={{ color: "#ef4444", fontWeight: 700, mb: 1 }}
              >
                Key Finding: Spatially Structured Vision Dependence
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: "#94a3b8", fontSize: "12px", lineHeight: 1.6 }}
              >
                Grid ablation reveals that ACT-ALOHA has strong spatial
                structure in its vision dependence. The lower-right quadrant
                (workspace area) is most critical for task success. Masking
                these cells causes catastrophic failure, while masking
                peripheral regions has minimal effect. This contrasts with
                VLA models that distribute visual information more uniformly.
              </Typography>
            </Box>
          </Box>

          {/* ─── Right: Injection Results ───────────────────── */}
          <Box sx={{ flex: "1 1 280px", minWidth: 260 }}>
            <Box
              sx={{
                bgcolor: "#1e293b",
                borderRadius: 2,
                p: 2.5,
                border: "1px solid #334155",
                mb: 2,
              }}
            >
              <Typography
                variant="subtitle1"
                sx={{ color: "#e2e8f0", fontWeight: 700, mb: 0.5 }}
              >
                Activation Injection
              </Typography>
              <Typography
                variant="caption"
                sx={{ color: "#475569", display: "block", mb: 2 }}
              >
                Cross-task injection: Transfer Cube activations injected into
                Insertion environment
              </Typography>

              {injectionLoading ? (
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "center",
                    py: 3,
                  }}
                >
                  <CircularProgress size={20} sx={{ color: "#ef4444" }} />
                </Box>
              ) : injectionSummary ? (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {Object.entries(injectionSummary)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([cond, stats]) => {
                      const label = cond
                        .replace("inject_", "")
                        .replace(/_/g, " ")
                        .replace("L0", "Layer 0");
                      return (
                        <Box
                          key={cond}
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 1,
                            p: 1,
                            bgcolor: "#0f172a",
                            borderRadius: 1,
                            border: "1px solid #1e293b",
                          }}
                        >
                          <Typography
                            sx={{
                              flex: 1,
                              color: "#e2e8f0",
                              fontSize: "11px",
                              fontWeight: 500,
                              textTransform: "capitalize",
                            }}
                          >
                            {label}
                          </Typography>
                          <Chip
                            label={`${(stats.sr * 100).toFixed(0)}%`}
                            size="small"
                            sx={{
                              height: 20,
                              fontSize: "10px",
                              fontWeight: 700,
                              bgcolor: srColor(stats.sr, baselineSR) + "25",
                              color: srColor(stats.sr, baselineSR),
                            }}
                          />
                          {stats.avgCos > 0 && (
                            <Tooltip
                              title={`Cosine similarity to baseline: ${stats.avgCos.toFixed(3)}`}
                              arrow
                            >
                              <Chip
                                label={`cos=${stats.avgCos.toFixed(2)}`}
                                size="small"
                                sx={{
                                  height: 18,
                                  fontSize: "9px",
                                  bgcolor: stats.avgCos > 0.99
                                    ? "#22c55e20"
                                    : "#f9731620",
                                  color: stats.avgCos > 0.99
                                    ? "#22c55e"
                                    : "#f97316",
                                }}
                              />
                            </Tooltip>
                          )}
                        </Box>
                      );
                    })}
                </Box>
              ) : (
                <Typography
                  variant="body2"
                  sx={{ color: "#475569", textAlign: "center", py: 2 }}
                >
                  No injection data available
                </Typography>
              )}
            </Box>

            {/* Injection key finding */}
            <Box
              sx={{
                bgcolor: "#1e293b",
                borderRadius: 2,
                p: 2,
                border: "1px solid #334155",
              }}
            >
              <Typography
                variant="subtitle2"
                sx={{ color: "#06b6d4", fontWeight: 700, mb: 1 }}
              >
                Finding: Residual Connections Wash Out Injections
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: "#94a3b8", fontSize: "12px", lineHeight: 1.6 }}
              >
                Cross-task activation injection has{" "}
                <strong style={{ color: "#22c55e" }}>ZERO effect</strong> on
                ACT-ALOHA behavior (cos_to_baseline = 1.0). The residual
                connections in the CVAE decoder completely negate injected
                activations, making internal steering impossible. This is a
                fundamental architectural difference from VLA models like
                Pi0.5 and OpenVLA-OFT where injection meaningfully alters
                behavior.
              </Typography>
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function StatBox({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <Box sx={{ bgcolor: "#0f172a", p: 1.5, borderRadius: 1.5 }}>
      <Typography variant="caption" sx={{ color: "#64748b", display: "block" }}>
        {label}
      </Typography>
      <Typography
        sx={{ color, fontWeight: 700, fontSize: "22px", lineHeight: 1.2 }}
      >
        {value}
      </Typography>
    </Box>
  );
}
