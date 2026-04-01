"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Alert, Box, Typography, Select, MenuItem, FormControl, InputLabel, Tooltip, Chip } from "@mui/material";
import { useAppSelector } from "@/redux/hooks";
import { DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
import { API_BASE_URL } from "@/config/api";

interface GridCell {
  row: number;
  col: number;
  success_rate: number;
  mean_reward: number;
  std_reward?: number;
  n_episodes: number;
  has_videos?: boolean;
  bbox?: number[];
  label?: string;
}

interface GridData {
  model: string;
  task?: string;
  suite?: string;
  grid_size: number;
  grid: Record<string, GridCell>;
  baseline?: { success_rate: number; mean_reward: number; n_episodes: number };
  noise?: Record<string, { success_rate: number; mean_reward: number }>;
  critical_cell?: string;
  video_base_path?: string;
  available_tasks?: string[];
  available_suites?: string[];
  grid_type?: string;
  empty?: boolean;
  message?: string;
}

function srColor(sr: number, baseline: number): string {
  if (sr <= 0.15) return "#dc2626"; // red
  if (sr < baseline * 0.5) return "#f97316"; // orange
  if (sr < baseline * 0.8) return "#eab308"; // yellow
  if (sr >= baseline) return "#22c55e"; // green
  return "#84cc16"; // lime
}

export default function GridAblationDemo() {
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);
  // Use global model as the local model
  const effectiveModel = currentModel;
  const datasetSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
  const defaultSuite = datasetSuites.length > 0 ? datasetSuites[0] : 'libero_goal';
  const [model, setModel] = useState<string>(effectiveModel);
  const [task, setTask] = useState<string>("AlohaInsertion-v0");
  const [suite, setSuite] = useState<string>(defaultSuite);
  const [data, setData] = useState<GridData | null>(null);
  const [selectedCell, setSelectedCell] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Sync local model with global model changes
  useEffect(() => { setModel(effectiveModel); }, [effectiveModel]);

  // Reset suite when dataset changes — force refetch with first suite of new dataset
  useEffect(() => {
    const newSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
    if (newSuites.length > 0) {
      setSuite(newSuites[0]);
    }
  }, [currentDataset]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      let params: string;
      if (model === "act") {
        params = `model=act&task=${task}`;
      } else {
        params = `model=${model}&suite=${suite}`;
      }
      const res = await fetch(`${API_BASE_URL}/api/vla/grid_ablation?${params}`);
      if (!res.ok && res.status === 404) { setData(null); setLoading(false); return; }
      const json = await res.json();
      if (json.status === 404) setData(null);
      else setData(json);
    } catch { setData(null); }
    setLoading(false);
  }, [model, task, suite]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const baselineSR = typeof data?.baseline === 'number'
    ? data.baseline
    : (data?.baseline?.success_rate ?? 1.0);

  return (
    <Box>
      <Typography variant="body2" sx={{ color: "#94a3b8", mb: 2 }}>
        Interactive 4x4 spatial grid ablation. Each cell shows the task success rate when that image region is masked.
        Click a cell to see details. <strong>Red = catastrophic failure, Green = resilient.</strong>
      </Typography>

      {/* Controls */}
      <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
        <Chip label={{act: "ACT-ALOHA", pi05: "Pi0.5", openvla: "OpenVLA-OFT", xvla: "X-VLA", smolvla: "SmolVLA", groot: "GR00T"}[model] || model.toUpperCase()}
          sx={{ bgcolor: "#1e293b", color: "#e2e8f0", fontWeight: 600 }} />

        {model === "act" ? (
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel sx={{ color: "#64748b" }}>Task</InputLabel>
            <Select value={task} label="Task" onChange={(e) => setTask(e.target.value)}
              sx={{ color: "#e2e8f0", ".MuiOutlinedInput-notchedOutline": { borderColor: "#334155" } }}>
              <MenuItem value="AlohaInsertion-v0">Insertion (baseline 80%)</MenuItem>
              <MenuItem value="AlohaTransferCube-v0">Transfer Cube (baseline 100%)</MenuItem>
            </Select>
          </FormControl>
        ) : (
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel sx={{ color: "#64748b" }}>Suite</InputLabel>
            <Select value={suite} label="Suite" onChange={(e) => setSuite(e.target.value)}
              sx={{ color: "#e2e8f0", ".MuiOutlinedInput-notchedOutline": { borderColor: "#334155" } }}>
              {(datasetSuites.length > 0 ? datasetSuites : ["libero_goal", "libero_10", "libero_object", "libero_spatial"]).map(s => (
                <MenuItem key={s} value={s}>{s.replace(/_/g, ' ')}</MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Box>

      {loading && <Typography sx={{ color: "#64748b" }}>Loading...</Typography>}

      {data && data.empty && (
        <Box sx={{ py: 3, textAlign: "center" }}>
          <Typography variant="body2" sx={{ color: "#475569" }}>
            {data.message || `No grid ablation data for ${model} / ${suite}.`}
          </Typography>
          {data.available_suites && data.available_suites.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 1 }}>
                Available suites:
              </Typography>
              <Box sx={{ display: "flex", gap: 0.5, justifyContent: "center", flexWrap: "wrap" }}>
                {data.available_suites.map((s: string) => (
                  <Chip key={s} label={s.replace(/_/g, ' ')} size="small"
                    onClick={() => setSuite(s)}
                    sx={{ cursor: "pointer", bgcolor: "#1e293b", color: "#94a3b8", fontSize: "11px",
                      "&:hover": { bgcolor: "#334155", color: "#e2e8f0" } }} />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      )}

      {data && !data.empty && Object.keys(data.grid || {}).length > 0 && (
        <Box sx={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {/* Grid */}
          <Box>
            {/* Grid type label for layer ablation */}
            {data.grid_type === 'layer_ablation' && (
              <Typography variant="caption" sx={{ color: "#f59e0b", fontWeight: 600, display: "block", mb: 1, fontSize: "11px" }}>
                Layer Ablation Grid (each cell = one layer zeroed)
              </Typography>
            )}

            {/* Baseline chip */}
            {data.baseline && (
              <Chip label={`Baseline: ${(baselineSR * 100).toFixed(0)}%`}
                sx={{ mb: 1, bgcolor: "#22c55e20", color: "#22c55e", fontWeight: 600, fontSize: "12px" }} />
            )}

            {/* Note when all cells show 0% (e.g., X-VLA where zeroing any layer destroys performance) */}
            {baselineSR > 0 && (() => {
              const cells = Object.values(data.grid);
              const allZero = cells.length > 0 && cells.every((c: any) => c.success_rate === 0);
              if (allZero) {
                return (
                  <Alert
                    severity="warning"
                    sx={{
                      mb: 2,
                      bgcolor: 'rgba(245, 158, 11, 0.1)',
                      color: '#fbbf24',
                      border: '1px solid rgba(245, 158, 11, 0.3)',
                      '& .MuiAlert-icon': { color: '#f59e0b' },
                    }}
                  >
                    <strong>All {data.grid_type === 'layer_ablation' ? 'layers' : 'regions'} critical:</strong>{' '}
                    Zeroing any single {data.grid_type === 'layer_ablation' ? 'layer' : 'region'} causes complete
                    task failure (0% success vs {(baselineSR * 100).toFixed(0)}% baseline).
                    This indicates a narrow, non-redundant architecture where every component is essential.
                  </Alert>
                );
              }
              return null;
            })()}

            {(() => {
              // For layer ablation with more than 16 layers, expand grid columns
              const gridKeys = Object.keys(data.grid).sort((a, b) => {
                const [ar, ac] = a.split('_').map(Number);
                const [br, bc] = b.split('_').map(Number);
                return ar * 100 + ac - (br * 100 + bc);
              });
              const numCells = gridKeys.length;
              const isLayerAblation = data.grid_type === 'layer_ablation';
              // Use a wider grid for layer ablation to show labels
              const gridCols = isLayerAblation && numCells > 16 ? 6 : 4;
              const gridRows = Math.ceil(numCells / gridCols);
              const cellSize = isLayerAblation ? 70 : 80;

              return (
                <Box sx={{
                  display: "grid",
                  gridTemplateColumns: `repeat(${gridCols}, ${cellSize}px)`,
                  gap: "4px",
                  border: "2px solid #334155", borderRadius: 1, p: 1, bgcolor: "#0f172a"
                }}>
                  {Array.from({ length: gridRows }, (_, row) =>
                    Array.from({ length: gridCols }, (_, col) => {
                      const idx = row * gridCols + col;
                      if (idx >= numCells && !isLayerAblation) {
                        // For spatial grid, always render 4x4
                        const key = `${row}_${col}`;
                        const cell = data.grid[key] || data.grid[`grid_${key}`];
                        const sr = cell?.success_rate ?? -1;
                        const isSelected = selectedCell === key;
                        const isCritical = data.critical_cell === key;
                        return (
                          <Tooltip key={key} title={sr >= 0 ? `Cell (${row},${col}): ${(sr * 100).toFixed(0)}% success` : "No data"}>
                            <Box
                              onClick={() => setSelectedCell(key)}
                              sx={{
                                width: cellSize, height: cellSize, display: "flex", flexDirection: "column",
                                alignItems: "center", justifyContent: "center", cursor: "pointer",
                                bgcolor: sr >= 0 ? srColor(sr, baselineSR) + "30" : "#1e293b",
                                border: isSelected ? "2px solid #ef4444" : isCritical ? "2px solid #dc2626" : "1px solid #334155",
                                borderRadius: 1, transition: "all 0.15s",
                                "&:hover": { bgcolor: sr >= 0 ? srColor(sr, baselineSR) + "50" : "#334155" }
                              }}
                            >
                              <Typography sx={{ color: sr >= 0 ? srColor(sr, baselineSR) : "#475569", fontWeight: 700, fontSize: "18px" }}>
                                {sr >= 0 ? `${(sr * 100).toFixed(0)}%` : "\u2014"}
                              </Typography>
                              <Typography sx={{ color: "#64748b", fontSize: "9px" }}>
                                ({row},{col})
                              </Typography>
                            </Box>
                          </Tooltip>
                        );
                      }
                      if (idx >= numCells) return null;

                      const key = isLayerAblation ? gridKeys[idx] : `${row}_${col}`;
                      const cell = data.grid[key] || data.grid[`grid_${key}`];
                      const sr = cell?.success_rate ?? -1;
                      const isSelected = selectedCell === key;
                      const isCritical = data.critical_cell === key;
                      const cellLabel = cell?.label || (isLayerAblation ? `L${idx}` : `(${row},${col})`);

                      return (
                        <Tooltip key={key} title={sr >= 0 ? `${cellLabel}: ${(sr * 100).toFixed(0)}% success (baseline: ${(baselineSR * 100).toFixed(0)}%)` : "No data"}>
                          <Box
                            onClick={() => setSelectedCell(key)}
                            sx={{
                              width: cellSize, height: cellSize, display: "flex", flexDirection: "column",
                              alignItems: "center", justifyContent: "center", cursor: "pointer",
                              bgcolor: sr >= 0 ? srColor(sr, baselineSR) + "30" : "#1e293b",
                              border: isSelected ? "2px solid #ef4444" : isCritical ? "2px solid #dc2626" : "1px solid #334155",
                              borderRadius: 1, transition: "all 0.15s",
                              "&:hover": { bgcolor: sr >= 0 ? srColor(sr, baselineSR) + "50" : "#334155" }
                            }}
                          >
                            <Typography sx={{ color: sr >= 0 ? srColor(sr, baselineSR) : "#475569", fontWeight: 700, fontSize: isLayerAblation ? "14px" : "18px" }}>
                              {sr >= 0 ? `${(sr * 100).toFixed(0)}%` : "\u2014"}
                            </Typography>
                            <Typography sx={{ color: "#64748b", fontSize: "8px", textAlign: "center", lineHeight: 1.1 }}>
                              {cellLabel}
                            </Typography>
                          </Box>
                        </Tooltip>
                      );
                    })
                  )}
                </Box>
              );
            })()}

            {/* Noise results */}
            {data.noise && Object.keys(data.noise).length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" sx={{ color: "#64748b", fontWeight: 600, display: "block", mb: 0.5 }}>
                  Gaussian Noise
                </Typography>
                <Box sx={{ display: "flex", gap: 1 }}>
                  {Object.entries(data.noise).map(([k, v]) => (
                    <Chip key={k} label={`${k}: ${(v.success_rate * 100).toFixed(0)}%`} size="small"
                      sx={{ bgcolor: srColor(v.success_rate, baselineSR) + "30", color: srColor(v.success_rate, baselineSR), fontSize: "11px" }} />
                  ))}
                </Box>
              </Box>
            )}
          </Box>

          {/* Detail panel */}
          <Box sx={{ flex: 1, bgcolor: "#1e293b", borderRadius: 1, p: 2, minHeight: 200 }}>
            {selectedCell ? (() => {
              const cell = data.grid[selectedCell] || data.grid[`grid_${selectedCell}`];
              if (!cell) return <Typography sx={{ color: "#475569" }}>No data for this cell</Typography>;
              const delta = data.baseline ? (cell.success_rate - baselineSR) * 100 : 0;
              const isLayerAblation = data.grid_type === 'layer_ablation';
              const cellTitle = cell.label || (isLayerAblation ? `Layer ${selectedCell}` : `Cell (${cell.row}, ${cell.col})`);
              return (
                <>
                  <Typography variant="h6" sx={{ color: "#e2e8f0", mb: 1 }}>
                    {cellTitle}
                  </Typography>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, mb: 2 }}>
                    <Box sx={{ bgcolor: "#0f172a", p: 1.5, borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ color: "#64748b" }}>Success Rate</Typography>
                      <Typography sx={{ color: srColor(cell.success_rate, baselineSR), fontWeight: 700, fontSize: "24px" }}>
                        {(cell.success_rate * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: "#0f172a", p: 1.5, borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ color: "#64748b" }}>Delta from Baseline</Typography>
                      <Typography sx={{ color: delta >= 0 ? "#22c55e" : "#ef4444", fontWeight: 700, fontSize: "24px" }}>
                        {delta >= 0 ? "+" : ""}{delta.toFixed(0)}pp
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: "#0f172a", p: 1.5, borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ color: "#64748b" }}>Mean Reward</Typography>
                      <Typography sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "18px" }}>
                        {cell.mean_reward?.toFixed(1) ?? "—"}
                      </Typography>
                    </Box>
                    <Box sx={{ bgcolor: "#0f172a", p: 1.5, borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ color: "#64748b" }}>Episodes</Typography>
                      <Typography sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "18px" }}>
                        {cell.n_episodes || "—"}
                      </Typography>
                    </Box>
                  </Box>
                  {cell.bbox && cell.bbox.length === 4 && (
                    <Typography variant="caption" sx={{ color: "#64748b" }}>
                      Image region: [{cell.bbox.map(v => v.toFixed(2)).join(", ")}]
                    </Typography>
                  )}
                  {data.critical_cell === selectedCell && (
                    <Chip label={data.grid_type === 'layer_ablation' ? "MOST CRITICAL LAYER" : "CRITICAL CELL \u2014 Primary manipulation workspace"} size="small"
                      sx={{ mt: 1, bgcolor: "#dc262630", color: "#ef4444", fontWeight: 600 }} />
                  )}
                </>
              );
            })() : (
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#475569" }}>
                <Typography>Click a grid cell to see details</Typography>
              </Box>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}
