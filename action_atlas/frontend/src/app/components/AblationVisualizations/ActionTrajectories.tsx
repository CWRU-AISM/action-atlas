"use client";
import React, { useState, useEffect, useMemo } from "react";
import {
  Box,
  Typography,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { API_BASE_URL } from "@/config/api";

// Dimension labels and colors
const DIM_LABELS = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"];
const DIM_COLORS = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899",
];

interface FileInfo {
  filename: string;
  layer: string;
  suite: string;
  label: string;
}

interface ConceptInfo {
  concept: string;
  n_tasks: number;
  tasks: string[];
}

interface TrajectoryData {
  layer: string;
  suite: string;
  concept: string;
  task_id: string;
  episode: number;
  n_episodes: number;
  success_rate: number;
  baseline_success_rate?: number;
  delta: number;
  baseline_trajectory: number[][];
  ablated_trajectory: number[][];
  dim_labels: string[];
  baseline_steps: number;
  ablated_steps: number;
  baseline_available: boolean;
}

export default function ActionTrajectories() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [concepts, setConcepts] = useState<ConceptInfo[]>([]);
  const [selectedConcept, setSelectedConcept] = useState<string>("");
  const [selectedTask, setSelectedTask] = useState<string>("0");
  const [selectedEpisode, setSelectedEpisode] = useState<number>(0);
  const [trajectoryData, setTrajectoryData] = useState<TrajectoryData | null>(null);
  const [selectedDims, setSelectedDims] = useState<number[]>([0, 1, 2]); // dx, dy, dz by default
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available files
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/vla/action_trajectories/files`);
        if (!res.ok) throw new Error("Failed to load files");
        const data = await res.json();
        setFiles(data.files || []);
        if (data.files?.length > 0) {
          setSelectedFile(`${data.files[0].layer}_${data.files[0].suite}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load files");
      }
    };
    fetchFiles();
  }, []);

  // Parse selected file into layer/suite
  const { layer, suite } = useMemo(() => {
    if (!selectedFile) return { layer: "", suite: "" };
    const parts = selectedFile.split("_");
    return { layer: parts[0], suite: parts.slice(1).join("_") };
  }, [selectedFile]);

  // Load concepts when file changes
  useEffect(() => {
    if (!layer || !suite) return;
    const fetchConcepts = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/vla/action_trajectories?layer=${layer}&suite=${suite}`
        );
        if (!res.ok) throw new Error("Failed to load concepts");
        const data = await res.json();
        setConcepts(data.concepts || []);
        if (data.concepts?.length > 0) {
          setSelectedConcept(data.concepts[0].concept);
          if (data.concepts[0].tasks?.length > 0) {
            setSelectedTask(data.concepts[0].tasks[0]);
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load concepts");
      } finally {
        setLoading(false);
      }
    };
    fetchConcepts();
  }, [layer, suite]);

  // Load trajectory data
  useEffect(() => {
    if (!layer || !suite || !selectedConcept || !selectedTask) return;
    const fetchTrajectory = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/vla/action_trajectories?layer=${layer}&suite=${suite}&concept=${encodeURIComponent(selectedConcept)}&task_id=${selectedTask}&episode=${selectedEpisode}`
        );
        if (!res.ok) throw new Error("Failed to load trajectory");
        const data = await res.json();
        setTrajectoryData(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load trajectory");
      } finally {
        setLoading(false);
      }
    };
    fetchTrajectory();
  }, [layer, suite, selectedConcept, selectedTask, selectedEpisode]);

  // Get available tasks for selected concept
  const availableTasks = useMemo(() => {
    const concept = concepts.find((c) => c.concept === selectedConcept);
    return concept?.tasks || [];
  }, [concepts, selectedConcept]);

  // Chart data
  const chartData = useMemo(() => {
    if (!trajectoryData) return [];
    const maxLen = Math.max(
      trajectoryData.baseline_trajectory.length,
      trajectoryData.ablated_trajectory.length
    );
    const data = [];
    for (let i = 0; i < maxLen; i++) {
      const point: Record<string, number | undefined> = { step: i };
      const bStep = trajectoryData.baseline_trajectory[i];
      const aStep = trajectoryData.ablated_trajectory[i];
      for (const dim of selectedDims) {
        point[`baseline_${DIM_LABELS[dim]}`] = bStep?.[dim];
        point[`ablated_${DIM_LABELS[dim]}`] = aStep?.[dim];
      }
      data.push(point);
    }
    return data;
  }, [trajectoryData, selectedDims]);

  const toggleDim = (dim: number) => {
    setSelectedDims((prev) => {
      if (prev.includes(dim)) {
        if (prev.length <= 1) return prev; // Keep at least one
        return prev.filter((d) => d !== dim);
      }
      return [...prev, dim];
    });
  };

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
        <Typography
          variant="subtitle2"
          sx={{ color: "#94a3b8", fontWeight: 600, fontSize: "12px" }}
        >
          Action Trajectories -- Baseline vs Ablated
        </Typography>
        <Chip
          label="OpenVLA-OFT"
          size="small"
          sx={{
            height: 18,
            fontSize: "9px",
            bgcolor: "#8b5cf6",
            color: "white",
          }}
        />
        <Chip
          label="7-DOF"
          size="small"
          sx={{
            height: 18,
            fontSize: "9px",
            bgcolor: "#1e293b",
            color: "#94a3b8",
          }}
        />
      </Box>

      {/* Controls */}
      <Box
        sx={{
          display: "flex",
          gap: 1.5,
          mb: 2,
          p: 1.5,
          bgcolor: "#1e293b",
          borderRadius: 1,
          border: "1px solid #334155",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <FormControl size="small" sx={{ minWidth: 180 }}>
          <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
            Layer / Suite
          </InputLabel>
          <Select
            value={selectedFile}
            label="Layer / Suite"
            onChange={(e) => setSelectedFile(e.target.value)}
            sx={{
              color: "#e2e8f0",
              fontSize: "11px",
              "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
              "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
            }}
          >
            {files.map((f) => (
              <MenuItem key={`${f.layer}_${f.suite}`} value={`${f.layer}_${f.suite}`}>
                {f.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
            Concept
          </InputLabel>
          <Select
            value={selectedConcept}
            label="Concept"
            onChange={(e) => {
              setSelectedConcept(e.target.value);
              setSelectedEpisode(0);
            }}
            sx={{
              color: "#e2e8f0",
              fontSize: "11px",
              "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
              "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
            }}
          >
            {concepts.map((c) => (
              <MenuItem key={c.concept} value={c.concept}>
                {c.concept}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 90 }}>
          <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
            Task
          </InputLabel>
          <Select
            value={selectedTask}
            label="Task"
            onChange={(e) => {
              setSelectedTask(e.target.value);
              setSelectedEpisode(0);
            }}
            sx={{
              color: "#e2e8f0",
              fontSize: "11px",
              "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
              "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
            }}
          >
            {availableTasks.map((t) => (
              <MenuItem key={t} value={t}>
                Task {t}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {trajectoryData && trajectoryData.n_episodes > 1 && (
          <FormControl size="small" sx={{ minWidth: 90 }}>
            <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
              Episode
            </InputLabel>
            <Select
              value={selectedEpisode}
              label="Episode"
              onChange={(e) => setSelectedEpisode(Number(e.target.value))}
              sx={{
                color: "#e2e8f0",
                fontSize: "11px",
                "& .MuiOutlinedInput-notchedOutline": { borderColor: "#334155" },
                "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#475569" },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#ef4444" },
              }}
            >
              {Array.from({ length: trajectoryData.n_episodes }, (_, i) => (
                <MenuItem key={i} value={i}>
                  Ep {i}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        {loading && (
          <CircularProgress size={16} sx={{ color: "#ef4444" }} />
        )}
      </Box>

      {/* Dimension toggles */}
      <Box sx={{ display: "flex", gap: 0.5, mb: 2, flexWrap: "wrap" }}>
        <Typography
          variant="caption"
          sx={{ color: "#64748b", fontSize: "9px", mr: 1, alignSelf: "center" }}
        >
          Dimensions:
        </Typography>
        {DIM_LABELS.map((label, idx) => (
          <Chip
            key={label}
            label={label}
            size="small"
            onClick={() => toggleDim(idx)}
            sx={{
              height: 20,
              fontSize: "9px",
              fontWeight: 600,
              bgcolor: selectedDims.includes(idx)
                ? DIM_COLORS[idx]
                : "#1e293b",
              color: selectedDims.includes(idx) ? "white" : "#64748b",
              border: `1px solid ${selectedDims.includes(idx) ? DIM_COLORS[idx] : "#334155"}`,
              cursor: "pointer",
              "&:hover": {
                bgcolor: selectedDims.includes(idx)
                  ? DIM_COLORS[idx]
                  : "#334155",
              },
            }}
          />
        ))}
      </Box>

      {error && (
        <Alert
          severity="error"
          sx={{
            mb: 2,
            bgcolor: "#7f1d1d",
            color: "#fca5a5",
            "& .MuiAlert-icon": { color: "#fca5a5" },
            fontSize: "11px",
          }}
        >
          {error}
        </Alert>
      )}

      {/* Stats bar */}
      {trajectoryData && (
        <Box
          sx={{
            display: "flex",
            gap: 2,
            mb: 2,
            p: 1,
            bgcolor: "#1e293b",
            borderRadius: 1,
            border: "1px solid #334155",
            flexWrap: "wrap",
          }}
        >
          <Box>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
              Success Rate
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color:
                  trajectoryData.success_rate > 0.5
                    ? "#22c55e"
                    : trajectoryData.success_rate > 0
                    ? "#f59e0b"
                    : "#ef4444",
                fontWeight: 600,
                fontSize: "12px",
                fontFamily: "monospace",
              }}
            >
              {(trajectoryData.success_rate * 100).toFixed(0)}%
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
              Delta
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: trajectoryData.delta < 0 ? "#ef4444" : "#22c55e",
                fontWeight: 600,
                fontSize: "12px",
                fontFamily: "monospace",
              }}
            >
              {trajectoryData.delta > 0 ? "+" : ""}
              {(trajectoryData.delta * 100).toFixed(1)}pp
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
              Concept
            </Typography>
            <Typography
              variant="body2"
              sx={{ color: "#e2e8f0", fontSize: "11px" }}
            >
              {trajectoryData.concept}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
              Baseline Steps
            </Typography>
            <Typography
              variant="body2"
              sx={{ color: "#94a3b8", fontSize: "11px", fontFamily: "monospace" }}
            >
              {trajectoryData.baseline_available ? trajectoryData.baseline_steps : "N/A"}
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
              Ablated Steps
            </Typography>
            <Typography
              variant="body2"
              sx={{ color: "#94a3b8", fontSize: "11px", fontFamily: "monospace" }}
            >
              {trajectoryData.ablated_steps}
            </Typography>
          </Box>
        </Box>
      )}

      {/* Baseline not available notice */}
      {trajectoryData && !trajectoryData.baseline_available && (
        <Alert
          severity="info"
          sx={{
            mb: 1,
            bgcolor: "#1e293b",
            color: "#93c5fd",
            "& .MuiAlert-icon": { color: "#3b82f6" },
            fontSize: "10px",
          }}
        >
          Baseline action trajectories not stored in ablation data. Showing ablated trajectory only.
          Baseline success rate: {((trajectoryData.baseline_success_rate || 0) * 100).toFixed(0)}%.
        </Alert>
      )}

      {/* Chart */}
      {chartData.length > 0 ? (
        <Box
          sx={{
            bgcolor: "#0f172a",
            borderRadius: 1,
            border: "1px solid #1e293b",
            p: 2,
          }}
        >
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="step"
                stroke="#475569"
                fontSize={9}
                label={{
                  value: "Timestep",
                  position: "bottom",
                  fill: "#475569",
                  fontSize: 9,
                }}
              />
              <YAxis
                stroke="#475569"
                fontSize={9}
                label={{
                  value: "Action Value",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#475569",
                  fontSize: 9,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: 4,
                  fontSize: 10,
                  color: "#e2e8f0",
                }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Legend
                wrapperStyle={{ fontSize: 9, color: "#94a3b8" }}
              />
              {selectedDims.map((dim) => (
                <React.Fragment key={dim}>
                  <Line
                    type="monotone"
                    dataKey={`baseline_${DIM_LABELS[dim]}`}
                    stroke={DIM_COLORS[dim]}
                    strokeWidth={2}
                    dot={false}
                    name={`baseline ${DIM_LABELS[dim]}`}
                    connectNulls={false}
                  />
                  <Line
                    type="monotone"
                    dataKey={`ablated_${DIM_LABELS[dim]}`}
                    stroke={DIM_COLORS[dim]}
                    strokeWidth={2}
                    strokeDasharray="5 3"
                    dot={false}
                    name={`ablated ${DIM_LABELS[dim]}`}
                    opacity={0.7}
                    connectNulls={false}
                  />
                </React.Fragment>
              ))}
            </LineChart>
          </ResponsiveContainer>
          <Box sx={{ display: "flex", justifyContent: "center", mt: 1, gap: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <Box
                sx={{
                  width: 20,
                  height: 2,
                  bgcolor: "#94a3b8",
                }}
              />
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
                Baseline (solid)
              </Typography>
            </Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <Box
                sx={{
                  width: 20,
                  height: 0,
                  borderTop: "2px dashed #94a3b8",
                }}
              />
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
                Ablated (dashed)
              </Typography>
            </Box>
          </Box>
        </Box>
      ) : (
        !loading && (
          <Box
            sx={{
              height: 200,
              bgcolor: "#0f172a",
              borderRadius: 1,
              border: "1px solid #1e293b",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Typography variant="caption" sx={{ color: "#475569" }}>
              Select a file, concept, and task to view action trajectories
            </Typography>
          </Box>
        )
      )}
    </Box>
  );
}
