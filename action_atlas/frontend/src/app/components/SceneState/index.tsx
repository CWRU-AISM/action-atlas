"use client";
import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Box,
  Paper,
  Typography,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Divider,
} from "@mui/material";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import RouteIcon from "@mui/icons-material/Route";
import { API_BASE_URL } from "@/config/api";
import { useAppSelector } from "@/redux/hooks";
import { VLA_MODELS, VLAModelType, DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
import dynamic from "next/dynamic";
import * as THREE from "three";

// Dynamically import Canvas to avoid SSR issues with Three.js
const Canvas = dynamic(
  () => import("@react-three/fiber").then((mod) => mod.Canvas),
  { ssr: false }
);
const OrbitControls = dynamic(
  () => import("@react-three/drei").then((mod) => mod.OrbitControls),
  { ssr: false }
);

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------
interface PairInfo {
  id: string;
  task_a: number;
  task_b: number;
  prompt_a: string;
  prompt_b: string;
  conditions: string[];
}

interface SceneData {
  n_steps: number;
  robot_eef_trajectory: number[][];
  object_trajectories: Record<string, number[][]>;
  object_displacements: Record<
    string,
    { distance: number; init_pos: number[]; final_pos: number[] }
  >;
  initial_state: Record<string, unknown>;
  final_state: Record<string, unknown>;
}

interface SceneStateResponse {
  pair: string;
  condition: string;
  suite: string;
  task_a: number;
  task_b: number;
  prompt_a: string;
  prompt_b: string;
  steps: number;
  success: boolean;
  video_url: string | null;
  scene: SceneData;
}

// ------------------------------------------------------------------
// Color helpers
// ------------------------------------------------------------------
const OBJECT_COLORS = [
  "#f97316", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
  "#14b8a6", "#f43f5e", "#eab308", "#3b82f6", "#a855f7",
];

function displacementColor(d: number): string {
  if (d < 0.005) return "#22c55e";
  if (d < 0.05) return "#f59e0b";
  return "#ef4444";
}

/** Viridis colormap stops */
const VIRIDIS_STOPS = [
  { t: 0.0, r: 68, g: 1, b: 84 },     // dark purple
  { t: 0.25, r: 59, g: 82, b: 139 },   // blue-purple
  { t: 0.5, r: 33, g: 145, b: 140 },   // teal
  { t: 0.75, r: 94, g: 201, b: 98 },   // green
  { t: 1.0, r: 253, g: 231, b: 37 },   // yellow
];

/** Viridis-like colormap returning [r, g, b] in 0-255 range */
function viridisRGB(t: number): [number, number, number] {
  const clamp = Math.max(0, Math.min(1, t));
  let lo = VIRIDIS_STOPS[0], hi = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1];
  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    if (clamp >= VIRIDIS_STOPS[i].t && clamp <= VIRIDIS_STOPS[i + 1].t) {
      lo = VIRIDIS_STOPS[i];
      hi = VIRIDIS_STOPS[i + 1];
      break;
    }
  }
  const f = hi.t === lo.t ? 0 : (clamp - lo.t) / (hi.t - lo.t);
  return [
    Math.round(lo.r + f * (hi.r - lo.r)),
    Math.round(lo.g + f * (hi.g - lo.g)),
    Math.round(lo.b + f * (hi.b - lo.b)),
  ];
}

/** Viridis returning CSS rgb string */
function viridisColor(t: number): string {
  const [r, g, b] = viridisRGB(t);
  return `rgb(${r},${g},${b})`;
}

// ------------------------------------------------------------------
// 3D Trajectory Plot -- Three.js with viridis colormap
// ------------------------------------------------------------------
interface TrajectoryPlotProps {
  scene: SceneData | null;
  label: string;
  color: string;
  taskPrompt?: string;
  success?: boolean;
  stepCount?: number;
  overlayScene?: SceneData | null;
  overlayColor?: string;
  overlayLabel?: string;
  overlaySuccess?: boolean;
  useViridis?: boolean;
}

/** Build a colored tube geometry from trajectory points with viridis coloring */
function ViridisTrajectoryTube({
  points,
  tubeRadius = 0.008,
}: {
  points: THREE.Vector3[];
  tubeRadius?: number;
}) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.3);
    const tubularSegments = Math.max(points.length * 4, 64);
    const geo = new THREE.TubeGeometry(curve, tubularSegments, tubeRadius, 8, false);
    // Color each vertex by its position along the tube (t parameter)
    const count = geo.attributes.position.count;
    const colors = new Float32Array(count * 3);
    // TubeGeometry: segments along length = tubularSegments+1, radial = 8+1 = 9
    const radialSegments = 9;
    const lengthSegments = tubularSegments + 1;
    for (let i = 0; i < count; i++) {
      const lengthIdx = Math.floor(i / radialSegments);
      const t = lengthIdx / (lengthSegments - 1);
      const [r, g, b] = viridisRGB(t);
      colors[i * 3] = r / 255;
      colors[i * 3 + 1] = g / 255;
      colors[i * 3 + 2] = b / 255;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geo;
  }, [points, tubeRadius]);

  if (!geometry) return null;
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.4} metalness={0.1} />
    </mesh>
  );
}

/** Overlay trajectory rendered as a solid-color dashed tube */
function SolidTrajectoryTube({
  points,
  color,
  tubeRadius = 0.006,
  opacity = 0.6,
}: {
  points: THREE.Vector3[];
  color: string;
  tubeRadius?: number;
  opacity?: number;
}) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.3);
    const tubularSegments = Math.max(points.length * 4, 64);
    return new THREE.TubeGeometry(curve, tubularSegments, tubeRadius, 8, false);
  }, [points, tubeRadius]);

  if (!geometry) return null;
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} transparent opacity={opacity} roughness={0.5} />
    </mesh>
  );
}

/** Grid plane at bottom of the scene */
function SceneGrid({ size, divisions }: { size: number; divisions: number }) {
  return (
    <gridHelper
      args={[size, divisions, "#334155", "#1e293b"]}
      position={[0, 0, 0]}
      rotation={[0, 0, 0]}
    />
  );
}

/** A single axis rendered as a thin cylinder */
function AxisCylinder({
  from,
  to,
  color,
}: {
  from: [number, number, number];
  to: [number, number, number];
  color: string;
}) {
  const { position, quaternion, length } = useMemo(() => {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const dir = new THREE.Vector3().subVectors(end, start);
    const len = dir.length();
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    const quat = new THREE.Quaternion();
    // CylinderGeometry is along Y by default; rotate to align with direction
    const up = new THREE.Vector3(0, 1, 0);
    quat.setFromUnitVectors(up, dir.clone().normalize());
    return { position: mid, quaternion: quat, length: len };
  }, [from, to]);

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[0.003, 0.003, length, 6]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

/** Axis lines and labels */
function AxisLabels({ size }: { size: number }) {
  const half = size / 2;
  return (
    <group>
      {/* X axis - red */}
      <AxisCylinder from={[0, 0, 0]} to={[half, 0, 0]} color="#ef4444" />
      {/* Y axis - green (up) */}
      <AxisCylinder from={[0, 0, 0]} to={[0, half, 0]} color="#22c55e" />
      {/* Z axis - blue */}
      <AxisCylinder from={[0, 0, 0]} to={[0, 0, half]} color="#3b82f6" />
      {/* Axis endpoint labels as small colored spheres */}
      <mesh position={[half + 0.02, 0, 0]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshStandardMaterial color="#ef4444" />
      </mesh>
      <mesh position={[0, half + 0.02, 0]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshStandardMaterial color="#22c55e" />
      </mesh>
      <mesh position={[0, 0, half + 0.02]}>
        <sphereGeometry args={[0.008, 8, 8]} />
        <meshStandardMaterial color="#3b82f6" />
      </mesh>
    </group>
  );
}

/** Object sphere with label */
function ObjectMarker({
  position,
  finalPosition,
  name,
  color,
}: {
  position: [number, number, number];
  finalPosition?: [number, number, number];
  name: string;
  color: string;
}) {
  return (
    <group>
      {/* Initial position - solid sphere */}
      <mesh position={position}>
        <sphereGeometry args={[0.012, 16, 16]} />
        <meshStandardMaterial color={color} roughness={0.3} />
      </mesh>
      {/* Final position - wireframe sphere */}
      {finalPosition && (
        <>
          <mesh position={finalPosition}>
            <sphereGeometry args={[0.01, 16, 16]} />
            <meshStandardMaterial color={color} wireframe opacity={0.8} transparent />
          </mesh>
          {/* Thin cylinder from init to final position */}
          <AxisCylinder from={position} to={finalPosition} color={color} />
        </>
      )}
    </group>
  );
}

/** The 3D scene content */
function TrajectoryScene({
  scene,
  overlayScene,
  overlayColor,
}: {
  scene: SceneData;
  overlayScene?: SceneData | null;
  overlayColor?: string;
}) {
  // Compute center and scale for camera positioning
  const { center, sceneScale, trajectoryPoints, overlayPoints, gridSize } = useMemo(() => {
    const allPts: number[][] = [];
    const addTraj = (traj: number[][]) => {
      traj.forEach((p) => {
        if (p && p.length >= 3) allPts.push(p);
      });
    };

    if (scene.robot_eef_trajectory) addTraj(scene.robot_eef_trajectory);
    if (overlayScene?.robot_eef_trajectory) addTraj(overlayScene.robot_eef_trajectory);
    if (scene.object_trajectories) {
      Object.values(scene.object_trajectories).forEach((t) => {
        if (t && Array.isArray(t)) addTraj(t);
      });
    }

    if (allPts.length === 0) {
      return {
        center: new THREE.Vector3(0, 0, 0),
        sceneScale: 1,
        trajectoryPoints: [] as THREE.Vector3[],
        overlayPoints: [] as THREE.Vector3[],
        gridSize: 1,
      };
    }

    const xs = allPts.map((p) => p[0]);
    const ys = allPts.map((p) => p[1]);
    const zs = allPts.map((p) => p[2]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const rangeX = Math.max(...xs) - Math.min(...xs);
    const rangeY = Math.max(...ys) - Math.min(...ys);
    const rangeZ = Math.max(...zs) - Math.min(...zs);
    const maxRange = Math.max(rangeX, rangeY, rangeZ, 0.01);

    // Normalize: center data and scale so max range fits in ~1 unit
    const scale = 1 / maxRange;

    const toLocal = (p: number[]): THREE.Vector3 =>
      new THREE.Vector3(
        (p[0] - cx) * scale,
        (p[2] - cz) * scale, // Z becomes Y (up) in Three.js
        (p[1] - cy) * scale  // Y becomes Z (depth) in Three.js
      );

    const trajPts = scene.robot_eef_trajectory
      ? scene.robot_eef_trajectory.filter((p) => p && p.length >= 3).map(toLocal)
      : [];
    const ovPts = overlayScene?.robot_eef_trajectory
      ? overlayScene.robot_eef_trajectory.filter((p) => p && p.length >= 3).map(toLocal)
      : [];

    return {
      center: new THREE.Vector3(cx, cy, cz),
      sceneScale: scale,
      trajectoryPoints: trajPts,
      overlayPoints: ovPts,
      gridSize: 1.2,
    };
  }, [scene, overlayScene]);

  // Transform object trajectories to local coordinates
  const objectData = useMemo(() => {
    if (!scene.object_trajectories) return [];
    const toLocal = (p: number[]): [number, number, number] => [
      (p[0] - center.x) * sceneScale,
      (p[2] - center.z) * sceneScale, // Z -> Y (up)
      (p[1] - center.y) * sceneScale, // Y -> Z (depth)
    ];
    return Object.entries(scene.object_trajectories).map(([name, traj], idx) => {
      if (!traj || traj.length === 0 || !traj[0] || traj[0].length < 3) return null;
      const first = toLocal(traj[0]);
      const last = traj.length > 1 && traj[traj.length - 1]?.length >= 3
        ? toLocal(traj[traj.length - 1])
        : undefined;
      return {
        name,
        color: OBJECT_COLORS[idx % OBJECT_COLORS.length],
        position: first,
        finalPosition: last,
      };
    }).filter(Boolean) as { name: string; color: string; position: [number, number, number]; finalPosition?: [number, number, number] }[];
  }, [scene, center, sceneScale]);

  const hasTrajectory = trajectoryPoints.length > 1;
  const hasOverlay = overlayPoints.length > 1;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      <directionalLight position={[-3, 5, -3]} intensity={0.3} />

      {/* Grid */}
      <SceneGrid size={gridSize} divisions={10} />

      {/* Axis indicators */}
      <AxisLabels size={gridSize} />

      {/* Primary trajectory - viridis colored tube */}
      {hasTrajectory && (
        <>
          <ViridisTrajectoryTube points={trajectoryPoints} tubeRadius={0.008} />
          {/* Start marker - green sphere */}
          <mesh position={trajectoryPoints[0]}>
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.3} />
          </mesh>
          {/* End marker - red sphere */}
          <mesh position={trajectoryPoints[trajectoryPoints.length - 1]}>
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.3} />
          </mesh>
        </>
      )}

      {/* Overlay trajectory - solid color tube */}
      {hasOverlay && (
        <>
          <SolidTrajectoryTube
            points={overlayPoints}
            color={overlayColor || "#94a3b8"}
            tubeRadius={0.006}
            opacity={0.6}
          />
          {/* Start marker */}
          <mesh position={overlayPoints[0]}>
            <sphereGeometry args={[0.015, 16, 16]} />
            <meshStandardMaterial color="#22c55e" transparent opacity={0.7} />
          </mesh>
          {/* End marker */}
          <mesh position={overlayPoints[overlayPoints.length - 1]}>
            <sphereGeometry args={[0.015, 16, 16]} />
            <meshStandardMaterial color="#ef4444" transparent opacity={0.7} />
          </mesh>
        </>
      )}

      {/* Object markers */}
      {objectData.map((obj) => (
        <ObjectMarker
          key={obj.name}
          position={obj.position}
          finalPosition={obj.finalPosition}
          name={obj.name}
          color={obj.color}
        />
      ))}
    </>
  );
}

function TrajectoryPlot({
  scene,
  label,
  color,
  taskPrompt,
  success,
  stepCount,
  overlayScene,
  overlayColor,
  overlayLabel,
  overlaySuccess,
  useViridis = true,
}: TrajectoryPlotProps) {
  const plotHeight = 440;

  // Check if we have any trajectory data at all
  const hasTrajectory = scene?.robot_eef_trajectory && scene.robot_eef_trajectory.length > 0;
  const hasObjects = scene?.object_trajectories && Object.keys(scene.object_trajectories).length > 0;

  // Empty state: no scene data at all
  if (!scene) {
    return (
      <Box
        sx={{
          width: "100%",
          minHeight: 300,
          bgcolor: "#0f172a",
          borderRadius: 2,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          border: "1px solid #1e293b",
          p: 3,
        }}
      >
        <Typography variant="body2" sx={{ color: "#64748b", mb: 1 }}>
          No trajectory data available
        </Typography>
        <Typography variant="caption" sx={{ color: "#475569", textAlign: "center" }}>
          Select a task pair and condition above to view the EEF trajectory.
        </Typography>
      </Box>
    );
  }

  // Scene exists but no trajectory inside it
  if (!hasTrajectory && !hasObjects) {
    return (
      <Box
        sx={{
          width: "100%",
          minHeight: 300,
          bgcolor: "#0f172a",
          borderRadius: 2,
          border: "1px solid #334155",
          p: 3,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 1,
        }}
      >
        <Typography variant="subtitle2" sx={{ color: "#94a3b8", fontWeight: 600 }}>
          {label}
        </Typography>
        {taskPrompt && (
          <Typography variant="caption" sx={{ color: "#64748b", textAlign: "center", maxWidth: 400 }}>
            &ldquo;{taskPrompt}&rdquo;
          </Typography>
        )}
        <Alert
          severity="info"
          sx={{
            mt: 1,
            bgcolor: "#1e293b",
            color: "#94a3b8",
            "& .MuiAlert-icon": { color: "#3b82f6" },
            maxWidth: 400,
          }}
        >
          This condition has no trajectory data. The rollout may not have been recorded for this pair/condition combination.
        </Alert>
        {success !== undefined && (
          <Chip
            label={success ? "Task Succeeded" : "Task Failed"}
            size="small"
            sx={{
              mt: 1,
              height: 22,
              fontSize: "10px",
              fontWeight: 600,
              bgcolor: success ? "#166534" : "#991b1b",
              color: success ? "#86efac" : "#fca5a5",
            }}
          />
        )}
      </Box>
    );
  }

  const nSteps = scene.n_steps || stepCount || 0;

  return (
    <Box
      sx={{
        bgcolor: "#0f172a",
        borderRadius: 2,
        border: "1px solid #1e293b",
        p: 1.5,
        overflow: "hidden",
      }}
    >
      {/* Header: label, task prompt, success badge */}
      <Box sx={{ mb: 1 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5, flexWrap: "wrap" }}>
          <Typography
            variant="caption"
            sx={{ color: "#e2e8f0", fontWeight: 700, fontSize: "12px" }}
          >
            {label}
          </Typography>
          {success !== undefined && (
            <Chip
              label={success ? "SUCCESS" : "FAILED"}
              size="small"
              sx={{
                height: 20,
                fontSize: "9px",
                fontWeight: 700,
                bgcolor: success ? "#166534" : "#991b1b",
                color: success ? "#86efac" : "#fca5a5",
                letterSpacing: "0.5px",
              }}
            />
          )}
          {overlayLabel && overlaySuccess !== undefined && (
            <Chip
              label={`Overlay: ${overlaySuccess ? "SUCCESS" : "FAILED"}`}
              size="small"
              variant="outlined"
              sx={{
                height: 20,
                fontSize: "9px",
                fontWeight: 600,
                borderColor: overlaySuccess ? "#166534" : "#991b1b",
                color: overlaySuccess ? "#86efac" : "#fca5a5",
              }}
            />
          )}
          {nSteps > 0 && (
            <Typography variant="caption" sx={{ color: "#475569", fontSize: "10px", ml: "auto" }}>
              {nSteps} steps
            </Typography>
          )}
        </Box>
        {taskPrompt && (
          <Typography
            variant="caption"
            sx={{ color: "#94a3b8", fontSize: "11px", display: "block", lineHeight: 1.3 }}
          >
            &ldquo;{taskPrompt}&rdquo;
          </Typography>
        )}
      </Box>

      {/* 3D Canvas */}
      <Box
        sx={{
          width: "100%",
          height: plotHeight,
          borderRadius: 1,
          overflow: "hidden",
          bgcolor: "#0a0f1e",
          border: "1px solid #1e293b",
        }}
      >
        <Canvas
          camera={{
            position: [0.8, 0.8, 0.8],
            fov: 50,
            near: 0.01,
            far: 100,
          }}
          style={{ width: "100%", height: "100%" }}
          gl={{ antialias: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.setClearColor("#0a0f1e", 1);
          }}
        >
          <OrbitControls
            enablePan
            enableZoom
            enableRotate
            autoRotate={false}
            minDistance={0.2}
            maxDistance={5}
            target={[0, 0, 0]}
          />
          <TrajectoryScene
            scene={scene}
            overlayScene={overlayScene}
            overlayColor={overlayColor}
          />
        </Canvas>
      </Box>

      {/* Legend bar */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mt: 1,
          px: 1,
          flexWrap: "wrap",
          gap: 1,
        }}
      >
        {/* Viridis gradient legend */}
        {useViridis && hasTrajectory && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px", mr: 0.5 }}>
              Time:
            </Typography>
            <Box
              sx={{
                width: 80,
                height: 8,
                borderRadius: 1,
                background: "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))",
                border: "1px solid #334155",
              }}
            />
            <Typography variant="caption" sx={{ color: "#475569", fontSize: "8px", ml: 0.5 }}>
              t=0 ... t=1
            </Typography>
          </Box>
        )}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                bgcolor: "#22c55e",
                border: "1.5px solid #fff",
              }}
            />
            <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "10px" }}>
              Start
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                bgcolor: "#ef4444",
                border: "1.5px solid #fff",
              }}
            />
            <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "10px" }}>
              End
            </Typography>
          </Box>
          {overlayLabel && overlayColor && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              <Box
                sx={{
                  width: 16,
                  height: 0,
                  borderTop: `2px dashed ${overlayColor}`,
                }}
              />
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "9px" }}>
                {overlayLabel}
              </Typography>
            </Box>
          )}
          <Typography variant="caption" sx={{ color: "#475569", fontSize: "9px", ml: 1 }}>
            Drag to rotate / scroll to zoom / right-click to pan
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}

// ------------------------------------------------------------------
// Main SceneState Component
// ------------------------------------------------------------------
// ---------------------------------------------------------------
// Model display name helper
// ---------------------------------------------------------------
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  pi05: "Pi0.5",
  openvla: "OpenVLA-OFT",
  xvla: "X-VLA",
  smolvla: "SmolVLA",
  groot: "GR00T N1.5",
};

// ---------------------------------------------------------------
// Suite options per model for scene-state
// ---------------------------------------------------------------
const MODEL_SUITES: Record<string, { value: string; label: string }[]> = {
  pi05: [
    { value: "goal", label: "LIBERO Goal" },
    { value: "spatial", label: "LIBERO Spatial" },
    { value: "object", label: "LIBERO Object" },
    { value: "10", label: "LIBERO-10" },
  ],
  openvla: [
    { value: "goal", label: "LIBERO Goal" },
    { value: "spatial", label: "LIBERO Spatial" },
    { value: "object", label: "LIBERO Object" },
    { value: "10", label: "LIBERO-10" },
  ],
  xvla: [
    { value: "goal", label: "LIBERO Goal" },
    { value: "spatial", label: "LIBERO Spatial" },
    { value: "object", label: "LIBERO Object" },
    { value: "10", label: "LIBERO-10" },
  ],
  smolvla: [
    { value: "goal", label: "LIBERO Goal" },
    { value: "spatial", label: "LIBERO Spatial" },
    { value: "object", label: "LIBERO Object" },
    { value: "10", label: "LIBERO-10" },
  ],
  groot: [
    { value: "goal", label: "LIBERO Goal" },
    { value: "object", label: "LIBERO Object" },
    { value: "long", label: "LIBERO Long" },
  ],
};

// ---------------------------------------------------------------
// Displacement analysis inline data
// ---------------------------------------------------------------
interface DisplacementMetrics {
  sourceOverrideRate: number;
  overrideRateLabel: string;
  keyFinding: string;
  episodeCount: number;
  details: { metric: string; value: string; description: string }[];
}

interface DisplacementData {
  modelName: string;
  sourceOverrideRate: number;
  overrideRateLabel: string;
  expertVsVlmRatio: number;
  keyFinding: string;
  episodeCount: number;
  details: { metric: string; value: string; description: string }[];
  // Optional object displacement metrics (toggle-able)
  objectDisplacement?: DisplacementMetrics;
}

const DISPLACEMENT_DATA: Partial<Record<string, DisplacementData>> = {
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
    objectDisplacement: {
      sourceOverrideRate: 0,
      overrideRateLabel: "Object displacement under counterfactual/VP conditions",
      keyFinding: "Object displacement analysis across 4 LIBERO suites shows how counterfactual prompts and vision perturbations alter object interactions. Null prompts produce the largest displacement changes.",
      episodeCount: 3150,
      details: [
        { metric: "Counterfactual Suites", value: "4", description: "LIBERO Goal, Object, Spatial, LIBERO-10" },
        { metric: "VP Suites", value: "4", description: "Vision perturbation displacement across all suites" },
        { metric: "Key Object (ketchup)", value: "0.54", description: "Mean displacement for ketchup under null prompt" },
        { metric: "Key Object (alphabet_soup)", value: "0.42", description: "Mean displacement for alphabet_soup under null prompt" },
      ],
    },
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
    objectDisplacement: {
      sourceOverrideRate: 14.7,
      overrideRateLabel: "Expert 14.7% override (cosine, MetaWorld Easy)",
      keyFinding: "MetaWorld cross-task injection: expert pathway mean cosine to source 0.15 vs destination 0.41. VLM pathway shows weaker override (9.0%). Displacement varies by difficulty level, with easier tasks showing clearer source behavior transfer.",
      episodeCount: 4576,
      details: [
        { metric: "Expert Cos to Source", value: "0.15", description: "Mean cosine similarity to source (MetaWorld Easy, expert_all)" },
        { metric: "Expert Cos to Dest", value: "0.41", description: "Mean cosine similarity to destination (MetaWorld Easy)" },
        { metric: "Expert Early Cos Src", value: "-0.27", description: "Expert early layers: negative cosine to source" },
        { metric: "Expert Mid Cos Src", value: "0.13", description: "Expert mid layers: weak source alignment" },
        { metric: "Total Episodes", value: "4,576", description: "MetaWorld Easy difficulty level" },
      ],
    },
  },
  pi05: {
    modelName: "Pi0.5",
    sourceOverrideRate: 99.6,
    overrideRateLabel: "99.6% episodes more similar to source",
    expertVsVlmRatio: 0,
    keyFinding: "Cross-task injection produces near-total source behavior override. Expert pathway: 98.8% override rate. PaliGemma pathway: 100% override rate. Both pathways show strong source trajectory alignment (expert cos=0.92, PaliGemma cos=1.00).",
    episodeCount: 1968,
    details: [
      { metric: "Override Rate", value: "99.6%", description: "Episodes where injected trajectory matches source more than destination" },
      { metric: "Mean Cosine Sim (Source)", value: "0.97", description: "Average cosine similarity to source trajectory" },
      { metric: "Mean Cosine Sim (Dest)", value: "0.13", description: "Average cosine similarity to destination trajectory" },
      { metric: "Expert Override", value: "98.8%", description: "Expert pathway override rate (656 episodes)" },
      { metric: "PaliGemma Override", value: "100%", description: "PaliGemma pathway override rate (1,312 episodes)" },
    ],
  },
  openvla: {
    modelName: "OpenVLA-OFT",
    sourceOverrideRate: 77.9,
    overrideRateLabel: "77.9% episodes more similar to source",
    expertVsVlmRatio: 0,
    keyFinding: "Cross-task injection overrides destination behavior in 77.9% of episodes. Mean cosine similarity to source (0.82) vs destination (0.38) confirms source trajectory dominance, though weaker than single-pathway models.",
    episodeCount: 1079,
    details: [
      { metric: "Override Rate", value: "77.9%", description: "Episodes where injected trajectory matches source more than destination" },
      { metric: "Mean Cosine Sim (Source)", value: "0.82", description: "Average cosine similarity to source trajectory" },
      { metric: "Mean Cosine Sim (Dest)", value: "0.38", description: "Average cosine similarity to destination trajectory" },
      { metric: "Success Rate", value: "5.0%", description: "Task success rate under injection (vs baseline)" },
      { metric: "Suites", value: "4", description: "LIBERO Goal, Object, Spatial, LIBERO-10" },
    ],
  },
  groot: {
    modelName: "GR00T N1.5",
    sourceOverrideRate: 0,
    overrideRateLabel: "Cross-task injection (17 pairs x 3 suites)",
    expertVsVlmRatio: 0,
    keyFinding: "GR00T cross-task injection across 17 task pairs and 3 LIBERO suites. EEF trajectory data available for cosine similarity computation. DiT, Eagle LM, and VL-SA pathways tested with cross-prompt and own-prompt conditions.",
    episodeCount: 17,
    details: [
      { metric: "Task Pairs", value: "17", description: "Cross-task injection pairs per suite" },
      { metric: "Suites", value: "3", description: "LIBERO Goal, Object, Long" },
      { metric: "Conditions", value: "8+", description: "DiT, Eagle, VL-SA layer injections per pair" },
      { metric: "Trajectory Data", value: "Yes", description: "Full EEF + object trajectories available" },
    ],
    objectDisplacement: {
      sourceOverrideRate: 0,
      overrideRateLabel: "Object displacement analysis (123K scenes)",
      keyFinding: "GR00T object displacement across 123,612 scenes shows pathway-specific effects: DiT layers cause avg max displacement of 0.232, Eagle 0.285, VL-SA 0.296. DiT layers run fewer steps (264 avg), suggesting earlier task failure.",
      episodeCount: 123612,
      details: [
        { metric: "DiT Displacement", value: "0.232", description: "Average max object displacement from DiT layer ablation (43,891 scenes)" },
        { metric: "Eagle Displacement", value: "0.285", description: "Average max object displacement from Eagle layer ablation (61,020 scenes)" },
        { metric: "VL-SA Displacement", value: "0.296", description: "Average max object displacement from VL-SA layer ablation (18,701 scenes)" },
        { metric: "DiT Avg Steps", value: "264", description: "Average episode length under DiT ablation" },
        { metric: "Total Scenes", value: "123,612", description: "Total scenes analyzed across all layer types" },
      ],
    },
  },
};

const DEFAULT_DISPLACEMENT: DisplacementData = {
  modelName: "",
  sourceOverrideRate: 0,
  overrideRateLabel: "No displacement data available",
  expertVsVlmRatio: 0,
  keyFinding: "No displacement data available for this model.",
  episodeCount: 0,
  details: [],
};

type SceneStateTab = "trajectories" | "displacement";

export default function SceneState() {
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);
  const model = currentModel;
  const [suite, setSuite] = useState<string>("goal");
  const [pairs, setPairs] = useState<PairInfo[]>([]);
  const [selectedPair, setSelectedPair] = useState<string>("");
  const [baselineCondition, setBaselineCondition] = useState<string>("");
  const [injectionCondition, setInjectionCondition] = useState<string>("");
  const [baselineScene, setBaselineScene] = useState<SceneStateResponse | null>(null);
  const [injectionScene, setInjectionScene] = useState<SceneStateResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingPairs, setLoadingPairs] = useState(true);
  const [pairsError, setPairsError] = useState<string | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"side_by_side" | "overlay">("side_by_side");
  const [sceneTab, setSceneTab] = useState<SceneStateTab>("trajectories");

  // Derive suite options from current dataset, falling back to hardcoded MODEL_SUITES.
  // The SceneState API uses short names for LIBERO (goal, spatial, object, 10)
  // but full names for MetaWorld (metaworld_easy) and SimplerEnv (simplerenv_widowx).
  const availableModelSuites = useMemo(() => {
    const dsEntry = DATASET_SUITES[currentDataset as DatasetType] || [];
    if (dsEntry.length > 0) {
      return dsEntry.map(s => ({
        // Only strip libero_ prefix; MetaWorld and SimplerEnv use full names in the API
        value: s.value.replace(/^libero_/, ''),
        label: s.display,
      }));
    }
    return MODEL_SUITES[model] || MODEL_SUITES.pi05;
  }, [currentDataset, model]);

  // Reset suite when model or dataset changes (if current suite is not available)
  useEffect(() => {
    if (availableModelSuites.length > 0 && !availableModelSuites.some((s) => s.value === suite)) {
      setSuite(availableModelSuites[0].value);
    }
  }, [model, currentDataset, availableModelSuites]);

  // Combined error for display
  const error = pairsError || sceneError;

  // Load pairs when model or suite changes
  useEffect(() => {
    let cancelled = false;
    const fetchPairs = async () => {
      setLoadingPairs(true);
      setPairsError(null);
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/vla/scene_state/pairs?suite=${suite}&model=${model}`
        );
        if (cancelled) return;
        if (!res.ok) {
          // Try to parse error body for a better message
          let msg = `Failed to fetch pairs: ${res.statusText}`;
          try {
            const errBody = await res.json();
            if (errBody?.error) msg = errBody.error;
          } catch { /* ignore parse error */ }
          throw new Error(msg);
        }
        const data = await res.json();
        if (cancelled) return;
        const fetchedPairs = data.pairs || [];
        setPairs(fetchedPairs);
        if (fetchedPairs.length > 0) {
          const firstPair = fetchedPairs[0];
          setSelectedPair(firstPair.id);
          // Set default baseline to first baseline condition found for this pair
          // Support multiple naming conventions: baseline_task_X, src_taskX_baseline, etc.
          const firstBaseline = firstPair.conditions.find((c: string) =>
            c.startsWith("baseline_task_")
          ) || firstPair.conditions.find((c: string) =>
            c.endsWith("_baseline") || (c.toLowerCase().includes("baseline") && !c.includes("cross_prompt") && !c.includes("own_prompt"))
          ) || (firstPair.conditions.length > 0 ? firstPair.conditions[0] : "");
          if (firstBaseline) setBaselineCondition(firstBaseline);
        } else {
          setPairsError(`No scene state data available for this suite. The ${MODEL_DISPLAY_NAMES[model] || model} rollout data may not be mounted.`);
        }
      } catch (e) {
        if (cancelled) return;
        setPairsError(e instanceof Error ? e.message : "Failed to load pairs");
        setPairs([]);
      } finally {
        if (!cancelled) setLoadingPairs(false);
      }
    };
    fetchPairs();
    return () => { cancelled = true; };
  }, [suite, model]);

  // Get current pair info
  const currentPair = useMemo(
    () => pairs.find((p) => p.id === selectedPair),
    [pairs, selectedPair]
  );

  // Helper: detect whether a condition name represents a baseline
  const isBaselineCondition = useCallback((c: string): boolean => {
    // Standard format: baseline_task_0, baseline_task_1
    if (c.startsWith("baseline_task_")) return true;
    // X-VLA/GR00T format: inject_0_into_1/src_task0_baseline, inject_0_into_1/dst_task1_baseline
    if (c.includes("baseline") && !c.includes("cross_prompt") && !c.includes("own_prompt") && !c.includes("transformer")) return true;
    // Any condition ending with "_baseline"
    if (c.endsWith("_baseline")) return true;
    return false;
  }, []);

  // Get baseline conditions for current pair (dynamic, supports multiple naming formats)
  const baselineConditions = useMemo(() => {
    if (!currentPair) return [];
    const baselines = currentPair.conditions.filter((c) => isBaselineCondition(c));
    // If no baselines found, try to find any condition containing "baseline" as a substring
    if (baselines.length === 0) {
      const fallback = currentPair.conditions.filter((c) => c.toLowerCase().includes("baseline"));
      if (fallback.length > 0) return fallback;
    }
    // If still no baselines, use the first available condition as the baseline
    if (baselines.length === 0 && currentPair.conditions.length > 0) {
      return [currentPair.conditions[0]];
    }
    return baselines;
  }, [currentPair, isBaselineCondition]);

  // Get injection conditions for current pair (everything that isn't a baseline)
  const injectionConditions = useMemo(() => {
    if (!currentPair) return [];
    return currentPair.conditions.filter((c) => !isBaselineCondition(c));
  }, [currentPair, isBaselineCondition]);

  // Reset baseline condition when pair changes (if current value isn't valid)
  useEffect(() => {
    if (baselineConditions.length > 0 && !baselineConditions.includes(baselineCondition)) {
      setBaselineCondition(baselineConditions[0]);
    }
  }, [baselineConditions, baselineCondition]);

  // Set default injection condition when pair changes
  useEffect(() => {
    if (injectionConditions.length > 0 && !injectionConditions.includes(injectionCondition)) {
      setInjectionCondition(injectionConditions[0]);
    }
  }, [injectionConditions, injectionCondition]);

  // Load scene data
  const fetchScene = useCallback(
    async (condition: string): Promise<SceneStateResponse | null> => {
      const res = await fetch(
        `${API_BASE_URL}/api/vla/scene_state?suite=${suite}&pair=${selectedPair}&condition=${encodeURIComponent(condition)}&model=${model}`
      );
      if (!res.ok) {
        let msg = `Scene data unavailable (${res.status})`;
        try {
          const errBody = await res.json();
          if (errBody?.error) msg = errBody.error;
        } catch { /* ignore parse error */ }
        throw new Error(msg);
      }
      return await res.json();
    },
    [suite, selectedPair, model]
  );

  // Fetch both scenes when selection changes
  useEffect(() => {
    if (!selectedPair || !baselineCondition || pairs.length === 0) return;

    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setSceneError(null);
      try {
        const [baseline, injection] = await Promise.all([
          fetchScene(baselineCondition),
          injectionCondition ? fetchScene(injectionCondition) : Promise.resolve(null),
        ]);
        if (cancelled) return;
        setBaselineScene(baseline);
        setInjectionScene(injection);
      } catch (e) {
        if (cancelled) return;
        setSceneError(e instanceof Error ? e.message : "Failed to load scene data");
        setBaselineScene(null);
        setInjectionScene(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [selectedPair, baselineCondition, injectionCondition, fetchScene, pairs.length]);

  // Format condition name for display
  const formatCondition = (c: string) => {
    if (!c) return "";

    // Handle baseline with dynamic task IDs
    const baselineMatch = c.match(/^baseline_task_(\d+)$/);
    if (baselineMatch) {
      const taskId = baselineMatch[1];
      // Use A/B labeling relative to the current pair
      if (currentPair) {
        const label = String(currentPair.task_a) === taskId ? "A" : "B";
        return `Baseline Task ${label} (T${taskId})`;
      }
      return `Baseline Task ${taskId}`;
    }

    // Handle injection direction patterns: inject_X_into_Y/condition
    const injectMatch = c.match(/^inject_(\d+)_into_(\d+)\/(.+)$/);
    if (injectMatch) {
      const [, srcId, tgtId, condName] = injectMatch;
      // Build human-readable source/target labels
      let srcLabel = `T${srcId}`;
      let tgtLabel = `T${tgtId}`;
      if (currentPair) {
        if (String(currentPair.task_a) === srcId) srcLabel = `A(T${srcId})`;
        else if (String(currentPair.task_b) === srcId) srcLabel = `B(T${srcId})`;
        if (String(currentPair.task_a) === tgtId) tgtLabel = `A(T${tgtId})`;
        else if (String(currentPair.task_b) === tgtId) tgtLabel = `B(T${tgtId})`;
      }
      // Handle baseline conditions within injection directions
      if (condName.includes("baseline")) {
        const srcBaseMatch = condName.match(/src_task(\d+)_baseline/);
        const dstBaseMatch = condName.match(/dst_task(\d+)_baseline/);
        if (srcBaseMatch) {
          return `Baseline Source ${srcLabel} (T${srcBaseMatch[1]})`;
        }
        if (dstBaseMatch) {
          return `Baseline Dest ${tgtLabel} (T${dstBaseMatch[1]})`;
        }
      }
      const readableCond = condName
        .replace("cross_prompt_", "x-prompt ")
        .replace("own_prompt_", "own-prompt ")
        .replace("pali_ALL", "PaliGemma(ALL)")
        .replace("pali_L0", "PaliGemma(L0)")
        .replace(/expert_L16_L17/g, "Expert(L16+L17)")
        .replace(/expert_L16/g, "Expert(L16)")
        .replace(/transformer_ALL/g, "All Layers")
        .replace(/transformer_L(\d+)_L(\d+)/g, "L$1+L$2")
        .replace(/transformer_L(\d+)/g, "L$1")
        .replace("no_inject", "No Injection");
      return `Inject ${srcLabel}\u2192${tgtLabel}: ${readableCond}`;
    }

    // Fallback: replace underscores and return
    return c
      .replace("cross_prompt_", "x-prompt ")
      .replace("own_prompt_", "own-prompt ")
      .replace("pali_ALL", "PaliGemma(ALL)")
      .replace("pali_L0", "PaliGemma(L0)")
      .replace(/expert_L16_L17/g, "Expert(L16+L17)")
      .replace(/expert_L16/g, "Expert(L16)")
      .replace(/transformer_ALL/g, "All Layers")
      .replace(/transformer_L(\d+)/g, "L$1")
      .replace("no_inject", "No Injection");
  };

  return (
    <Paper className="h-full flex flex-col rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className="h-10 flex items-center px-4 bg-[#0a1628] rounded-t-lg border-b border-slate-700">
        <Typography
          variant="subtitle2"
          sx={{ color: "white", fontWeight: 600 }}
        >
          Scene State
        </Typography>
        <Chip
          label={MODEL_DISPLAY_NAMES[model] || model}
          size="small"
          sx={{
            ml: 2,
            height: 18,
            fontSize: "9px",
            bgcolor: "#3b82f6",
            color: "white",
          }}
        />
        <Chip
          label="3D EEF + Objects"
          size="small"
          sx={{
            ml: 1,
            height: 18,
            fontSize: "9px",
            bgcolor: "#1e293b",
            color: "#94a3b8",
          }}
        />
        {/* Sub-tab toggle: Trajectories vs Displacement */}
        <Box sx={{ ml: "auto" }}>
          <ToggleButtonGroup
            value={sceneTab}
            exclusive
            onChange={(_, v) => v && setSceneTab(v)}
            size="small"
            sx={{
              "& .MuiToggleButton-root": {
                color: "#64748b",
                borderColor: "#334155",
                fontSize: "10px",
                py: 0.25,
                px: 1.5,
                textTransform: "none",
                "&.Mui-selected": {
                  bgcolor: "rgba(239, 68, 68, 0.15)",
                  color: "#ef4444",
                  borderColor: "#ef4444",
                },
              },
            }}
          >
            <ToggleButton value="trajectories">
              <RouteIcon sx={{ fontSize: 12, mr: 0.5 }} />
              Trajectories
            </ToggleButton>
            <ToggleButton value="displacement">
              <CompareArrowsIcon sx={{ fontSize: 12, mr: 0.5 }} />
              Displacement
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </div>

      {/* Content */}
      <Box className="flex-1 overflow-auto bg-slate-900 p-3">
        {sceneTab === "displacement" ? (
          <DisplacementInline model={model} />
        ) : (
        <>
        {/* Controls Row */}
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
          {/* Suite selector (model-aware) */}
          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
              Suite
            </InputLabel>
            <Select
              value={suite}
              label="Suite"
              onChange={(e) => setSuite(e.target.value)}
              sx={{
                color: "#e2e8f0",
                fontSize: "11px",
                "& .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#334155",
                },
                "&:hover .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#475569",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#ef4444",
                },
              }}
            >
              {availableModelSuites.map((s) => (
                <MenuItem key={s.value} value={s.value}>{s.label}</MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Pair selector */}
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
              Task Pair
            </InputLabel>
            <Select
              value={pairs.some((p) => p.id === selectedPair) ? selectedPair : ""}
              label="Task Pair"
              onChange={(e) => setSelectedPair(e.target.value)}
              disabled={loadingPairs || pairs.length === 0}
              sx={{
                color: "#e2e8f0",
                fontSize: "11px",
                "& .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#334155",
                },
                "&:hover .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#475569",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#ef4444",
                },
              }}
            >
              {pairs.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  <Box>
                    <Typography sx={{ fontSize: "11px" }}>
                      Pair {p.task_a}-{p.task_b} (T{p.task_a} vs T{p.task_b})
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Baseline condition */}
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
              Baseline
            </InputLabel>
            <Select
              value={baselineConditions.includes(baselineCondition) ? baselineCondition : ""}
              label="Baseline"
              onChange={(e) => setBaselineCondition(e.target.value)}
              disabled={baselineConditions.length === 0}
              sx={{
                color: "#e2e8f0",
                fontSize: "11px",
                "& .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#334155",
                },
                "&:hover .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#475569",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#ef4444",
                },
              }}
            >
              {baselineConditions.map((c, idx) => {
                const label = formatCondition(c);
                return (
                  <MenuItem key={c} value={c}>
                    {label || `Baseline ${idx === 0 ? "Task A" : "Task B"}`}
                  </MenuItem>
                );
              })}
            </Select>
          </FormControl>

          {/* Injection condition */}
          <FormControl size="small" sx={{ minWidth: 280 }}>
            <InputLabel sx={{ color: "#64748b", fontSize: "11px" }}>
              Injection Condition
            </InputLabel>
            <Select
              value={injectionConditions.includes(injectionCondition) ? injectionCondition : ""}
              label="Injection Condition"
              onChange={(e) => setInjectionCondition(e.target.value)}
              disabled={injectionConditions.length === 0}
              displayEmpty
              sx={{
                color: "#e2e8f0",
                fontSize: "11px",
                "& .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#334155",
                },
                "&:hover .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#475569",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "#ef4444",
                },
              }}
            >
              {injectionConditions.length === 0 && (
                <MenuItem value="" disabled>
                  <Typography sx={{ fontSize: "11px", color: "#475569" }}>
                    No injection data for this pair
                  </Typography>
                </MenuItem>
              )}
              {injectionConditions.map((c) => (
                <MenuItem key={c} value={c}>
                  <Typography sx={{ fontSize: "11px" }}>
                    {formatCondition(c)}
                  </Typography>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* View mode toggle */}
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, v) => v && setViewMode(v)}
            size="small"
            sx={{
              "& .MuiToggleButton-root": {
                color: "#64748b",
                borderColor: "#334155",
                fontSize: "10px",
                py: 0.5,
                px: 1.5,
                "&.Mui-selected": {
                  bgcolor: "#ef4444",
                  color: "white",
                  "&:hover": { bgcolor: "#dc2626" },
                },
              },
            }}
          >
            <ToggleButton value="side_by_side">Side by Side</ToggleButton>
            <ToggleButton value="overlay">Overlay</ToggleButton>
          </ToggleButtonGroup>

          {loading && (
            <CircularProgress size={16} sx={{ color: "#ef4444", ml: 1 }} />
          )}
        </Box>

        {/* Pair Info */}
        {currentPair && (
          <Box
            sx={{
              display: "flex",
              gap: 2,
              mb: 2,
              p: 1,
              bgcolor: "#1e293b",
              borderRadius: 1,
              border: "1px solid #334155",
            }}
          >
            <Box sx={{ flex: 1 }}>
              <Typography
                variant="caption"
                sx={{ color: "#3b82f6", fontWeight: 600, fontSize: "9px" }}
              >
                TASK A (T{currentPair.task_a})
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: "#e2e8f0", fontSize: "11px" }}
              >
                {currentPair.prompt_a}
              </Typography>
              {baselineScene && (
                <Box sx={{ display: "flex", gap: 1, mt: 0.5 }}>
                  <Chip
                    label={baselineScene.success ? "Success" : "Failure"}
                    size="small"
                    sx={{
                      height: 16,
                      fontSize: "8px",
                      bgcolor: baselineScene.success ? "#166534" : "#991b1b",
                      color: baselineScene.success ? "#86efac" : "#fca5a5",
                    }}
                  />
                  <Typography
                    variant="caption"
                    sx={{ color: "#475569", fontSize: "9px" }}
                  >
                    {baselineScene.steps} steps
                  </Typography>
                </Box>
              )}
            </Box>
            <Box
              sx={{
                width: 1,
                bgcolor: "#334155",
                mx: 1,
              }}
            />
            <Box sx={{ flex: 1 }}>
              <Typography
                variant="caption"
                sx={{ color: "#f97316", fontWeight: 600, fontSize: "9px" }}
              >
                TASK B (T{currentPair.task_b})
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: "#e2e8f0", fontSize: "11px" }}
              >
                {currentPair.prompt_b}
              </Typography>
              {injectionScene && (
                <Box sx={{ display: "flex", gap: 1, mt: 0.5 }}>
                  <Chip
                    label={injectionScene.success ? "Success" : "Failure"}
                    size="small"
                    sx={{
                      height: 16,
                      fontSize: "8px",
                      bgcolor: injectionScene.success ? "#166534" : "#991b1b",
                      color: injectionScene.success ? "#86efac" : "#fca5a5",
                    }}
                  />
                  <Typography
                    variant="caption"
                    sx={{ color: "#475569", fontSize: "9px" }}
                  >
                    {injectionScene.steps} steps |{" "}
                    {formatCondition(injectionCondition)}
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {error && (
          <Alert
            severity="error"
            sx={{
              mb: 2,
              bgcolor: "#7f1d1d",
              color: "#fca5a5",
              "& .MuiAlert-icon": { color: "#fca5a5" },
            }}
          >
            {error}
          </Alert>
        )}

        {/* Loading state */}
        {(loadingPairs || (loading && !baselineScene)) && !error && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              py: 8,
              gap: 2,
            }}
          >
            <CircularProgress size={32} sx={{ color: "#ef4444" }} />
            <Typography variant="caption" sx={{ color: "#64748b", fontSize: "11px" }}>
              {loadingPairs ? "Loading task pairs..." : "Loading scene data..."}
            </Typography>
          </Box>
        )}

        {/* No data state */}
        {!loadingPairs && pairs.length === 0 && !error && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              py: 8,
              gap: 1,
            }}
          >
            <Typography variant="body2" sx={{ color: "#64748b", fontSize: "12px" }}>
              No scene state data available.
            </Typography>
            <Typography variant="caption" sx={{ color: "#475569", fontSize: "10px" }}>
              {MODEL_DISPLAY_NAMES[model] || model} cross-task rollout data is not available for this suite.
            </Typography>
          </Box>
        )}

        {/* Trajectory Plots */}
        {!loading && !error && viewMode === "side_by_side" ? (
          <Box sx={{ display: "flex", gap: 2, mb: 2, flexWrap: "wrap" }}>
            <Box sx={{ flex: 1, minWidth: 400 }}>
              <TrajectoryPlot
                scene={baselineScene?.scene || null}
                label={`Baseline - ${formatCondition(baselineCondition)}`}
                color="#3b82f6"
                taskPrompt={baselineScene?.prompt_a || currentPair?.prompt_a}
                success={baselineScene?.success}
                stepCount={baselineScene?.steps}
              />
            </Box>
            <Box sx={{ flex: 1, minWidth: 400 }}>
              <TrajectoryPlot
                scene={injectionScene?.scene || null}
                label={`Injection - ${formatCondition(injectionCondition)}`}
                color="#ef4444"
                taskPrompt={injectionScene?.prompt_a || currentPair?.prompt_b}
                success={injectionScene?.success}
                stepCount={injectionScene?.steps}
              />
            </Box>
          </Box>
        ) : !loading && !error ? (
          <Box sx={{ mb: 2 }}>
            <TrajectoryPlot
              scene={baselineScene?.scene || null}
              label={`Baseline - ${formatCondition(baselineCondition)}`}
              color="#3b82f6"
              taskPrompt={baselineScene?.prompt_a || currentPair?.prompt_a}
              success={baselineScene?.success}
              stepCount={baselineScene?.steps}
              overlayScene={injectionScene?.scene || null}
              overlayColor="#ef4444"
              overlayLabel={formatCondition(injectionCondition)}
              overlaySuccess={injectionScene?.success}
            />
          </Box>
        ) : null}

        {/* Object Displacement Table */}
        {baselineScene?.scene?.object_displacements && (
          <Box sx={{ mb: 2 }}>
            <Typography
              variant="subtitle2"
              sx={{
                color: "#94a3b8",
                mb: 1,
                fontWeight: 600,
                fontSize: "11px",
                textTransform: "uppercase",
              }}
            >
              Object Displacements
            </Typography>
            <Box sx={{ display: "flex", gap: 2 }}>
              {/* Baseline table */}
              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: "#3b82f6",
                    fontWeight: 600,
                    fontSize: "9px",
                    mb: 0.5,
                    display: "block",
                  }}
                >
                  {formatCondition(baselineCondition)}
                </Typography>
                <TableContainer
                  sx={{
                    bgcolor: "#0f172a",
                    borderRadius: 1,
                    border: "1px solid #1e293b",
                  }}
                >
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell
                          sx={{
                            color: "#64748b",
                            fontSize: "9px",
                            py: 0.5,
                            borderColor: "#1e293b",
                          }}
                        >
                          Object
                        </TableCell>
                        <TableCell
                          sx={{
                            color: "#64748b",
                            fontSize: "9px",
                            py: 0.5,
                            borderColor: "#1e293b",
                          }}
                        >
                          Displacement
                        </TableCell>
                        <TableCell
                          sx={{
                            color: "#64748b",
                            fontSize: "9px",
                            py: 0.5,
                            borderColor: "#1e293b",
                          }}
                        >
                          Init Pos
                        </TableCell>
                        <TableCell
                          sx={{
                            color: "#64748b",
                            fontSize: "9px",
                            py: 0.5,
                            borderColor: "#1e293b",
                          }}
                        >
                          Final Pos
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(
                        baselineScene.scene.object_displacements
                      ).map(([name, disp]) => (
                        <TableRow key={name}>
                          <TableCell
                            sx={{
                              color: "#e2e8f0",
                              fontSize: "9px",
                              py: 0.3,
                              borderColor: "#1e293b",
                            }}
                          >
                            {name.replace(/_\d+$/, "").replace(/_/g, " ")}
                          </TableCell>
                          <TableCell
                            sx={{
                              color: displacementColor(disp.distance),
                              fontSize: "9px",
                              py: 0.3,
                              fontFamily: "monospace",
                              fontWeight: 600,
                              borderColor: "#1e293b",
                            }}
                          >
                            {disp.distance.toFixed(4)}
                          </TableCell>
                          <TableCell
                            sx={{
                              color: "#94a3b8",
                              fontSize: "8px",
                              py: 0.3,
                              fontFamily: "monospace",
                              borderColor: "#1e293b",
                            }}
                          >
                            {disp.init_pos?.map((v) => v.toFixed(2)).join(", ")}
                          </TableCell>
                          <TableCell
                            sx={{
                              color: "#94a3b8",
                              fontSize: "8px",
                              py: 0.3,
                              fontFamily: "monospace",
                              borderColor: "#1e293b",
                            }}
                          >
                            {disp.final_pos
                              ?.map((v) => v.toFixed(2))
                              .join(", ")}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>

              {/* Injection table */}
              {injectionScene?.scene?.object_displacements && (
                <Box sx={{ flex: 1 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "#ef4444",
                      fontWeight: 600,
                      fontSize: "9px",
                      mb: 0.5,
                      display: "block",
                    }}
                  >
                    {formatCondition(injectionCondition)}
                  </Typography>
                  <TableContainer
                    sx={{
                      bgcolor: "#0f172a",
                      borderRadius: 1,
                      border: "1px solid #1e293b",
                    }}
                  >
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell
                            sx={{
                              color: "#64748b",
                              fontSize: "9px",
                              py: 0.5,
                              borderColor: "#1e293b",
                            }}
                          >
                            Object
                          </TableCell>
                          <TableCell
                            sx={{
                              color: "#64748b",
                              fontSize: "9px",
                              py: 0.5,
                              borderColor: "#1e293b",
                            }}
                          >
                            Displacement
                          </TableCell>
                          <TableCell
                            sx={{
                              color: "#64748b",
                              fontSize: "9px",
                              py: 0.5,
                              borderColor: "#1e293b",
                            }}
                          >
                            Init Pos
                          </TableCell>
                          <TableCell
                            sx={{
                              color: "#64748b",
                              fontSize: "9px",
                              py: 0.5,
                              borderColor: "#1e293b",
                            }}
                          >
                            Final Pos
                          </TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(
                          injectionScene.scene.object_displacements
                        ).map(([name, disp]) => (
                          <TableRow key={name}>
                            <TableCell
                              sx={{
                                color: "#e2e8f0",
                                fontSize: "9px",
                                py: 0.3,
                                borderColor: "#1e293b",
                              }}
                            >
                              {name.replace(/_\d+$/, "").replace(/_/g, " ")}
                            </TableCell>
                            <TableCell
                              sx={{
                                color: displacementColor(disp.distance),
                                fontSize: "9px",
                                py: 0.3,
                                fontFamily: "monospace",
                                fontWeight: 600,
                                borderColor: "#1e293b",
                              }}
                            >
                              {disp.distance.toFixed(4)}
                            </TableCell>
                            <TableCell
                              sx={{
                                color: "#94a3b8",
                                fontSize: "8px",
                                py: 0.3,
                                fontFamily: "monospace",
                                borderColor: "#1e293b",
                              }}
                            >
                              {disp.init_pos
                                ?.map((v) => v.toFixed(2))
                                .join(", ")}
                            </TableCell>
                            <TableCell
                              sx={{
                                color: "#94a3b8",
                                fontSize: "8px",
                                py: 0.3,
                                fontFamily: "monospace",
                                borderColor: "#1e293b",
                              }}
                            >
                              {disp.final_pos
                                ?.map((v) => v.toFixed(2))
                                .join(", ")}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {/* Comparison Summary */}
        {baselineScene && injectionScene && (
          <Box
            sx={{
              p: 1.5,
              bgcolor: "#1e293b",
              borderRadius: 1,
              border: "1px solid #334155",
            }}
          >
            <Typography
              variant="subtitle2"
              sx={{ color: "#ef4444", fontWeight: 600, mb: 0.5, fontSize: "11px" }}
            >
              Comparison Summary
            </Typography>
            <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: "#64748b", fontSize: "9px" }}
                >
                  Steps
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: "#e2e8f0", fontSize: "12px", fontFamily: "monospace" }}
                >
                  {baselineScene.steps} vs {injectionScene.steps}{" "}
                  <span
                    style={{
                      color:
                        injectionScene.steps > baselineScene.steps
                          ? "#f59e0b"
                          : "#22c55e",
                      fontSize: "10px",
                    }}
                  >
                    ({injectionScene.steps > baselineScene.steps ? "+" : ""}
                    {injectionScene.steps - baselineScene.steps})
                  </span>
                </Typography>
              </Box>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: "#64748b", fontSize: "9px" }}
                >
                  Baseline Success
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: baselineScene.success ? "#22c55e" : "#ef4444",
                    fontSize: "12px",
                    fontWeight: 600,
                  }}
                >
                  {baselineScene.success ? "Yes" : "No"}
                </Typography>
              </Box>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: "#64748b", fontSize: "9px" }}
                >
                  Injection Success
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: injectionScene.success ? "#22c55e" : "#ef4444",
                    fontSize: "12px",
                    fontWeight: 600,
                  }}
                >
                  {injectionScene.success ? "Yes" : "No"}
                </Typography>
              </Box>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: "#64748b", fontSize: "9px" }}
                >
                  Injection Type
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: "#e2e8f0", fontSize: "11px" }}
                >
                  {formatCondition(injectionCondition)}
                </Typography>
              </Box>
            </Box>
          </Box>
        )}

        {/* Video comparison section - shows when injection condition has video */}
        {injectionScene?.video_url && (
          <Box sx={{ mt: 2, pt: 2, borderTop: "1px solid #1e293b" }}>
            <Typography
              variant="subtitle2"
              sx={{ color: "#94a3b8", fontSize: "11px", mb: 1, fontWeight: 600 }}
            >
              Video Comparison
            </Typography>
            <Box sx={{ display: "flex", gap: 1 }}>
              {/* Baseline video (own_prompt_no_inject in same injection direction) */}
              {baselineScene?.video_url && (
                <Box sx={{ flex: 1 }}>
                  <Typography
                    variant="caption"
                    sx={{ color: "#64748b", fontSize: "9px", mb: 0.5, display: "block" }}
                  >
                    Baseline ({formatCondition(baselineCondition)})
                  </Typography>
                  <video
                    src={`${API_BASE_URL}${baselineScene.video_url}`}
                    controls
                    loop
                    muted
                    style={{
                      width: "100%",
                      maxHeight: 200,
                      borderRadius: 4,
                      border: "1px solid #334155",
                      background: "#0f172a",
                    }}
                  />
                </Box>
              )}
              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "#64748b", fontSize: "9px", mb: 0.5, display: "block" }}
                >
                  Injection ({formatCondition(injectionCondition)})
                  <Chip
                    label={injectionScene.success ? "Success" : "Fail"}
                    size="small"
                    sx={{
                      ml: 0.5,
                      height: 14,
                      fontSize: "8px",
                      backgroundColor: injectionScene.success ? "#166534" : "#991b1b",
                      color: "#fff",
                    }}
                  />
                </Typography>
                <video
                  src={`${API_BASE_URL}${injectionScene.video_url}`}
                  controls
                  loop
                  muted
                  style={{
                    width: "100%",
                    maxHeight: 200,
                    borderRadius: 4,
                    border: "1px solid #334155",
                    background: "#0f172a",
                  }}
                />
              </Box>
            </Box>
          </Box>
        )}
        </>
        )}
      </Box>
    </Paper>
  );
}

// ---------------------------------------------------------------
// DisplacementInline - Displacement analysis rendered inside SceneState
// ---------------------------------------------------------------
function DisplacementInline({ model }: { model: string }) {
  const fullData = DISPLACEMENT_DATA[model] || { ...DEFAULT_DISPLACEMENT, modelName: MODEL_DISPLAY_NAMES[model] || model.toUpperCase() };
  const hasObjectDisp = !!fullData.objectDisplacement;
  const [metricView, setMetricView] = useState<"cosine" | "object">("cosine");
  const activeMetrics = metricView === "object" && hasObjectDisp ? fullData.objectDisplacement! : fullData;
  const displacementInfo = { ...fullData, ...activeMetrics };
  const hasData = model in DISPLACEMENT_DATA;

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
      <Typography variant="h5" sx={{ color, fontWeight: 700, fontSize: "1.5rem", lineHeight: 1.2 }}>
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

  return (
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
          {renderStatCard("Models", "6", "#3b82f6")}
          {renderStatCard("Episodes", "284,000+", "#ef4444")}
          {renderStatCard("SAEs Trained", "388", "#8b5cf6")}
          {renderStatCard("Concepts", "82+", "#10b981")}
          {renderStatCard("Benchmarks", "4", "#f59e0b", "LIBERO, MetaWorld, SimplerEnv, ALOHA")}
        </Box>
      </Box>

      <Divider sx={{ borderColor: "#1e293b" }} />

      {/* Current Model Displacement */}
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
          <Typography
            variant="caption"
            sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em" }}
          >
            {displacementInfo.modelName || MODEL_DISPLAY_NAMES[model] || model} Displacement Analysis
          </Typography>
          {hasObjectDisp && (
            <ToggleButtonGroup
              value={metricView}
              exclusive
              onChange={(_, v) => v && setMetricView(v)}
              size="small"
              sx={{
                "& .MuiToggleButton-root": {
                  color: "#64748b",
                  borderColor: "#334155",
                  fontSize: "9px",
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
              <ToggleButton value="cosine">Cosine Sim</ToggleButton>
              <ToggleButton value="object">Object Disp</ToggleButton>
            </ToggleButtonGroup>
          )}
        </Box>

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
              <Typography variant="h3" sx={{ color: "#ef4444", fontWeight: 800, fontSize: "2.5rem", lineHeight: 1 }}>
                {displacementInfo.sourceOverrideRate}%
              </Typography>
              <Typography variant="body2" sx={{ color: "#cbd5e1", fontSize: "0.8rem", mt: 1 }}>
                {displacementInfo.overrideRateLabel}
              </Typography>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.65rem", display: "block", mt: 0.5 }}>
                Based on {displacementInfo.episodeCount.toLocaleString()} episodes
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
                {displacementInfo.keyFinding}
              </Typography>
            </Box>

            {/* Detail Metrics */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {displacementInfo.details.map((detail, idx) => (
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

            {/* Cross-Model Override Comparison */}
            <Divider sx={{ borderColor: "#1e293b", my: 2 }} />
            <Typography
              variant="caption"
              sx={{ color: "#64748b", fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", mb: 1 }}
            >
              Cross-Model Override Comparison
            </Typography>
            {Object.entries(DISPLACEMENT_DATA).map(([modelId, modelData]) => {
              if (!modelData) return null;
              const isCurrent = modelId === model;
              return (
                <Box
                  key={modelId}
                  sx={{
                    bgcolor: isCurrent ? "rgba(239, 68, 68, 0.08)" : "rgba(15, 23, 42, 0.4)",
                    border: isCurrent ? "1px solid rgba(239, 68, 68, 0.3)" : "1px solid rgba(30, 41, 59, 0.5)",
                    borderRadius: 2,
                    p: 2,
                    mb: 1,
                  }}
                >
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Typography variant="body2" sx={{ color: "#f1f5f9", fontWeight: 600, fontSize: "0.8rem" }}>
                        {modelData.modelName}
                      </Typography>
                      {isCurrent && (
                        <Chip label="Selected" size="small" sx={{ height: 16, fontSize: "0.5rem", bgcolor: "#ef4444", color: "white" }} />
                      )}
                    </Box>
                    <Typography variant="body2" sx={{ color: "#ef4444", fontWeight: 700, fontSize: "0.9rem" }}>
                      {modelData.sourceOverrideRate}%
                    </Typography>
                  </Box>
                  <Box sx={{ width: "100%", height: 8, bgcolor: "rgba(15, 23, 42, 0.6)", borderRadius: 4, overflow: "hidden" }}>
                    <Box
                      sx={{
                        width: `${modelData.sourceOverrideRate}%`,
                        height: "100%",
                        bgcolor: isCurrent ? "#ef4444" : "#3b82f6",
                        borderRadius: 4,
                        transition: "width 0.5s ease",
                      }}
                    />
                  </Box>
                  <Typography variant="caption" sx={{ color: "#64748b", fontSize: "0.6rem", display: "block", mt: 0.5 }}>
                    {modelData.overrideRateLabel} ({modelData.episodeCount.toLocaleString()} episodes)
                  </Typography>
                </Box>
              );
            })}
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
              {displacementInfo.keyFinding}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
}
