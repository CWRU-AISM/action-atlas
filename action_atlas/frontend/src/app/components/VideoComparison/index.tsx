"use client";
import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Slider,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import SkipPreviousIcon from "@mui/icons-material/SkipPrevious";
import SkipNextIcon from "@mui/icons-material/SkipNext";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import FastForwardIcon from "@mui/icons-material/FastForward";
import CompareIcon from "@mui/icons-material/Compare";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import LayersIcon from "@mui/icons-material/Layers";
import DifferenceIcon from "@mui/icons-material/Difference";
import { API_BASE_URL } from "@/config/api";

interface VideoComparisonProps {
  baselineVideoPath: string;
  interventionVideoPath: string;
  baselineLabel?: string;
  interventionLabel?: string;
  baselineSuccess?: boolean;
  interventionSuccess?: boolean;
  title?: string;
  showDifferenceHighlight?: boolean;
  onFrameChange?: (frame: number) => void;
}

type ViewMode = "side-by-side" | "overlay" | "difference";

export default function VideoComparison({
  baselineVideoPath,
  interventionVideoPath,
  baselineLabel = "Baseline",
  interventionLabel = "Intervention",
  baselineSuccess,
  interventionSuccess,
  title,
  showDifferenceHighlight = false,
  onFrameChange,
}: VideoComparisonProps) {
  // Video refs
  const baselineVideoRef = useRef<HTMLVideoElement>(null);
  const interventionVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [overlayPosition, setOverlayPosition] = useState(50);
  const [showDifference, setShowDifference] = useState(showDifferenceHighlight);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Frame stepping (assuming 30fps standard)
  const FPS = 30;
  const frameStep = 1 / FPS;

  // Get video URL (handles both local paths and API paths)
  const getVideoUrl = (path: string): string => {
    if (path.startsWith("http") || path.startsWith("/")) {
      return path;
    }
    return `${API_BASE_URL}/api/vla/video/${path}`;
  };

  // Synchronize video playback
  const syncVideos = useCallback(() => {
    const baseline = baselineVideoRef.current;
    const intervention = interventionVideoRef.current;
    if (baseline && intervention) {
      intervention.currentTime = baseline.currentTime;
    }
  }, []);

  // Play both videos
  const playVideos = useCallback(() => {
    const baseline = baselineVideoRef.current;
    const intervention = interventionVideoRef.current;
    if (baseline && intervention) {
      baseline.play();
      intervention.play();
      setIsPlaying(true);
    }
  }, []);

  // Pause both videos
  const pauseVideos = useCallback(() => {
    const baseline = baselineVideoRef.current;
    const intervention = interventionVideoRef.current;
    if (baseline && intervention) {
      baseline.pause();
      intervention.pause();
      setIsPlaying(false);
    }
  }, []);

  // Toggle play/pause
  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      pauseVideos();
    } else {
      playVideos();
    }
  }, [isPlaying, playVideos, pauseVideos]);

  // Seek to specific time
  const seekTo = useCallback(
    (time: number) => {
      const baseline = baselineVideoRef.current;
      const intervention = interventionVideoRef.current;
      if (baseline && intervention) {
        const clampedTime = Math.max(0, Math.min(time, duration));
        baseline.currentTime = clampedTime;
        intervention.currentTime = clampedTime;
        setCurrentTime(clampedTime);
      }
    },
    [duration]
  );

  // Step forward one frame
  const stepForward = useCallback(() => {
    pauseVideos();
    seekTo(currentTime + frameStep);
  }, [currentTime, frameStep, pauseVideos, seekTo]);

  // Step backward one frame
  const stepBackward = useCallback(() => {
    pauseVideos();
    seekTo(currentTime - frameStep);
  }, [currentTime, frameStep, pauseVideos, seekTo]);

  // Jump forward 5 seconds
  const jumpForward = useCallback(() => {
    seekTo(currentTime + 5);
  }, [currentTime, seekTo]);

  // Jump backward 5 seconds
  const jumpBackward = useCallback(() => {
    seekTo(currentTime - 5);
  }, [currentTime, seekTo]);

  // Handle timeline scrubber change
  const handleTimelineChange = useCallback(
    (_: Event, value: number | number[]) => {
      const newTime = value as number;
      seekTo(newTime);
    },
    [seekTo]
  );

  // Handle overlay slider change
  const handleOverlayChange = useCallback(
    (_: Event, value: number | number[]) => {
      setOverlayPosition(value as number);
    },
    []
  );

  // Change playback speed
  const handleSpeedChange = useCallback((speed: number) => {
    const baseline = baselineVideoRef.current;
    const intervention = interventionVideoRef.current;
    if (baseline && intervention) {
      baseline.playbackRate = speed;
      intervention.playbackRate = speed;
      setPlaybackSpeed(speed);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if focus is on an input element
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlayPause();
          break;
        case "ArrowLeft":
          e.preventDefault();
          if (e.shiftKey) {
            jumpBackward();
          } else {
            stepBackward();
          }
          break;
        case "ArrowRight":
          e.preventDefault();
          if (e.shiftKey) {
            jumpForward();
          } else {
            stepForward();
          }
          break;
        case "0":
        case "Home":
          e.preventDefault();
          seekTo(0);
          break;
        case "End":
          e.preventDefault();
          seekTo(duration);
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    togglePlayPause,
    stepForward,
    stepBackward,
    jumpForward,
    jumpBackward,
    seekTo,
    duration,
  ]);

  // Update time display and sync videos
  useEffect(() => {
    const baseline = baselineVideoRef.current;
    if (!baseline) return;

    const handleTimeUpdate = () => {
      setCurrentTime(baseline.currentTime);
      syncVideos();
      if (onFrameChange) {
        onFrameChange(Math.floor(baseline.currentTime * FPS));
      }
    };

    const handleLoadedMetadata = () => {
      setDuration(baseline.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    baseline.addEventListener("timeupdate", handleTimeUpdate);
    baseline.addEventListener("loadedmetadata", handleLoadedMetadata);
    baseline.addEventListener("ended", handleEnded);

    return () => {
      baseline.removeEventListener("timeupdate", handleTimeUpdate);
      baseline.removeEventListener("loadedmetadata", handleLoadedMetadata);
      baseline.removeEventListener("ended", handleEnded);
    };
  }, [syncVideos, onFrameChange, FPS]);

  // Compute difference between frames (for difference mode)
  useEffect(() => {
    if (viewMode !== "difference" || !showDifference) return;

    const canvas = canvasRef.current;
    const baseline = baselineVideoRef.current;
    const intervention = interventionVideoRef.current;

    if (!canvas || !baseline || !intervention) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const computeDifference = () => {
      // Only compute if both videos are loaded
      if (baseline.readyState < 2 || intervention.readyState < 2) return;

      const width = canvas.width;
      const height = canvas.height;

      // Draw baseline frame
      ctx.drawImage(baseline, 0, 0, width, height);
      const baselineData = ctx.getImageData(0, 0, width, height);

      // Draw intervention frame
      ctx.drawImage(intervention, 0, 0, width, height);
      const interventionData = ctx.getImageData(0, 0, width, height);

      // Compute difference
      const diffData = ctx.createImageData(width, height);
      for (let i = 0; i < baselineData.data.length; i += 4) {
        const diffR = Math.abs(baselineData.data[i] - interventionData.data[i]);
        const diffG = Math.abs(
          baselineData.data[i + 1] - interventionData.data[i + 1]
        );
        const diffB = Math.abs(
          baselineData.data[i + 2] - interventionData.data[i + 2]
        );

        // Highlight differences in red
        const threshold = 30;
        if (diffR > threshold || diffG > threshold || diffB > threshold) {
          diffData.data[i] = 239; // Red (#ef4444)
          diffData.data[i + 1] = 68;
          diffData.data[i + 2] = 68;
          diffData.data[i + 3] = 200;
        } else {
          // Show grayscale baseline
          const gray =
            (baselineData.data[i] +
              baselineData.data[i + 1] +
              baselineData.data[i + 2]) /
            3;
          diffData.data[i] = gray;
          diffData.data[i + 1] = gray;
          diffData.data[i + 2] = gray;
          diffData.data[i + 3] = 255;
        }
      }

      ctx.putImageData(diffData, 0, 0);
    };

    const interval = setInterval(computeDifference, 100);
    return () => clearInterval(interval);
  }, [viewMode, showDifference]);

  // Format time as MM:SS.ms
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
  };

  // Current frame number
  const currentFrame = Math.floor(currentTime * FPS);
  const totalFrames = Math.floor(duration * FPS);

  return (
    <Paper
      sx={{
        backgroundColor: "#0a1628",
        borderRadius: 2,
        overflow: "hidden",
        border: "1px solid #1e3a5f",
      }}
    >
      {/* Header */}
      {title && (
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderBottom: "1px solid #1e3a5f",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <CompareIcon sx={{ color: "#ef4444", fontSize: 20 }} />
            <Typography
              variant="subtitle1"
              sx={{ color: "white", fontWeight: 600 }}
            >
              {title}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography
              variant="caption"
              sx={{ color: "#94a3b8", fontSize: 10 }}
            >
              Frame {currentFrame} / {totalFrames}
            </Typography>
          </Box>
        </Box>
      )}

      {/* View Mode Toggle */}
      <Box
        sx={{
          px: 2,
          py: 1,
          borderBottom: "1px solid #1e3a5f",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, value) => value && setViewMode(value)}
          size="small"
          sx={{
            "& .MuiToggleButton-root": {
              color: "#94a3b8",
              borderColor: "#334155",
              fontSize: "11px",
              px: 1.5,
              py: 0.5,
              "&.Mui-selected": {
                backgroundColor: "#ef4444",
                color: "white",
                "&:hover": { backgroundColor: "#dc2626" },
              },
              "&:hover": { backgroundColor: "#334155" },
            },
          }}
        >
          <ToggleButton value="side-by-side">
            <ViewColumnIcon sx={{ fontSize: 16, mr: 0.5 }} />
            Side-by-Side
          </ToggleButton>
          <ToggleButton value="overlay">
            <LayersIcon sx={{ fontSize: 16, mr: 0.5 }} />
            Overlay
          </ToggleButton>
          <ToggleButton value="difference">
            <DifferenceIcon sx={{ fontSize: 16, mr: 0.5 }} />
            Difference
          </ToggleButton>
        </ToggleButtonGroup>

        {/* Speed Control */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="caption" sx={{ color: "#94a3b8" }}>
            Speed:
          </Typography>
          {[0.25, 0.5, 1, 2].map((speed) => (
            <Chip
              key={speed}
              label={`${speed}x`}
              size="small"
              onClick={() => handleSpeedChange(speed)}
              sx={{
                fontSize: "10px",
                height: 20,
                backgroundColor:
                  playbackSpeed === speed ? "#ef4444" : "transparent",
                color: playbackSpeed === speed ? "white" : "#94a3b8",
                border: "1px solid #334155",
                cursor: "pointer",
                "&:hover": {
                  backgroundColor: playbackSpeed === speed ? "#dc2626" : "#1e3a5f",
                },
              }}
            />
          ))}
        </Box>
      </Box>

      {/* Video Container */}
      <Box sx={{ position: "relative", backgroundColor: "#000" }}>
        {/* Side-by-Side View */}
        {viewMode === "side-by-side" && (
          <Box sx={{ display: "flex" }}>
            {/* Baseline Video */}
            <Box sx={{ flex: 1, position: "relative" }}>
              <Box
                sx={{
                  position: "absolute",
                  top: 8,
                  left: 8,
                  zIndex: 10,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <Chip
                  label={baselineLabel}
                  size="small"
                  sx={{
                    fontSize: "10px",
                    height: 20,
                    backgroundColor: "rgba(59, 130, 246, 0.9)",
                    color: "white",
                  }}
                />
                {baselineSuccess !== undefined && (
                  <Chip
                    label={baselineSuccess ? "Success" : "Failure"}
                    size="small"
                    sx={{
                      fontSize: "10px",
                      height: 20,
                      backgroundColor: baselineSuccess
                        ? "rgba(34, 197, 94, 0.9)"
                        : "rgba(239, 68, 68, 0.9)",
                      color: "white",
                    }}
                  />
                )}
              </Box>
              <video
                ref={baselineVideoRef}
                src={getVideoUrl(baselineVideoPath)}
                muted
                playsInline
                style={{
                  width: "100%",
                  height: "auto",
                  maxHeight: "300px",
                  objectFit: "contain",
                }}
              />
            </Box>

            {/* Divider */}
            <Box sx={{ width: 2, backgroundColor: "#ef4444" }} />

            {/* Intervention Video */}
            <Box sx={{ flex: 1, position: "relative" }}>
              <Box
                sx={{
                  position: "absolute",
                  top: 8,
                  left: 8,
                  zIndex: 10,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <Chip
                  label={interventionLabel}
                  size="small"
                  sx={{
                    fontSize: "10px",
                    height: 20,
                    backgroundColor: "rgba(239, 68, 68, 0.9)",
                    color: "white",
                  }}
                />
                {interventionSuccess !== undefined && (
                  <Chip
                    label={interventionSuccess ? "Success" : "Failure"}
                    size="small"
                    sx={{
                      fontSize: "10px",
                      height: 20,
                      backgroundColor: interventionSuccess
                        ? "rgba(34, 197, 94, 0.9)"
                        : "rgba(239, 68, 68, 0.9)",
                      color: "white",
                    }}
                  />
                )}
              </Box>
              <video
                ref={interventionVideoRef}
                src={getVideoUrl(interventionVideoPath)}
                muted
                playsInline
                style={{
                  width: "100%",
                  height: "auto",
                  maxHeight: "300px",
                  objectFit: "contain",
                }}
              />
            </Box>
          </Box>
        )}

        {/* Overlay View */}
        {viewMode === "overlay" && (
          <Box sx={{ position: "relative", overflow: "hidden" }}>
            {/* Labels */}
            <Box
              sx={{
                position: "absolute",
                top: 8,
                left: 8,
                zIndex: 20,
                display: "flex",
                gap: 1,
              }}
            >
              <Chip
                label={`${baselineLabel} (Left)`}
                size="small"
                sx={{
                  fontSize: "10px",
                  height: 20,
                  backgroundColor: "rgba(59, 130, 246, 0.9)",
                  color: "white",
                }}
              />
            </Box>
            <Box
              sx={{
                position: "absolute",
                top: 8,
                right: 8,
                zIndex: 20,
                display: "flex",
                gap: 1,
              }}
            >
              <Chip
                label={`${interventionLabel} (Right)`}
                size="small"
                sx={{
                  fontSize: "10px",
                  height: 20,
                  backgroundColor: "rgba(239, 68, 68, 0.9)",
                  color: "white",
                }}
              />
            </Box>

            {/* Baseline Video (full width, behind) */}
            <video
              ref={baselineVideoRef}
              src={getVideoUrl(baselineVideoPath)}
              muted
              playsInline
              style={{
                width: "100%",
                height: "auto",
                maxHeight: "300px",
                objectFit: "contain",
              }}
            />

            {/* Intervention Video (clipped by overlay position) */}
            <Box
              sx={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                overflow: "hidden",
                clipPath: `inset(0 ${100 - overlayPosition}% 0 0)`,
              }}
            >
              <video
                ref={interventionVideoRef}
                src={getVideoUrl(interventionVideoPath)}
                muted
                playsInline
                style={{
                  width: "100%",
                  height: "auto",
                  maxHeight: "300px",
                  objectFit: "contain",
                }}
              />
            </Box>

            {/* Slider Handle */}
            <Box
              sx={{
                position: "absolute",
                top: 0,
                bottom: 0,
                left: `${overlayPosition}%`,
                width: 4,
                backgroundColor: "#ef4444",
                cursor: "ew-resize",
                zIndex: 15,
                "&::before": {
                  content: '""',
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  width: 24,
                  height: 24,
                  borderRadius: "50%",
                  backgroundColor: "#ef4444",
                  border: "2px solid white",
                },
              }}
            />

            {/* Overlay Slider (invisible, for interaction) */}
            <Box
              sx={{
                position: "absolute",
                bottom: 60,
                left: "10%",
                right: "10%",
                zIndex: 20,
              }}
            >
              <Slider
                value={overlayPosition}
                onChange={handleOverlayChange}
                min={0}
                max={100}
                sx={{
                  color: "#ef4444",
                  "& .MuiSlider-thumb": {
                    width: 16,
                    height: 16,
                  },
                  "& .MuiSlider-track": {
                    height: 4,
                  },
                  "& .MuiSlider-rail": {
                    height: 4,
                    opacity: 0.5,
                  },
                }}
              />
            </Box>
          </Box>
        )}

        {/* Difference View */}
        {viewMode === "difference" && (
          <Box sx={{ position: "relative" }}>
            <Box
              sx={{
                position: "absolute",
                top: 8,
                left: 8,
                zIndex: 10,
              }}
            >
              <Chip
                label="Difference Highlight"
                size="small"
                icon={<DifferenceIcon sx={{ fontSize: 14, color: "white" }} />}
                sx={{
                  fontSize: "10px",
                  height: 20,
                  backgroundColor: "rgba(239, 68, 68, 0.9)",
                  color: "white",
                  "& .MuiChip-icon": { color: "white" },
                }}
              />
            </Box>

            {/* Hidden videos for difference computation */}
            <video
              ref={baselineVideoRef}
              src={getVideoUrl(baselineVideoPath)}
              muted
              playsInline
              style={{
                width: "100%",
                height: "auto",
                maxHeight: "300px",
                objectFit: "contain",
                display: showDifference ? "none" : "block",
              }}
            />
            <video
              ref={interventionVideoRef}
              src={getVideoUrl(interventionVideoPath)}
              muted
              playsInline
              style={{ display: "none" }}
            />

            {/* Difference Canvas */}
            {showDifference && (
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                style={{
                  width: "100%",
                  height: "auto",
                  maxHeight: "300px",
                  objectFit: "contain",
                }}
              />
            )}

            {/* Toggle Difference Highlight */}
            <Box
              sx={{
                position: "absolute",
                top: 8,
                right: 8,
                zIndex: 10,
              }}
            >
              <Chip
                label={showDifference ? "Show Original" : "Show Difference"}
                size="small"
                onClick={() => setShowDifference(!showDifference)}
                sx={{
                  fontSize: "10px",
                  height: 20,
                  backgroundColor: "#334155",
                  color: "white",
                  cursor: "pointer",
                  "&:hover": { backgroundColor: "#475569" },
                }}
              />
            </Box>
          </Box>
        )}
      </Box>

      {/* Controls */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderTop: "1px solid #1e3a5f",
          backgroundColor: "#0f1d32",
        }}
      >
        {/* Timeline Scrubber */}
        <Box sx={{ mb: 1.5 }}>
          <Slider
            value={currentTime}
            onChange={handleTimelineChange}
            min={0}
            max={duration || 100}
            step={0.01}
            sx={{
              color: "#ef4444",
              height: 6,
              "& .MuiSlider-thumb": {
                width: 14,
                height: 14,
                "&:hover, &.Mui-focusVisible": {
                  boxShadow: "0 0 0 8px rgba(239, 68, 68, 0.16)",
                },
              },
              "& .MuiSlider-track": {
                height: 6,
                borderRadius: 3,
              },
              "& .MuiSlider-rail": {
                height: 6,
                borderRadius: 3,
                opacity: 0.3,
                backgroundColor: "#475569",
              },
            }}
          />
        </Box>

        {/* Playback Controls */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            {/* Jump backward */}
            <Tooltip title="Jump backward 5s (Shift+Left)">
              <IconButton
                onClick={jumpBackward}
                size="small"
                sx={{ color: "#94a3b8", "&:hover": { color: "white" } }}
              >
                <FastRewindIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            {/* Step backward */}
            <Tooltip title="Previous frame (Left Arrow)">
              <IconButton
                onClick={stepBackward}
                size="small"
                sx={{ color: "#94a3b8", "&:hover": { color: "white" } }}
              >
                <SkipPreviousIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            {/* Play/Pause */}
            <Tooltip title={isPlaying ? "Pause (Space)" : "Play (Space)"}>
              <IconButton
                onClick={togglePlayPause}
                sx={{
                  color: "white",
                  backgroundColor: "#ef4444",
                  "&:hover": { backgroundColor: "#dc2626" },
                  mx: 1,
                }}
              >
                {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
              </IconButton>
            </Tooltip>

            {/* Step forward */}
            <Tooltip title="Next frame (Right Arrow)">
              <IconButton
                onClick={stepForward}
                size="small"
                sx={{ color: "#94a3b8", "&:hover": { color: "white" } }}
              >
                <SkipNextIcon fontSize="small" />
              </IconButton>
            </Tooltip>

            {/* Jump forward */}
            <Tooltip title="Jump forward 5s (Shift+Right)">
              <IconButton
                onClick={jumpForward}
                size="small"
                sx={{ color: "#94a3b8", "&:hover": { color: "white" } }}
              >
                <FastForwardIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Time Display */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Typography
              variant="caption"
              sx={{ color: "#94a3b8", fontFamily: "monospace" }}
            >
              {formatTime(currentTime)} / {formatTime(duration)}
            </Typography>
            <Chip
              label={`Frame ${currentFrame}`}
              size="small"
              sx={{
                fontSize: "10px",
                height: 20,
                backgroundColor: "#1e3a5f",
                color: "#94a3b8",
              }}
            />
          </Box>
        </Box>

        {/* Keyboard Shortcuts Help */}
        <Box
          sx={{
            mt: 1.5,
            pt: 1,
            borderTop: "1px solid #1e3a5f",
            display: "flex",
            justifyContent: "center",
            gap: 3,
          }}
        >
          <Typography variant="caption" sx={{ color: "#64748b", fontSize: 9 }}>
            <kbd
              style={{
                backgroundColor: "#1e3a5f",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              Space
            </kbd>{" "}
            Play/Pause
          </Typography>
          <Typography variant="caption" sx={{ color: "#64748b", fontSize: 9 }}>
            <kbd
              style={{
                backgroundColor: "#1e3a5f",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              ← →
            </kbd>{" "}
            Frame Step
          </Typography>
          <Typography variant="caption" sx={{ color: "#64748b", fontSize: 9 }}>
            <kbd
              style={{
                backgroundColor: "#1e3a5f",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              Shift + ← →
            </kbd>{" "}
            Jump 5s
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
}
