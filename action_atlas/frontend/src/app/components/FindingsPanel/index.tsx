"use client";
import React, { useState, useEffect } from "react";
import {
  Paper,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Chip,
  Tooltip,
  Divider,
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import LightbulbIcon from "@mui/icons-material/Lightbulb";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import TimelineIcon from "@mui/icons-material/Timeline";
import ScienceIcon from "@mui/icons-material/Science";
import TuneIcon from "@mui/icons-material/Tune";
import VerifiedIcon from "@mui/icons-material/Verified";
import WarningIcon from "@mui/icons-material/Warning";
import FileDownloadIcon from "@mui/icons-material/FileDownload";
import DataObjectIcon from "@mui/icons-material/DataObject";
import DescriptionIcon from "@mui/icons-material/Description";
import { API_BASE_URL } from "@/config/api";
import { useAppSelector } from "@/redux/hooks";

// Types for the API response
interface Finding {
  id: string;
  title: string;
  description: string;
  evidence: string;
  confidence: "high" | "medium" | "low";
  category: "interpretability" | "architecture" | "causal" | "temporal" | "control";
}

interface Metrics {
  total_features_analyzed?: number;
  layers_covered?: number;
  tasks_tested?: number;
  avg_reconstruction_accuracy?: number;
  [key: string]: number | string | undefined;
}

interface FindingsData {
  model: string;
  key_findings: Finding[];
  metrics: Metrics;
  limitations: string[];
}

interface FindingsResponse {
  data: FindingsData;
}

// Category configuration
const CATEGORY_CONFIG: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  interpretability: {
    color: "#3b82f6",
    icon: <LightbulbIcon fontSize="small" />,
    label: "Interpretability",
  },
  architecture: {
    color: "#8b5cf6",
    icon: <AccountTreeIcon fontSize="small" />,
    label: "Architecture",
  },
  causal: {
    color: "#ef4444",
    icon: <ScienceIcon fontSize="small" />,
    label: "Causal",
  },
  temporal: {
    color: "#f59e0b",
    icon: <TimelineIcon fontSize="small" />,
    label: "Temporal",
  },
  control: {
    color: "#22c55e",
    icon: <TuneIcon fontSize="small" />,
    label: "Control",
  },
};

// Confidence color mapping
const CONFIDENCE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  high: { bg: "rgba(34, 197, 94, 0.1)", text: "#22c55e", border: "#22c55e" },
  medium: { bg: "rgba(245, 158, 11, 0.1)", text: "#f59e0b", border: "#f59e0b" },
  low: { bg: "rgba(239, 68, 68, 0.1)", text: "#ef4444", border: "#ef4444" },
};

export default function FindingsPanel() {
  const [findings, setFindings] = useState<FindingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(["interpretability"]));
  const [exportMenuAnchor, setExportMenuAnchor] = useState<null | HTMLElement>(null);
  const currentModel = useAppSelector((state) => state.model.currentModel);

  useEffect(() => {
    fetchFindings();
  }, [currentModel]);

  const fetchFindings = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/vla/findings?model=${currentModel}`);
      if (!response.ok) {
        throw new Error("Failed to fetch findings");
      }
      const data: FindingsResponse = await response.json();
      setFindings(data.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load findings");
    } finally {
      setLoading(false);
    }
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  const handleExportMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };

  const handleExportMenuClose = () => {
    setExportMenuAnchor(null);
  };

  const downloadFile = (content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportAsJSON = () => {
    if (!findings) return;
    const jsonContent = JSON.stringify(findings, null, 2);
    const timestamp = new Date().toISOString().split("T")[0];
    downloadFile(jsonContent, `research-findings-${timestamp}.json`, "application/json");
    handleExportMenuClose();
  };

  const exportAsMarkdown = () => {
    if (!findings) return;

    const timestamp = new Date().toISOString().split("T")[0];
    let markdown = `# Research Findings Report\n\n`;
    markdown += `**Model:** ${findings.model}\n`;
    markdown += `**Export Date:** ${timestamp}\n\n`;
    markdown += `---\n\n`;

    // Metrics section
    if (findings.metrics) {
      markdown += `## Study Metrics\n\n`;
      markdown += `| Metric | Value |\n`;
      markdown += `|--------|-------|\n`;
      if (findings.metrics.total_features_analyzed !== undefined) {
        markdown += `| Features Analyzed | ${findings.metrics.total_features_analyzed.toLocaleString()} |\n`;
      }
      if (findings.metrics.layers_covered !== undefined) {
        markdown += `| Layers Covered | ${findings.metrics.layers_covered} |\n`;
      }
      if (findings.metrics.tasks_tested !== undefined) {
        markdown += `| Tasks Tested | ${findings.metrics.tasks_tested} |\n`;
      }
      if (findings.metrics.avg_reconstruction_accuracy !== undefined) {
        markdown += `| Avg Reconstruction Accuracy | ${findings.metrics.avg_reconstruction_accuracy}% |\n`;
      }
      markdown += `\n`;
    }

    // Key findings by category
    markdown += `## Key Findings\n\n`;

    const groupedByCategory = findings.key_findings.reduce((acc, finding) => {
      if (!acc[finding.category]) {
        acc[finding.category] = [];
      }
      acc[finding.category].push(finding);
      return acc;
    }, {} as Record<string, Finding[]>);

    Object.entries(CATEGORY_CONFIG).forEach(([category, config]) => {
      const categoryFindings = groupedByCategory[category];
      if (!categoryFindings || categoryFindings.length === 0) return;

      markdown += `### ${config.label}\n\n`;

      categoryFindings.forEach((finding, idx) => {
        markdown += `#### ${idx + 1}. ${finding.title}\n\n`;
        markdown += `**Confidence:** ${finding.confidence.charAt(0).toUpperCase() + finding.confidence.slice(1)}\n\n`;
        markdown += `${finding.description}\n\n`;
        markdown += `> **Evidence:** ${finding.evidence}\n\n`;
      });
    });

    // Limitations section
    if (findings.limitations && findings.limitations.length > 0) {
      markdown += `## Limitations\n\n`;
      findings.limitations.forEach((limitation, idx) => {
        markdown += `${idx + 1}. ${limitation}\n`;
      });
      markdown += `\n`;
    }

    markdown += `---\n\n`;
    markdown += `*Generated from Action Atlas*\n`;

    downloadFile(markdown, `research-findings-${timestamp}.md`, "text/markdown");
    handleExportMenuClose();
  };

  // Group findings by category
  const groupedFindings = findings?.key_findings.reduce((acc, finding) => {
    if (!acc[finding.category]) {
      acc[finding.category] = [];
    }
    acc[finding.category].push(finding);
    return acc;
  }, {} as Record<string, Finding[]>) || {};

  const renderMetricCard = (label: string, value: number | string | undefined, unit?: string) => {
    if (value === undefined) return null;
    return (
      <Box
        sx={{
          bgcolor: "rgba(15, 23, 42, 0.6)",
          borderRadius: 1,
          p: 1,
          textAlign: "center",
          minWidth: 80,
        }}
      >
        <Typography
          variant="h6"
          sx={{ color: "#ef4444", fontWeight: 700, fontSize: "1rem", lineHeight: 1.2 }}
        >
          {typeof value === "number" ? value.toLocaleString() : value}
          {unit && <span style={{ fontSize: "0.7rem", marginLeft: 2 }}>{unit}</span>}
        </Typography>
        <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: "0.6rem" }}>
          {label}
        </Typography>
      </Box>
    );
  };

  const renderFindingCard = (finding: Finding) => {
    const confidenceStyle = CONFIDENCE_COLORS[finding.confidence];
    const categoryConfig = CATEGORY_CONFIG[finding.category];

    return (
      <Accordion
        key={finding.id}
        sx={{
          bgcolor: "#1e293b",
          color: "#e2e8f0",
          "&:before": { display: "none" },
          boxShadow: "none",
          borderRadius: "6px !important",
          mb: 1,
          "&.Mui-expanded": {
            margin: "0 0 8px 0",
          },
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon sx={{ color: "#94a3b8" }} />}
          sx={{
            minHeight: 44,
            "&.Mui-expanded": { minHeight: 44 },
            "& .MuiAccordionSummary-content": {
              margin: "8px 0",
              "&.Mui-expanded": { margin: "8px 0" },
            },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: "100%" }}>
            <Tooltip title={`Confidence: ${finding.confidence}`} arrow>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  bgcolor: confidenceStyle.text,
                  flexShrink: 0,
                }}
              />
            </Tooltip>
            <Typography
              variant="body2"
              sx={{
                fontWeight: 600,
                fontSize: "0.75rem",
                color: "#f1f5f9",
                flex: 1,
                pr: 1,
              }}
            >
              {finding.title}
            </Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0, pb: 1.5 }}>
          <Typography
            variant="body2"
            sx={{ color: "#cbd5e1", fontSize: "0.7rem", mb: 1.5, lineHeight: 1.5 }}
          >
            {finding.description}
          </Typography>
          <Box
            sx={{
              bgcolor: "rgba(0, 0, 0, 0.2)",
              borderRadius: 1,
              p: 1.5,
              borderLeft: `3px solid ${categoryConfig?.color || "#64748b"}`,
            }}
          >
            <Typography
              variant="caption"
              sx={{
                color: "#94a3b8",
                display: "flex",
                alignItems: "center",
                gap: 0.5,
                mb: 0.5,
                fontSize: "0.6rem",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              <VerifiedIcon sx={{ fontSize: 12 }} />
              Evidence
            </Typography>
            <Typography
              variant="body2"
              sx={{ color: "#e2e8f0", fontSize: "0.65rem", lineHeight: 1.5, fontStyle: "italic" }}
            >
              {finding.evidence}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", gap: 1, mt: 1.5, flexWrap: "wrap" }}>
            <Chip
              label={finding.confidence}
              size="small"
              sx={{
                height: 18,
                fontSize: "0.55rem",
                bgcolor: confidenceStyle.bg,
                color: confidenceStyle.text,
                border: `1px solid ${confidenceStyle.border}`,
                textTransform: "capitalize",
              }}
            />
            <Chip
              label={categoryConfig?.label || finding.category}
              size="small"
              icon={categoryConfig?.icon as React.ReactElement}
              sx={{
                height: 18,
                fontSize: "0.55rem",
                bgcolor: `${categoryConfig?.color}20`,
                color: categoryConfig?.color,
                border: `1px solid ${categoryConfig?.color}`,
                "& .MuiChip-icon": {
                  color: categoryConfig?.color,
                  fontSize: 10,
                  marginLeft: "4px",
                },
              }}
            />
          </Box>
        </AccordionDetails>
      </Accordion>
    );
  };

  if (loading) {
    return (
      <Paper className="h-full flex flex-col overflow-hidden rounded-lg shadow-md">
        <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg flex-shrink-0">
          <span className="text-xs font-semibold text-white">Research Findings</span>
        </div>
        <Box
          sx={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            bgcolor: "#0f172a",
          }}
        >
          <CircularProgress size={24} sx={{ color: "#ef4444" }} />
        </Box>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper className="h-full flex flex-col overflow-hidden rounded-lg shadow-md">
        <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg flex-shrink-0">
          <span className="text-xs font-semibold text-white">Research Findings</span>
        </div>
        <Box
          sx={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            bgcolor: "#0f172a",
            p: 2,
          }}
        >
          <Typography variant="body2" sx={{ color: "#ef4444", textAlign: "center" }}>
            {error}
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper className="h-full flex flex-col overflow-hidden rounded-lg shadow-md">
      {/* Dark Navy Header */}
      <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg flex-shrink-0">
        <LightbulbIcon sx={{ fontSize: 14, color: "#f59e0b", mr: 1 }} />
        <span className="text-xs font-semibold text-white">Research Findings</span>
        <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 1 }}>
          {findings && (
            <>
              <Chip
                label={findings.model.toUpperCase()}
                size="small"
                sx={{
                  height: 16,
                  fontSize: "0.55rem",
                  bgcolor: "#ef4444",
                  color: "white",
                }}
              />
              <Tooltip title="Export findings" arrow>
                <IconButton
                  size="small"
                  onClick={handleExportMenuOpen}
                  sx={{
                    p: 0.5,
                    color: "#94a3b8",
                    "&:hover": {
                      bgcolor: "rgba(255, 255, 255, 0.1)",
                      color: "#f1f5f9",
                    },
                  }}
                >
                  <FileDownloadIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
              <Menu
                anchorEl={exportMenuAnchor}
                open={Boolean(exportMenuAnchor)}
                onClose={handleExportMenuClose}
                anchorOrigin={{
                  vertical: "bottom",
                  horizontal: "right",
                }}
                transformOrigin={{
                  vertical: "top",
                  horizontal: "right",
                }}
                PaperProps={{
                  sx: {
                    bgcolor: "#1e293b",
                    border: "1px solid #334155",
                    minWidth: 180,
                    "& .MuiMenuItem-root": {
                      fontSize: "0.75rem",
                      py: 1,
                      color: "#e2e8f0",
                      "&:hover": {
                        bgcolor: "rgba(255, 255, 255, 0.08)",
                      },
                    },
                  },
                }}
              >
                <MenuItem onClick={exportAsJSON}>
                  <ListItemIcon>
                    <DataObjectIcon sx={{ fontSize: 18, color: "#3b82f6" }} />
                  </ListItemIcon>
                  <ListItemText
                    primary="Export as JSON"
                    secondary="Raw data format"
                    primaryTypographyProps={{ fontSize: "0.75rem", color: "#e2e8f0" }}
                    secondaryTypographyProps={{ fontSize: "0.6rem", color: "#64748b" }}
                  />
                </MenuItem>
                <MenuItem onClick={exportAsMarkdown}>
                  <ListItemIcon>
                    <DescriptionIcon sx={{ fontSize: 18, color: "#22c55e" }} />
                  </ListItemIcon>
                  <ListItemText
                    primary="Export as Markdown"
                    secondary="Formatted document"
                    primaryTypographyProps={{ fontSize: "0.75rem", color: "#e2e8f0" }}
                    secondaryTypographyProps={{ fontSize: "0.6rem", color: "#64748b" }}
                  />
                </MenuItem>
              </Menu>
            </>
          )}
        </Box>
      </div>

      <Box sx={{ flex: 1, overflow: "auto", bgcolor: "#0f172a", p: 1.5 }}>
        {/* Metrics Summary */}
        {findings?.metrics && (
          <Box sx={{ mb: 2 }}>
            <Typography
              variant="caption"
              sx={{
                color: "#64748b",
                fontSize: "0.6rem",
                textTransform: "uppercase",
                letterSpacing: "0.1em",
                display: "block",
                mb: 1,
              }}
            >
              Study Metrics
            </Typography>
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              {renderMetricCard("Features", findings.metrics.total_features_analyzed)}
              {renderMetricCard("Layers", findings.metrics.layers_covered)}
              {renderMetricCard("Tasks", findings.metrics.tasks_tested)}
              {renderMetricCard("Accuracy", findings.metrics.avg_reconstruction_accuracy, "%")}
            </Box>
          </Box>
        )}

        <Divider sx={{ borderColor: "#1e293b", mb: 1.5 }} />

        {/* Grouped Findings by Category */}
        {Object.entries(CATEGORY_CONFIG).map(([category, config]) => {
          const categoryFindings = groupedFindings[category];
          if (!categoryFindings || categoryFindings.length === 0) return null;

          return (
            <Box key={category} sx={{ mb: 1.5 }}>
              <Box
                onClick={() => toggleCategory(category)}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  cursor: "pointer",
                  mb: 1,
                  p: 0.5,
                  borderRadius: 1,
                  "&:hover": { bgcolor: "rgba(255,255,255,0.05)" },
                }}
              >
                <Box sx={{ color: config.color }}>{config.icon}</Box>
                <Typography
                  variant="caption"
                  sx={{
                    color: config.color,
                    fontWeight: 600,
                    fontSize: "0.7rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  {config.label}
                </Typography>
                <Chip
                  label={categoryFindings.length}
                  size="small"
                  sx={{
                    height: 16,
                    fontSize: "0.55rem",
                    bgcolor: `${config.color}20`,
                    color: config.color,
                    minWidth: 20,
                  }}
                />
                <ExpandMoreIcon
                  sx={{
                    fontSize: 16,
                    color: "#64748b",
                    ml: "auto",
                    transform: expandedCategories.has(category) ? "rotate(180deg)" : "rotate(0deg)",
                    transition: "transform 0.2s",
                  }}
                />
              </Box>
              {expandedCategories.has(category) && (
                <Box sx={{ pl: 1 }}>
                  {categoryFindings.map((finding) => renderFindingCard(finding))}
                </Box>
              )}
            </Box>
          );
        })}

        {/* Limitations Section */}
        {findings?.limitations && findings.limitations.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Divider sx={{ borderColor: "#1e293b", mb: 1.5 }} />
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mb: 1,
              }}
            >
              <WarningIcon sx={{ fontSize: 14, color: "#f59e0b" }} />
              <Typography
                variant="caption"
                sx={{
                  color: "#f59e0b",
                  fontWeight: 600,
                  fontSize: "0.65rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Limitations
              </Typography>
            </Box>
            <Box
              sx={{
                bgcolor: "rgba(245, 158, 11, 0.1)",
                borderRadius: 1,
                p: 1.5,
                border: "1px solid rgba(245, 158, 11, 0.2)",
              }}
            >
              {findings.limitations.map((limitation, idx) => (
                <Typography
                  key={idx}
                  variant="body2"
                  sx={{
                    color: "#cbd5e1",
                    fontSize: "0.65rem",
                    lineHeight: 1.6,
                    mb: idx < findings.limitations.length - 1 ? 1 : 0,
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 1,
                  }}
                >
                  <span style={{ color: "#f59e0b", fontWeight: 600 }}>{idx + 1}.</span>
                  {limitation}
                </Typography>
              ))}
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
}
