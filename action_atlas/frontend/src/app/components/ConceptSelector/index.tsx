"use client";
import React, { useState, useEffect } from "react";
import {
  Paper,
  Typography,
  Chip,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Tooltip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import CategoryIcon from "@mui/icons-material/Category";
import PlaceIcon from "@mui/icons-material/Place";
import PanToolIcon from "@mui/icons-material/PanTool";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import { setSelectedConcept as setReduxConcept, clearSelectedConcept, updateHighlightedFeaturesByStrength } from "@/redux/features/conceptSlice";
import { API_BASE_URL } from "@/config/api";

interface ConceptData {
  motion: string[];
  object: string[];
  spatial: string[];
  action_phase?: string[];
}

interface ConceptFeatureInfo {
  total_features: number;
  concept_features: number;
  strong_features: number;
  very_strong_features: number;
  feature_indices: number[];
  avg_selectivity?: number;
  max_selectivity?: number;
}

export default function ConceptSelector() {
  const dispatch = useAppDispatch();
  const { selectedConcept, selectedType, conceptLayers } = useAppSelector((state) => state.concept);
  const selectedStrength = useAppSelector((state) => state.llm.selectedStrength);
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const [concepts, setConcepts] = useState<ConceptData | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch available concepts (filtered by model)
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/concepts/list?model=${currentModel}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 200) {
          setConcepts(data.data);
        }
      })
      .catch(console.error);
  }, [currentModel]);

  // Update highlighted features when strength changes
  useEffect(() => {
    if (selectedConcept && conceptLayers) {
      dispatch(updateHighlightedFeaturesByStrength(selectedStrength));
    }
  }, [selectedStrength, selectedConcept, conceptLayers, dispatch]);

  // Fetch features when concept is selected
  const handleConceptClick = async (concept: string, type: string) => {
    if (selectedConcept === concept && selectedType === type) {
      // Deselect
      dispatch(clearSelectedConcept());
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(
        `${API_BASE_URL}/api/concepts/features?concept=${concept}&type=${type}&model=${currentModel}`
      );
      const data = await res.json();
      if (data.status === 200) {
        dispatch(setReduxConcept({
          concept,
          type,
          layers: data.data.layers,
          strength: selectedStrength
        }));
      }
    } catch (error) {
      console.error("Failed to fetch concept features:", error);
    } finally {
      setLoading(false);
    }
  };

  const getConceptColor = (type: string) => {
    switch (type) {
      case "motion":
        return "primary";
      case "object":
        return "secondary";
      case "spatial":
        return "success";
      case "action_phase":
        return "warning";
      default:
        return "default";
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "motion":
        return <DirectionsRunIcon fontSize="small" />;
      case "object":
        return <CategoryIcon fontSize="small" />;
      case "spatial":
        return <PlaceIcon fontSize="small" />;
      case "action_phase":
        return <PanToolIcon fontSize="small" />;
      default:
        return null;
    }
  };

  const getTypeLabel = (type: string) => {
    switch (type) {
      case "action_phase":
        return "Action Phase";
      default:
        return type;
    }
  };

  const getTotalFeatures = () => {
    if (!conceptLayers) return 0;
    return Object.values(conceptLayers).reduce(
      (sum, layer) => sum + layer.concept_features,
      0
    );
  };

  const getStrongFeatures = () => {
    if (!conceptLayers) return 0;
    return Object.values(conceptLayers).reduce(
      (sum, layer) => sum + layer.strong_features,
      0
    );
  };

  if (!concepts) {
    return (
      <Paper className="p-4 h-full">
        <CircularProgress size={24} />
      </Paper>
    );
  }

  return (
    <Paper className="h-full flex flex-col overflow-hidden rounded-lg shadow-md">
      {/* Dark Navy Header */}
      <div className="h-8 flex items-center px-3 bg-[#0a1628] rounded-t-lg flex-shrink-0">
        <span className="text-xs font-semibold text-white">Concept Selector</span>
      </div>
      <div className="flex-1 p-3 overflow-auto bg-white">
      <Typography variant="caption" color="text.secondary" display="block" mb={1}>
        Select a concept to see related features across layers
      </Typography>

      {/* Concept Type Accordions — only show types that have data */}
      {(["motion", "object", "spatial", "action_phase"] as const).filter((type) => concepts[type] && concepts[type]!.length > 0).map((type) => {
        const items = concepts[type]!;
        return (
        <Accordion key={type} defaultExpanded sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box display="flex" alignItems="center" gap={1}>
              {getTypeIcon(type)}
              <Typography variant="body2" fontWeight="medium" textTransform="capitalize">
                {getTypeLabel(type)} Concepts
              </Typography>
              <Chip
                label={items.length}
                size="small"
                color={getConceptColor(type)}
                sx={{ height: 20, fontSize: "0.7rem" }}
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Box display="flex" flexWrap="wrap" gap={0.5}>
              {items.map((concept) => (
                <Tooltip
                  key={concept}
                  title={`Click to see "${concept}" features`}
                  arrow
                >
                  <Chip
                    label={concept.replace("_", " ")}
                    size="small"
                    color={
                      selectedConcept === concept && selectedType === type
                        ? getConceptColor(type)
                        : "default"
                    }
                    variant={
                      selectedConcept === concept && selectedType === type
                        ? "filled"
                        : "outlined"
                    }
                    onClick={() => handleConceptClick(concept, type)}
                    sx={{ cursor: "pointer", textTransform: "capitalize" }}
                  />
                </Tooltip>
              ))}
            </Box>
          </AccordionDetails>
        </Accordion>
        );
      })}

      {/* Selected Concept Info */}
      {selectedConcept && (
        <Box mt={2} p={1.5} bgcolor="action.hover" borderRadius={1}>
          {loading ? (
            <CircularProgress size={20} />
          ) : (
            <>
              <Typography variant="body2" fontWeight="bold" gutterBottom sx={{ fontSize: "0.8rem" }}>
                {selectedConcept.replace("_", " ")} ({selectedType})
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} mb={0.5}>
                <Typography variant="caption" sx={{ fontSize: "0.7rem" }}>
                  Total: <strong>{getTotalFeatures().toLocaleString()}</strong>
                </Typography>
                <Typography variant="caption" sx={{ fontSize: "0.7rem" }}>
                  Strong (&gt;3x): <strong>{getStrongFeatures().toLocaleString()}</strong>
                </Typography>
              </Box>
              <Typography variant="caption" display="block" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>
                Selectivity = in-concept / out-concept activation ratio
              </Typography>
              {conceptLayers && (
                <Box mt={1}>
                  <Typography variant="caption" fontWeight="medium" sx={{ fontSize: "0.7rem" }}>
                    Top Layers:
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={0.5} mt={0.5}>
                    {Object.entries(conceptLayers)
                      .sort((a, b) => b[1].concept_features - a[1].concept_features)
                      .slice(0, 5)
                      .map(([layer, info]) => (
                        <Tooltip
                          key={layer}
                          title={`Strong: ${info.strong_features}, Very Strong: ${info.very_strong_features}`}
                          arrow
                        >
                          <Chip
                            key={layer}
                            label={`L${layer.replace("action_expert_layer_", "")}: ${info.concept_features}`}
                            size="small"
                            color={info.strong_features > 0 ? "primary" : "default"}
                            sx={{ fontSize: "0.6rem", height: 16 }}
                          />
                        </Tooltip>
                      ))}
                  </Box>
                </Box>
              )}
            </>
          )}
        </Box>
      )}
      </div>
    </Paper>
  );
}
