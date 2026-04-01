"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Box, Typography, Select, MenuItem, FormControl, InputLabel, Chip, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { useAppSelector } from "@/redux/hooks";
import { DATASET_SUITES, DatasetType } from "@/redux/features/modelSlice";
import { API_BASE_URL } from "@/config/api";

export default function InjectionDemo() {
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);
  const effectiveModel = currentModel;
  const datasetSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
  const defaultSuite = datasetSuites.length > 0 ? datasetSuites[0] : 'libero_goal';
  const [model, setModel] = useState<string>(effectiveModel);
  const [suite, setSuite] = useState(defaultSuite);
  const [injectionType, setInjectionType] = useState<string>("cross_task");
  const [data, setData] = useState<any>(null);
  const [selectedPair, setSelectedPair] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Sync with global model
  useEffect(() => { setModel(effectiveModel); }, [effectiveModel]);

  // Reset suite when dataset changes — force refetch with first suite of new dataset
  useEffect(() => {
    const newSuites = (DATASET_SUITES[currentDataset as DatasetType] || []).map(s => s.value);
    if (newSuites.length > 0) {
      setSuite(newSuites[0]);
    }
  }, [currentDataset]);

  // Helper: load videos from the baked video index as fallback
  const loadVideosFallback = async (
    mdl: string,
    ste: string,
    injType: string
  ): Promise<Record<string, string>> => {
    const videoMap: Record<string, string> = {};
    // Map injection type to experiment_type in the baked index
    const experimentTypes: string[] = [];
    if (injType === 'cross_task') experimentTypes.push('cross_task');
    else if (injType === 'null') experimentTypes.push('null_injection');
    else if (injType === 'temporal') experimentTypes.push('temporal_injection');
    else if (injType === 'same_scene') experimentTypes.push('cross_scene_injection');
    else experimentTypes.push(injType);

    for (const expType of experimentTypes) {
      try {
        const vidRes = await fetch(
          `${API_BASE_URL}/api/vla/videos?model=${mdl}&experiment_type=${expType}&limit=500`
        );
        if (!vidRes.ok) continue;
        const vidData = await vidRes.json();
        const videos = vidData.videos || [];
        // Filter by suite if available
        const filtered = ste
          ? videos.filter((v: any) => !v.suite || v.suite === ste || v.suite === ste.replace('libero_', ''))
          : videos;
        for (const v of filtered.slice(0, 50)) {
          const name = v.filename || v.path?.split('/').pop() || `video_${Object.keys(videoMap).length}`;
          videoMap[name] = v.path;
        }
      } catch {
        // Ignore
      }
    }
    return videoMap;
  };

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/vla/injection?model=${model}&suite=${suite}&type=${injectionType}`);
      const json = await res.json();
      if (!json.status || json.status !== 404) {
        // If the injection API returned data but with few/no videos, augment from baked index
        const existingVideos = json.videos ? Object.keys(json.videos).length : 0;
        const hasConditions = json.conditions ? Object.keys(json.conditions).length : 0;
        if (existingVideos < 20 && hasConditions === 0) {
          const fallbackVideos = await loadVideosFallback(model, suite, injectionType);
          const mergedVideos = { ...(json.videos || {}), ...fallbackVideos };
          json.videos = mergedVideos;
          if (!json.key_finding && Object.keys(mergedVideos).length > 0) {
            json.key_finding = `${Object.keys(mergedVideos).length} ${injectionType.replace(/_/g, ' ')} injection videos available for ${model}.`;
          }
        }
        setData(json);
      } else if (json.available_suites && json.available_suites.length > 0) {
        // Backend returned 404 but with available_suites hint
        // Auto-switch to the first available suite
        if (!json.available_suites.includes(suite)) {
          setSuite(json.available_suites[0]);
          // The useEffect will re-fetch with the new suite
          setLoading(false);
          return;
        }
        // Still try to load videos from baked index as fallback
        const videoMap = await loadVideosFallback(model, suite, injectionType);
        setData({
          model,
          suite,
          injection_type: injectionType,
          conditions: {},
          summary: {},
          videos: videoMap,
          available_suites: json.available_suites,
          key_finding: Object.keys(videoMap).length > 0
            ? `${Object.keys(videoMap).length} ${injectionType.replace(/_/g, ' ')} videos available for ${model}.`
            : (json.error || `No data for ${suite}. Try: ${json.available_suites.slice(0, 3).join(', ')}`),
        });
      } else {
        // Fallback: try loading videos from baked video index
        const videoMap = await loadVideosFallback(model, suite, injectionType);
        if (Object.keys(videoMap).length > 0) {
          setData({
            model,
            suite,
            injection_type: injectionType,
            conditions: {},
            summary: {},
            videos: videoMap,
            n_pairs: 0,
            key_finding: `${Object.keys(videoMap).length} ${injectionType.replace(/_/g, ' ')} injection videos available for ${model}.`,
          });
        } else {
          setData(null);
        }
      }
    } catch { setData(null); }
    setLoading(false);
  }, [model, suite, injectionType]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const pairs = data?.conditions ? Object.entries(data.conditions) : [];

  return (
    <Box>
      <Typography variant="body2" sx={{ color: "#94a3b8", mb: 2 }}>
        Activation injection experiments: inject activations from one task/scene into another and observe the effect.
        <strong> Cross-task</strong> = inject Task B activations into Task A execution.
        <strong> Same-scene</strong> = inject correct-task activations from a different viewpoint.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
        <Chip label={{act: "ACT-ALOHA", pi05: "Pi0.5", openvla: "OpenVLA-OFT", xvla: "X-VLA", smolvla: "SmolVLA", groot: "GR00T"}[model] || model.toUpperCase()}
          sx={{ bgcolor: "#1e293b", color: "#e2e8f0", fontWeight: 600 }} />

        {model !== "act" && (
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel sx={{ color: "#64748b" }}>Suite</InputLabel>
            <Select value={suite} label="Suite" onChange={(e) => setSuite(e.target.value)}
              sx={{ color: "#e2e8f0", ".MuiOutlinedInput-notchedOutline": { borderColor: "#334155" } }}>
              {(datasetSuites.length > 0 ? datasetSuites : ["libero_goal", "libero_10", "libero_object", "libero_spatial"]).map(s => (
                <MenuItem key={s} value={s}>{s.replace(/_/g, ' ')}</MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        <ToggleButtonGroup value={injectionType} exclusive onChange={(_, v) => v && setInjectionType(v)} size="small"
          sx={{ "& .MuiToggleButton-root": { color: "#64748b", borderColor: "#334155", fontSize: "11px", px: 1.5,
            "&.Mui-selected": { color: "#ef4444", bgcolor: "#ef444420" } } }}>
          <ToggleButton value="cross_task">Cross-Task</ToggleButton>
          <ToggleButton value="same_scene">Same-Scene</ToggleButton>
          <ToggleButton value="null">Null</ToggleButton>
          <ToggleButton value="temporal">Temporal</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {loading && <Typography sx={{ color: "#64748b" }}>Loading...</Typography>}

      {data && (
        <>
          {/* Summary stats */}
          {data.summary && Object.keys(data.summary).length > 0 && (
            <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
              {Object.entries(data.summary as Record<string, any>).map(([k, v]: [string, any]) => (
                <Box key={k} sx={{ bgcolor: "#1e293b", borderRadius: 1, p: 2, minWidth: 130 }}>
                  <Typography variant="caption" sx={{ color: "#64748b", fontWeight: 600, display: "block" }}>
                    {k.replace(/_/g, " ")}
                  </Typography>
                  <Typography sx={{
                    color: v.success_rate > 0.5 ? "#22c55e" : v.success_rate > 0.2 ? "#eab308" : "#ef4444",
                    fontWeight: 700, fontSize: "24px"
                  }}>
                    {(v.success_rate * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="caption" sx={{ color: "#475569", fontSize: "9px" }}>
                    {v.successes}/{v.total}
                  </Typography>
                </Box>
              ))}
            </Box>
          )}

          {/* Key finding box */}
          {data.key_finding && (
            <Box sx={{ mb: 3, p: 2, bgcolor: "#1e293b", borderRadius: 1, borderLeft: "3px solid #3b82f6" }}>
              <Typography variant="body2" sx={{ color: "#93c5fd", fontSize: "12px" }}>
                {data.key_finding}
              </Typography>
            </Box>
          )}

          {/* Available suites hint */}
          {data.available_suites && data.available_suites.length > 0 && pairs.length === 0 && (
            <Box sx={{ mb: 2, p: 1.5, bgcolor: "#1e293b", borderRadius: 1 }}>
              <Typography variant="caption" sx={{ color: "#64748b", fontSize: "10px", display: "block", mb: 0.5 }}>
                Available suites for {({act: "ACT-ALOHA", pi05: "Pi0.5", openvla: "OpenVLA-OFT", xvla: "X-VLA", smolvla: "SmolVLA", groot: "GR00T"} as Record<string, string>)[model] || model}:
              </Typography>
              <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
                {data.available_suites.map((s: string) => (
                  <Chip key={s} label={s.replace(/_/g, ' ')} size="small"
                    onClick={() => setSuite(s)}
                    sx={{ cursor: "pointer", bgcolor: s === suite ? "#ef444430" : "#0f172a",
                      color: s === suite ? "#ef4444" : "#94a3b8", fontSize: "10px",
                      "&:hover": { bgcolor: "#334155", color: "#e2e8f0" } }} />
                ))}
              </Box>
            </Box>
          )}

          {/* No data message */}
          {!loading && pairs.length === 0 && (!data.videos || Object.keys(data.videos).length === 0) && (!data.summary || Object.keys(data.summary).length === 0) && (
            <Box sx={{ py: 4, textAlign: "center" }}>
              <Typography variant="body2" sx={{ color: "#475569" }}>
                No {injectionType.replace(/_/g, ' ')} injection data available for {({act: "ACT-ALOHA", pi05: "Pi0.5", openvla: "OpenVLA-OFT", xvla: "X-VLA", smolvla: "SmolVLA", groot: "GR00T"} as Record<string, string>)[model] || model} / {suite.replace(/_/g, ' ')}.
                {data?.available_suites?.length > 0 && ` Try: ${data.available_suites.slice(0, 3).join(', ')}`}
              </Typography>
            </Box>
          )}

          {/* Pair list */}
          {injectionType === "cross_task" && pairs.length > 0 && (
            <>
              <Typography variant="subtitle2" sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}>
                Task Pairs ({pairs.length} pairs)
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 1, maxHeight: 400, overflow: "auto" }}>
                {pairs.map(([pairKey, pairData]: [string, any]) => {
                  const isExpanded = selectedPair === pairKey;
                  const baseA = pairData.baseline_a || {};
                  const baseB = pairData.baseline_b || {};

                  return (
                    <Box key={pairKey}
                      onClick={() => setSelectedPair(isExpanded ? null : pairKey)}
                      sx={{
                        bgcolor: isExpanded ? "#1e293b" : "#0f172a", borderRadius: 1, p: 1.5,
                        cursor: "pointer", border: isExpanded ? "1px solid #ef4444" : "1px solid #1e293b",
                        "&:hover": { bgcolor: "#1e293b" }
                      }}
                    >
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <Typography sx={{ color: "#e2e8f0", fontWeight: 600, fontSize: "12px" }}>
                          {pairKey.replace(/_/g, " ")}
                        </Typography>
                        <Box sx={{ display: "flex", gap: 0.5 }}>
                          <Chip label={baseA.success ? "A:OK" : "A:FAIL"} size="small"
                            sx={{ height: 16, fontSize: "8px", bgcolor: baseA.success ? "#22c55e20" : "#dc262620", color: baseA.success ? "#22c55e" : "#ef4444" }} />
                          <Chip label={baseB.success ? "B:OK" : "B:FAIL"} size="small"
                            sx={{ height: 16, fontSize: "8px", bgcolor: baseB.success ? "#22c55e20" : "#dc262620", color: baseB.success ? "#22c55e" : "#ef4444" }} />
                        </Box>
                      </Box>

                      {pairData.desc_a && (
                        <Typography variant="caption" sx={{ color: "#475569", fontSize: "10px", display: "block", mt: 0.5 }}>
                          A: {pairData.desc_a}
                        </Typography>
                      )}
                      {pairData.desc_b && (
                        <Typography variant="caption" sx={{ color: "#475569", fontSize: "10px", display: "block" }}>
                          B: {pairData.desc_b}
                        </Typography>
                      )}

                      {isExpanded && pairData.injection_b_into_a && (
                        <Box sx={{ mt: 1.5, pt: 1, borderTop: "1px solid #334155" }}>
                          <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 600, display: "block", mb: 0.5 }}>
                            Inject B → A (per layer)
                          </Typography>
                          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                            {Object.entries(pairData.injection_b_into_a).map(([layer, result]: [string, any]) => (
                              <Chip key={layer}
                                label={`${layer}: ${result.success ? "OK" : "FAIL"} (${result.n_steps} steps)`}
                                size="small"
                                sx={{
                                  height: 20, fontSize: "9px",
                                  bgcolor: result.success ? "#22c55e20" : "#dc262620",
                                  color: result.success ? "#22c55e" : "#ef4444"
                                }}
                              />
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  );
                })}
              </Box>
            </>
          )}

          {/* Video section */}
          {data.videos && Object.keys(data.videos).length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" sx={{ color: "#94a3b8", mb: 1, fontWeight: 600, fontSize: "11px", textTransform: "uppercase" }}>
                Rollout Videos ({Object.keys(data.videos).length})
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 1, maxHeight: 300, overflow: "auto" }}>
                {Object.entries(data.videos).slice(0, 24).map(([name, path]: [string, any]) => {
                  const pathStr = path as string;
                  // Handle different path formats: /videos/..., full URL, or relative path
                  const videoUrl = pathStr.startsWith('http') ? pathStr
                    : pathStr.startsWith('/api/') ? `${API_BASE_URL}${pathStr}`
                    : `${API_BASE_URL}/api/vla/video/${pathStr.replace(/^\/videos\//, '')}`;
                  return (
                    <Box key={name} sx={{ bgcolor: "#0f172a", borderRadius: 1, overflow: "hidden" }}>
                      <video src={videoUrl}
                        controls muted style={{ width: "100%", height: 120, objectFit: "cover" }} />
                      <Typography variant="caption" sx={{ color: "#64748b", p: 0.5, display: "block", fontSize: "9px" }}>
                        {name}
                      </Typography>
                    </Box>
                  );
                })}
              </Box>
            </Box>
          )}
        </>
      )}
    </Box>
  );
}
