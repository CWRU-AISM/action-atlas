"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { useAppDispatch, useAppSelector } from "@/redux/hooks";
import { setCurrentModel, setCurrentDataset, VLA_MODELS, VLAModelType, DatasetType, DATASET_INFO, DATASET_SUITES } from "@/redux/features/modelSlice";
import { setSelectedLLM, setSelectedLayer, setSelectedSuite, setAvailableLayers, setAvailableSuites } from "@/redux/features/llmSlice";
import { setCurrentLLM } from "@/redux/features/querySlice";
import ConceptQuery from "@/app/components/Concept Query";
import VisualizationDashboard from "@/app/components/VisualizationDashboard";
import SectionC from "@/app/components/FeatureScatter/SectionC";
import ModelSteerVisualization from "@/app/components/Model Steering Visualization";
import FeatureExplorationDashboard from "./components/Feature Exploration Dashboard";
import SectionD1 from "./components/SectionD1";
import ConceptSelector from "./components/ConceptSelector";
import WireVisualization from "./components/WireVisualization";
import DemoVisualization from "./components/DemoVisualization";
import AblationVisualizations from "./components/AblationVisualizations";
import PerturbationTesting from "./components/PerturbationTesting";
import FindingsPanel from "./components/FindingsPanel";
import SceneState from "./components/SceneState";
import ACTLayerVisualization from "./components/ACTLayerVisualization";
import KeyboardShortcutsHelp from "./components/KeyboardShortcutsHelp";
// DisplacementAnalysis is now embedded within SceneState component

// Icons for tabs
const ScatterIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <circle cx="3" cy="5" r="2" />
    <circle cx="8" cy="3" r="2" />
    <circle cx="13" cy="6" r="2" />
    <circle cx="5" cy="11" r="2" />
    <circle cx="11" cy="12" r="2" />
  </svg>
);

const WireIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M1 8 L5 4 L11 12 L15 8" />
    <circle cx="1" cy="8" r="1.5" fill="currentColor" />
    <circle cx="5" cy="4" r="1.5" fill="currentColor" />
    <circle cx="11" cy="12" r="1.5" fill="currentColor" />
    <circle cx="15" cy="8" r="1.5" fill="currentColor" />
  </svg>
);

const VideoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <rect x="1" y="3" width="10" height="10" rx="1" />
    <path d="M12 6 L15 4 L15 12 L12 10 Z" />
  </svg>
);

const ChartIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <rect x="1" y="10" width="3" height="5" rx="0.5" />
    <rect x="5" y="6" width="3" height="9" rx="0.5" />
    <rect x="9" y="3" width="3" height="12" rx="0.5" />
    <rect x="13" y="8" width="2" height="7" rx="0.5" />
  </svg>
);

const PenTestIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    {/* Test tube / flask shape */}
    <path d="M5 2 L5 9 L3 13 C2.5 14 3 15 4 15 L12 15 C13 15 13.5 14 13 13 L11 9 L11 2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M4 2 L12 2" strokeLinecap="round" />
    {/* Bubbles inside */}
    <circle cx="6" cy="11" r="1" fill="currentColor" />
    <circle cx="9" cy="12" r="0.75" fill="currentColor" />
    <circle cx="7.5" cy="9" r="0.5" fill="currentColor" />
  </svg>
);

const FindingsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    {/* Lightbulb shape */}
    <path d="M8 1C5.5 1 3.5 3 3.5 5.5C3.5 7.2 4.5 8.7 6 9.5V11C6 11.6 6.4 12 7 12H9C9.6 12 10 11.6 10 11V9.5C11.5 8.7 12.5 7.2 12.5 5.5C12.5 3 10.5 1 8 1Z" />
    <rect x="6.5" y="13" width="3" height="1" rx="0.5" />
    <rect x="7" y="14.5" width="2" height="0.75" rx="0.375" />
  </svg>
);


// VLA Logo Component - Modern, clean design
function VLALogo() {
  return (
    <div className="flex items-center mr-4">
      {/* Clean modern logo with gradient accent */}
      <div className="relative">
        {/* Glow effect behind */}
        <div className="absolute inset-0 blur-lg opacity-30 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg" />
        <div className="relative flex items-baseline">
          <span
            className="text-3xl font-black tracking-tight"
            style={{
              background: "linear-gradient(135deg, #ef4444 0%, #f97316 50%, #ef4444 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              fontFamily: "'Inter', system-ui, sans-serif",
              letterSpacing: "-0.02em"
            }}
          >
            V
          </span>
          <span
            className="text-3xl font-black tracking-tight text-slate-200"
            style={{
              fontFamily: "'Inter', system-ui, sans-serif",
              letterSpacing: "-0.02em"
            }}
          >
            LA
          </span>
        </div>
      </div>
      {/* Waving robot icon */}
      <svg width="32" height="32" viewBox="0 0 64 64" className="ml-2 robot-wave">
        {/* Antenna */}
        <line x1="32" y1="4" x2="32" y2="14" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
        <circle cx="32" cy="3" r="2.5" fill="#ef4444" className="robot-antenna-blink" />
        {/* Head */}
        <rect x="20" y="14" width="24" height="18" rx="5" fill="#1e293b" stroke="#475569" strokeWidth="1.5" />
        {/* Eyes */}
        <circle cx="28" cy="23" r="3" fill="#22d3ee" className="robot-eye" />
        <circle cx="36" cy="23" r="3" fill="#22d3ee" className="robot-eye" />
        {/* Eye highlights */}
        <circle cx="29" cy="22" r="1" fill="white" opacity="0.7" />
        <circle cx="37" cy="22" r="1" fill="white" opacity="0.7" />
        {/* Mouth - friendly smile */}
        <path d="M27 28 Q32 31 37 28" stroke="#94a3b8" strokeWidth="1.5" fill="none" strokeLinecap="round" />
        {/* Body */}
        <rect x="22" y="34" width="20" height="14" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1.5" />
        {/* Chest light */}
        <circle cx="32" cy="41" r="2" fill="#ef4444" opacity="0.8" className="robot-antenna-blink" />
        {/* Left arm (static) */}
        <rect x="12" y="36" width="8" height="4" rx="2" fill="#334155" />
        <circle cx="12" cy="38" r="2.5" fill="#475569" />
        {/* Right arm (waving) */}
        <g className="robot-arm-wave">
          <rect x="44" y="34" width="8" height="4" rx="2" fill="#334155" />
          <circle cx="52" cy="36" r="2.5" fill="#475569" />
          {/* Hand */}
          <path d="M54 34 L56 30 M54 34 L57 33 M54 34 L56 37" stroke="#f97316" strokeWidth="1.5" strokeLinecap="round" />
        </g>
        {/* Legs */}
        <rect x="25" y="49" width="5" height="8" rx="2" fill="#334155" />
        <rect x="34" y="49" width="5" height="8" rx="2" fill="#334155" />
        {/* Feet */}
        <rect x="23" y="55" width="9" height="4" rx="2" fill="#475569" />
        <rect x="32" y="55" width="9" height="4" rx="2" fill="#475569" />
      </svg>
    </div>
  );
}

const GridAblationIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" opacity="0.9">
    {/* 4x4 grid representing spatial masking */}
    <rect x="1" y="1" width="3" height="3" rx="0.5" />
    <rect x="5" y="1" width="3" height="3" rx="0.5" opacity="0.6" />
    <rect x="9" y="1" width="3" height="3" rx="0.5" opacity="0.3" />
    <rect x="13" y="1" width="2" height="3" rx="0.5" opacity="0.6" />
    <rect x="1" y="5" width="3" height="3" rx="0.5" opacity="0.3" />
    <rect x="5" y="5" width="3" height="3" rx="0.5" />
    <rect x="9" y="5" width="3" height="3" rx="0.5" opacity="0.6" />
    <rect x="1" y="9" width="3" height="3" rx="0.5" opacity="0.6" />
    <rect x="5" y="9" width="3" height="3" rx="0.5" opacity="0.3" />
    <rect x="9" y="9" width="3" height="3" rx="0.5" />
    {/* X mark for ablation */}
    <path d="M12 10 L15 13 M15 10 L12 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
  </svg>
);

const SceneStateIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    {/* Robot arm trajectory */}
    <path d="M2 12 Q5 4 8 8 Q11 12 14 4" strokeLinecap="round" strokeLinejoin="round" />
    {/* Start point */}
    <circle cx="2" cy="12" r="1.5" fill="#22c55e" stroke="none" />
    {/* End point */}
    <circle cx="14" cy="4" r="1.5" fill="#ef4444" stroke="none" />
    {/* Object markers */}
    <rect x="6" y="11" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.5" />
    <rect x="10" y="10" width="2" height="2" rx="0.5" fill="currentColor" opacity="0.5" />
  </svg>
);

type MainView = "features" | "wires" | "videos" | "ablation" | "pentest" | "findings" | "scenestate" | "actlayers";

// Models with SAE data (all except ACT)
const MODELS_WITH_SAES: VLAModelType[] = ['pi05', 'openvla', 'xvla', 'smolvla', 'groot'];
// Tabs that require SAE / layer data
const SAE_ONLY_TABS: MainView[] = ['features', 'pentest', 'ablation'];
// Tabs restricted to models with scene state data (all SAE models have trajectories)
const SCENE_STATE_MODELS: VLAModelType[] = ['pi05', 'openvla', 'xvla', 'smolvla', 'groot', 'act'];
// Tabs restricted to ACT only
const ACT_ONLY_TABS: MainView[] = ['actlayers'];
// Tabs available for ALL models (wires handles ACT with architecture diagram)
// 'wires', 'videos', 'findings' are available for all models by default


function isTabAvailable(tab: MainView, model: VLAModelType): boolean {
  if (ACT_ONLY_TABS.includes(tab)) return model === 'act';
  if (tab === 'scenestate') return SCENE_STATE_MODELS.includes(model);
  if (SAE_ONLY_TABS.includes(tab)) return MODELS_WITH_SAES.includes(model);
  return true; // wires, videos, findings available for all models
}

export default function Home() {
  const [activeView, setActiveView] = useState<MainView>("features");
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const dispatch = useAppDispatch();
  const currentModel = useAppSelector((state) => state.model.currentModel);
  const currentDataset = useAppSelector((state) => state.model.currentDataset);

  // Available datasets for current model
  const availableDatasets = VLA_MODELS[currentModel].environments as readonly DatasetType[];

  // Sync the LLM/SAE slice when global model or dataset changes so feature explorer
  // loads the correct clustering data for the selected model.
  useEffect(() => {
    // Update currentLLM so scatter API sends correct model identifier
    dispatch(setCurrentLLM(currentModel));

    // Set layers based on model
    if (currentModel === 'pi05') {
      dispatch(setSelectedLayer('action_expert_layer_12'));
      dispatch(setSelectedSuite('concepts'));
      dispatch(setSelectedLLM('action_expert_layer_12-concepts'));
      dispatch(setAvailableLayers([
        { display: 'All', value: 'all_layers' },
        ...Array.from({ length: 18 }, (_, i) => ({ display: `L${i}`, value: `action_expert_layer_${i}` })),
        { display: 'InProj', value: 'action_in_proj' },
        { display: 'OutProjIn', value: 'action_out_proj_input' },
      ]));
    } else if (currentModel === 'openvla') {
      dispatch(setSelectedLayer('layer_16'));
      dispatch(setSelectedSuite('libero_goal'));
      dispatch(setSelectedLLM('layer_16-libero_goal'));
      dispatch(setAvailableLayers([
        ...Array.from({ length: 32 }, (_, i) => ({ display: `L${i}`, value: `layer_${i}` })),
      ]));
    } else if (currentModel === 'xvla') {
      dispatch(setSelectedLayer('layer_12'));
      const firstSuite = currentDataset === 'simplerenv' ? 'simplerenv_widowx' : 'libero_goal';
      dispatch(setSelectedSuite(firstSuite));
      dispatch(setSelectedLLM(`layer_12-${firstSuite}`));
      dispatch(setAvailableLayers([
        ...Array.from({ length: 24 }, (_, i) => ({ display: `L${i}`, value: `layer_${i}` })),
      ]));
    } else if (currentModel === 'smolvla') {
      dispatch(setSelectedLayer('expert_layer_16'));
      const firstSuite = currentDataset === 'metaworld' ? 'metaworld_easy' : 'libero_goal';
      dispatch(setSelectedSuite(firstSuite));
      dispatch(setSelectedLLM(`expert_layer_16-${firstSuite}`));
      dispatch(setAvailableLayers([
        ...Array.from({ length: 32 }, (_, i) => ({ display: `VLM L${i}`, value: `vlm_layer_${i}` })),
        ...Array.from({ length: 32 }, (_, i) => ({ display: `Expert L${i}`, value: `expert_layer_${i}` })),
      ]));
    } else if (currentModel === 'groot') {
      dispatch(setSelectedLayer('dit_layer_8'));
      dispatch(setSelectedSuite('libero_goal'));
      dispatch(setSelectedLLM('dit_layer_8-libero_goal'));
      dispatch(setAvailableLayers([
        ...Array.from({ length: 16 }, (_, i) => ({ display: `DiT L${i}`, value: `dit_layer_${i}` })),
        ...Array.from({ length: 12 }, (_, i) => ({ display: `Eagle L${i}`, value: `eagle_layer_${i}` })),
        ...Array.from({ length: 4 }, (_, i) => ({ display: `VL-SA L${i}`, value: `vlsa_layer_${i}` })),
      ]));
    }

    // Set available suites based on both model and dataset
    if (currentModel === 'pi05') {
      dispatch(setAvailableSuites([
        { display: 'Goal', value: 'concepts' },
        { display: 'Object', value: 'object' },
        { display: 'Spatial', value: 'spatial' },
        { display: 'LIBERO-10', value: 'libero_10' },
      ]));
    } else if (currentModel === 'xvla') {
      // Filter suites based on selected dataset
      dispatch(setAvailableSuites(DATASET_SUITES[currentDataset] || DATASET_SUITES.libero));
    } else if (currentModel === 'smolvla') {
      // Filter suites based on selected dataset
      dispatch(setAvailableSuites(DATASET_SUITES[currentDataset] || DATASET_SUITES.libero));
    } else if (currentModel === 'groot') {
      dispatch(setAvailableSuites([
        { display: 'Object', value: 'libero_object' },
        { display: 'Goal', value: 'libero_goal' },
        { display: 'Long', value: 'libero_long' },
      ]));
    } else if (currentModel === 'openvla') {
      dispatch(setAvailableSuites([
        { display: 'Goal', value: 'libero_goal' },
        { display: 'Spatial', value: 'libero_spatial' },
        { display: 'Object', value: 'libero_object' },
        { display: 'LIBERO-10', value: 'libero_10' },
      ]));
    }
    // ACT has no SAE data, no need to update
  }, [currentModel, currentDataset, dispatch]);

  // Auto-switch to an available tab whenever the model changes (covers both
  // dropdown changes AND programmatic dispatches from other components).
  // We use a ref to track the previous model so we can detect model changes
  // and always force-switch to the best default tab for the new model.
  const prevModelRef = useRef(currentModel);
  useEffect(() => {
    const modelChanged = prevModelRef.current !== currentModel;
    if (modelChanged) {
      prevModelRef.current = currentModel;
      // Only switch tab if the current tab is not available for the new model.
      // This preserves tabs like 'wires' that work across all models.
      if (!isTabAvailable(activeView, currentModel)) {
        if (currentModel === 'act') {
          setActiveView('actlayers');
        } else {
          setActiveView('features');
        }
      }
    } else if (!isTabAvailable(activeView, currentModel)) {
      // Safety net: if the active view somehow becomes invalid, fix it.
      if (currentModel === 'act') {
        setActiveView('actlayers');
      } else {
        setActiveView('features');
      }
    }
  }, [currentModel, activeView]);

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value as VLAModelType;
    dispatch(setCurrentModel(newModel));
    // Only switch tab if current tab is unavailable for the new model.
    // This preserves tabs like 'wires' that work across all models.
    if (!isTabAvailable(activeView, newModel)) {
      if (newModel === 'act') {
        setActiveView('actlayers');
      } else {
        setActiveView('features');
      }
    }
  };

  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDataset = e.target.value as DatasetType;
    dispatch(setCurrentDataset(newDataset));
  };

  // Global keyboard shortcut handler for ? key
  const handleGlobalKeyDown = useCallback((e: KeyboardEvent) => {
    // Ignore if focus is on an input element
    if (
      e.target instanceof HTMLInputElement ||
      e.target instanceof HTMLTextAreaElement
    ) {
      return;
    }

    // Check for ? key (Shift + / on most keyboards, or direct ?)
    if (e.key === "?" || (e.shiftKey && e.key === "/")) {
      e.preventDefault();
      setShowKeyboardHelp(true);
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [handleGlobalKeyDown]);

  const allTabs = [
    { id: "features" as MainView, label: "Feature Explorer", icon: <ScatterIcon /> },
    { id: "wires" as MainView, label: "Layer Circuits", icon: <WireIcon /> },
    { id: "actlayers" as MainView, label: "ACT Layers", icon: <GridAblationIcon /> },
    { id: "videos" as MainView, label: "Demos", icon: <VideoIcon /> },
    { id: "ablation" as MainView, label: "Ablation Studies", icon: <ChartIcon /> },
    { id: "pentest" as MainView, label: "Pen Testing", icon: <PenTestIcon /> },
    { id: "scenestate" as MainView, label: "Scene State", icon: <SceneStateIcon /> },
    { id: "findings" as MainView, label: "Findings", icon: <FindingsIcon /> },
  ];

  const tabs = allTabs.filter((tab) => isTabAvailable(tab.id, currentModel));

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      {/* Header - Dark Navy */}
      <div className="h-14 bg-[#0a1628] flex items-center px-4 shadow-lg">
        <VLALogo />
        <div className="flex flex-col">
          <h1 className="text-xl font-semibold text-white tracking-wide" style={{ fontFamily: "system-ui, sans-serif" }}>
            Action Atlas
          </h1>
          <span className="text-slate-500 text-[10px]">Mechanistic Interpretability for Vision-Language-Action Models</span>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <select
            value={currentModel}
            onChange={handleModelChange}
            className="text-[11px] text-slate-300 bg-slate-800 border border-slate-700 px-2 py-1 rounded cursor-pointer hover:bg-slate-700 focus:outline-none focus:ring-1 focus:ring-red-500"
          >
            {Object.values(VLA_MODELS).map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.params})
              </option>
            ))}
          </select>
          {/* Dataset selector - only show when model supports multiple datasets */}
          {availableDatasets.length > 1 ? (
            <select
              value={currentDataset}
              onChange={handleDatasetChange}
              className="text-[11px] text-slate-300 bg-slate-800 border border-slate-700 px-2 py-1 rounded cursor-pointer hover:bg-slate-700 focus:outline-none focus:ring-1 focus:ring-orange-500"
            >
              {availableDatasets.map((ds) => (
                <option key={ds} value={ds}>
                  {DATASET_INFO[ds].name}
                </option>
              ))}
            </select>
          ) : (
            <span className="text-[10px] text-slate-400 bg-slate-800/50 px-2 py-0.5 rounded border border-slate-700">
              {DATASET_INFO[currentDataset].name}
            </span>
          )}
          <span className="text-[10px] text-slate-400 bg-slate-800/50 px-2 py-0.5 rounded">
            {VLA_MODELS[currentModel].layers} Layers | {VLA_MODELS[currentModel].actionGen}
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-2 overflow-hidden bg-slate-100 p-2">
        {/* Left Panel - Wider */}
        <div className="h-full flex flex-col w-[300px] gap-2 overflow-hidden">
          <div className="h-[200px] flex-none">
            <ConceptQuery />
          </div>
          <div className="flex-1 overflow-auto min-h-0">
            <ConceptSelector />
          </div>
        </div>

        {/* Center Panel - Main Display with Tabs */}
        <div className="flex flex-col flex-1 overflow-hidden h-full bg-white rounded-lg shadow-lg">
          {/* Tab Bar - Dark Navy */}
          <div className="h-10 flex-none flex items-center bg-[#0a1628] px-3 gap-2 rounded-t-lg border-b border-slate-700">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id)}
                className={`flex items-center gap-2 px-4 py-1.5 rounded text-xs font-medium transition-all ${
                  activeView === tab.id
                    ? "bg-[#ef4444] text-white shadow-lg"
                    : "text-slate-400 hover:text-white hover:bg-slate-800"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Main Display Area - Only one view at a time */}
          <div className="flex-1 overflow-hidden">
            {activeView === "features" && isTabAvailable("features", currentModel) && <SectionC />}
            {activeView === "wires" && isTabAvailable("wires", currentModel) && <WireVisualization />}
            {activeView === "actlayers" && isTabAvailable("actlayers", currentModel) && <ACTLayerVisualization />}
            {activeView === "videos" && <DemoVisualization />}
            {activeView === "ablation" && isTabAvailable("ablation", currentModel) && <AblationVisualizations />}
            {activeView === "pentest" && isTabAvailable("pentest", currentModel) && <PerturbationTesting />}
            {activeView === "scenestate" && isTabAvailable("scenestate", currentModel) && <SceneState />}
            {activeView === "findings" && <FindingsPanel />}
          </div>
        </div>

        {/* Right Panel */}
        <div className="flex flex-col w-[300px] gap-2 overflow-hidden h-full">
          <div className="flex-1 overflow-auto min-h-0">
            <SectionD1 />
          </div>
          <div className="flex-1 overflow-auto min-h-0">
            <ModelSteerVisualization />
          </div>
        </div>
      </div>

      {/* Bottom Panel - Layer Dashboard */}
      <div className="h-[160px] flex-none p-1 pt-0 bg-slate-100">
        <VisualizationDashboard />
      </div>

      {/* Keyboard Shortcuts Help Modal */}
      <KeyboardShortcutsHelp
        open={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />
    </div>
  );
}
