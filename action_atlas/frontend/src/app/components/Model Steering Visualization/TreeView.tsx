"use client";
import {Button, Paper, TextField} from "@mui/material";
import React from "react";
import * as d3 from "d3";
import {AnimatePresence, motion} from "framer-motion";
import {X} from 'lucide-react';
import {TreeNode, TreeViewProps} from "@/types/types";

export default function TreeView({
                                   strengths,
                                   setStrengthsAction,
                                   prompt,
                                   results,
                                   setResultsAction,
                                   selectedBranch,
                                   setSelectedBranchAction
                                 }: TreeViewProps) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [redrawTrigger, setRedrawTrigger] = React.useState(0);
  const treeHeight = Math.max(strengths.length * 100, 400);
  const nodeSpacing = treeHeight / ((strengths.length + 1) * 1);

  const getStrokeColor = (strength: number) => {
    const absStrength = Math.abs(strength);
    if (absStrength === 0) return `rgb(100, 100, 100)`;
    const intensity = 180 + (absStrength / 300) * 75;
    return strength > 0 ? `rgb(${intensity}, 100, 100)` : `rgb(100, 100, ${intensity})`;
  };

  // 添加一个计算stroke width的函数
  const getStrokeWidth = (strength: number) => {
    const absStrength = Math.abs(strength);
    const minWidth = 2;
    const maxWidth = 12;
    // 将强度范围 [0, 300] 映射到线宽范围 [minWidth, maxWidth]
    const normalizedStrength = absStrength / 300;
    return minWidth + (maxWidth - minWidth) * normalizedStrength;
  };

  const getSortedBranches = () => {
    const branchesWithIndices = strengths.map((strength, originalIndex) => ({
      strength,
      originalIndex,
      isSelected: selectedBranch === originalIndex
    }));
    const [baseBranch, ...otherBranches] = branchesWithIndices;
    return [baseBranch, ...otherBranches.sort((a, b) => b.strength - a.strength)];
  };

  const handleAddStrength = () => {
    if (strengths.length >= 7) return;
    const newStrengths = [...strengths, 50];
    const [baseStrength, ...restStrengths] = newStrengths;
    const sortedStrengths = [baseStrength, ...restStrengths.sort((a, b) => b - a)];
    setStrengthsAction(sortedStrengths);
    setSelectedBranchAction(sortedStrengths.indexOf(50));
  };

  const handleBranchClick = (index: number) => {
    if (index === 0) return;
    setSelectedBranchAction(selectedBranch === index ? null : index);
  };

  const handleStrengthChange = (index: number, newValue: number) => {
    if (index === 0 || isNaN(newValue)) return;
    let clampedValue = Math.min(300, Math.max(-300, newValue));
    if (clampedValue === 0) clampedValue = newValue > 0 ? 1 : -1;
    const newStrengths = [...strengths];
    newStrengths[index] = clampedValue;
    const [baseStrength, ...restStrengths] = newStrengths;
    const sortedStrengths = [baseStrength, ...restStrengths.sort((a, b) => b - a)];
    setStrengthsAction(sortedStrengths);
    if (selectedBranch === index) setSelectedBranchAction(sortedStrengths.findIndex((s) => s === clampedValue));
  };

  const handleRemoveStrength = (index: number) => {
    if (index === 0) return;
    setStrengthsAction(strengths.filter((_, i) => i !== index));
    setResultsAction(results.filter((_, i) => i !== index));
    if (selectedBranch === index) setSelectedBranchAction(null);
  };

  const getBranchPosition = (index: number) => {
    const sortedBranches = getSortedBranches(); // 获取排序后的分支
    const branch = sortedBranches[index]; // 获取当前分支
    const strength = branch.strength;

    sortedBranches.findIndex((b) => b.strength === 0);
    const zeroPosition = treeHeight / 2;

    if (strength > 0) {
      const positiveBranches = sortedBranches.filter(b => b.strength > 0);
      const branchIndexInPositive = positiveBranches.indexOf(branch);
      const positiveOffset = (positiveBranches.length - branchIndexInPositive) * nodeSpacing;
      return zeroPosition - positiveOffset; // 越小越接近上方
    }

    if (strength < 0) {
      const negativeBranches = sortedBranches.filter(b => b.strength < 0);
      const branchIndexInNegative = negativeBranches.indexOf(branch);
      const negativeOffset = (branchIndexInNegative + 1) * nodeSpacing; // 绝对值越大，偏移量越大
      return zeroPosition + negativeOffset; // 绝对值越大越远离中间
    }

    return zeroPosition;
  };

  const handleDoneClick = () => {
    setSelectedBranchAction(null);
    const [baseBranch, ...restBranches] = strengths;
    const sortedStrengths = [baseBranch, ...restBranches.sort((a, b) => b - a)];
    setStrengthsAction(sortedStrengths);
    const [baseResult] = results;
    const newResults = [baseResult];
    restBranches.forEach(strength => {
      const originalIndex = strengths.indexOf(strength);
      if (originalIndex > 0) {
        newResults.push(results[originalIndex]);
      }
    });
    setResultsAction(newResults);
    setRedrawTrigger((prev) => prev + 1);
  };
  React.useEffect(() => {
    if (!svgRef.current || !results.length) return;
    const width = 800, height = 600, margin = {top: 20, right: 90, bottom: 30, left: 90};
    d3.select(svgRef.current).selectAll("*").remove();
    const treeData: TreeNode = {
      id: "root",
      value: prompt,
      children: results.map((result, i) => ({
        id: `strength-${i}`,
        value: result.strength,
        children: [{id: `output-${i}`, value: result.model_output}]
      }))
    };
    const treeLayout = d3.tree<TreeNode>().size([height - margin.top - margin.bottom, width - margin.left - margin.right]);
    const root = d3.hierarchy<TreeNode>(treeData);
    const nodes = treeLayout(root);
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    svg.selectAll(".link")
      .data(nodes.links())
      .join("path")
      .attr("class", "link")
      .attr("d", d3.linkHorizontal<d3.HierarchyPointLink<TreeNode>, d3.HierarchyPointNode<TreeNode>>()
        .x((d) => d.y)
        .y((d) => d.x))
      .style("fill", "none")
      .style("stroke", "#999")
      .style("stroke-width", 1.5);
    const node = svg.selectAll(".node")
      .data(nodes.descendants())
      .join("g")
      .attr("class", "node")
      .attr("transform", (d) => `translate(${d.y},${d.x})`);
    node.append("circle")
      .attr("r", 5)
      .style("fill", "#fff")
      .style("stroke", "#4CAF50")
      .style("stroke-width", 2);
    node.append("text")
      .attr("dy", ".35em")
      .attr("x", (d) => (d.children ? -13 : 13))
      .style("text-anchor", (d) => (d.children ? "end" : "start"))
      .text((d) => String(d.data.value).substring(0, 30));
  }, [results, prompt, redrawTrigger]);

  return (
    <Paper className="p-4 h-full overflow-hidden flex flex-col">
      <div className="relative w-full overflow-x-auto" style={{height: `${treeHeight}px`}}>
        {selectedBranch !== null && (
          <div 
            className="absolute bg-white p-2 rounded shadow-lg border border-gray-200 z-10"
            style={{
              top: `${getBranchPosition(getSortedBranches().findIndex(b => b.originalIndex === selectedBranch)) + 10}px`,
              left: '120px'
            }}
          >
            <div className="flex items-center gap-2">
              <TextField type="number"
                         value={strengths[selectedBranch]}
                         onChange={(e) => handleStrengthChange(selectedBranch, Number(e.target.value))}
                         size="small"
                         className="w-20"
                         helperText={strengths.length >= 7 ? "Maximum branches reached" : ""}
                         error={strengths.length >= 7}/>
              <Button size="small" onClick={() => handleRemoveStrength(selectedBranch)} color="error" variant="outlined"
                      className="min-w-[40px]">
                <X size={20}/>
              </Button>
              <Button size="small" onClick={handleDoneClick} variant="outlined">Done</Button>
            </div>
          </div>
        )}
        <svg width="100%" height={treeHeight} key={redrawTrigger}>
          <g transform={`translate(100, ${treeHeight / 2 - 80})`}
             className={`cursor-${strengths.length >= 7 ? 'not-allowed' : 'pointer'} opacity-${strengths.length >= 7 ? '50' : '100'}`}>
            <line x1="-55" y1="0" x2="-55" y2="60" stroke="#7FB3D5" strokeWidth="2"/>
            <circle cx="-55" r="15" fill={strengths.length >= 7 ? "#ccc" : "#7FB3D5"}
                    className={`cursor-${strengths.length >= 7 ? 'not-allowed' : 'pointer'}`}
                    onClick={handleAddStrength}/>
            <line x1="-62" y1="0" x2="-48" y2="0" stroke="white" strokeWidth="2"/>
            <line x1="-55" y1="-7" x2="-55" y2="7" stroke="white" strokeWidth="2"/>
          </g>
          <g transform={`translate(100, ${treeHeight / 2})`}>
            <rect x="-80" y="-25" width="50" height="50" rx="10" ry="10" fill="#7FB3D5"/>
            <g transform="translate(-77.5, -25) scale(0.045)">
              <path d="M683.7 922.7h-345c-73.5 0-133.3-59.8-133.3-133.3V459.8c0-73.5 59.8-133.3 133.3-133.3h345c73.5 0 133.3 59.8 133.3 133.3v329.6c0 73.5-59.8 133.3-133.3 133.3z m-345-506.9c-24.3 0-44.1 19.8-44.1 44.1v329.6c0 24.3 19.8 44.1 44.1 44.1h345c24.3 0 44.1-19.8 44.1-44.1V459.8c0-24.3-19.8-44.1-44.1-44.1h-345zM914.3 759.6c-24.6 0-44.6-20-44.6-44.6V534.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6V715c0 24.7-20 44.6-44.6 44.6zM111.7 759.6c-24.6 0-44.6-20-44.6-44.6V534.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6V715c0 24.7-19.9 44.6-44.6 44.6z" fill="#ffffff"/>
              <path d="M511.2 415.8c-24.6 0-44.6-20-44.6-44.6V239.3c0-24.6 20-44.6 44.6-44.6s44.6 20 44.6 44.6v131.9c0 24.6-20 44.6-44.6 44.6z" fill="#ffffff"/>
              <path d="M511.2 276.6c-49.2 0-89.2-40-89.2-89.2s40-89.2 89.2-89.2 89.2 40 89.2 89.2-40 89.2-89.2 89.2z m0-89.2h0.2-0.2z m0 0h0.2-0.2z m0 0h0.2-0.2z m0 0h0.2-0.2z m0 0z m0 0h0.2-0.2z m0 0h0.2-0.2z m0-0.1h0.2-0.2zM399 675.5c-28.1 0-50.9-22.8-50.9-50.9 0-28.1 22.8-50.9 50.9-50.9s50.9 22.8 50.9 50.9c0 28.1-22.8 50.9-50.9 50.9zM622.9 675.5c-28.1 0-50.9-22.8-50.9-50.9 0-28.1 22.8-50.9 50.9-50.9 28.1 0 50.9 22.8 50.9 50.9 0 28.1-22.8 50.9-50.9 50.9z" fill="#ffffff"/>
            </g>
          </g>
          <AnimatePresence>
            {getSortedBranches().map(({strength, originalIndex}) => {
              const y = getBranchPosition(originalIndex);
              return (
                <g key={`branch-${originalIndex}`}>
                  <motion.path initial={{pathLength: 0, opacity: 0}}
                               animate={{pathLength: 1, opacity: 1}}
                               exit={{opacity: 0}}
                               d={`M 75 ${treeHeight / 2} C 100 ${treeHeight / 2}, 100 ${y}, 120 ${y}`}
                               fill="none"
                               stroke={getStrokeColor(strength)}
                              //  strokeWidth={Math.max(4, Math.abs(strength / 40))}
                               strokeWidth={getStrokeWidth(strength)}
                               className={`cursor-${originalIndex === 0 ? 'default' : 'pointer'} opacity-80 transition-all duration-300`}
                               onClick={() => handleBranchClick(originalIndex)}/>
                  {results[originalIndex] && (
                    <motion.foreignObject initial={{opacity: 0, x: 50}}
                                          animate={{opacity: 1, x: 0, scale: 1}}
                                          transition={{duration: 0.3}}
                                          exit={{opacity: 0}}
                                          x="120"
                                          y={y - 40}
                                          width="400"
                                          height="80">
                      <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 w-[350px] max-w-full h-[80px] overflow-hidden">
                        <div 
                          className="text-gray-800 text-sm overflow-y-auto whitespace-pre-wrap break-words scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100 pr-1 h-[60px]"
                          style={{ overflowWrap: 'break-word' }}
                        >
                          {results[originalIndex].model_output}
                        </div>
                      </div>
                    </motion.foreignObject>
                  )}
                </g>
              );
            })}
          </AnimatePresence>
        </svg>
      </div>
    </Paper>
  );
}
