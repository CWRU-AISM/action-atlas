"use client";
import React from "react";
import {Box, Paper} from "@mui/material";
import FeatureExplanation from "./FeatureExplanation";
import {FeatureSectionProps, Point} from "@/types/types";
import TokenHistogram from "./TokenHistogram";
import LogisticsChart from "./LogisticsChart";
import GridMatrixPlot from "./GridMatrixPlot";

export default function FeatureExplorationDashboard({
                                                      featureInfo,
                                                    }: FeatureSectionProps) {
  const [selectedPoints, setSelectedPoints] = React.useState<Point[]>([]);
  const [selectedKeywords, setSelectedKeywords] = React.useState<string[]>([]);
  const [_, setDialogOpen] = React.useState(false);

  const handlePointsSelected = React.useCallback((points: Point[]) => {
    const keywords = points.map((point) => point.keyword);
    setSelectedKeywords(keywords);
    setSelectedPoints(points);
  }, []);

  return (
    <Paper className="flex h-full overflow-y-auto p-2">
      <Box
        className="w-[40%] h-full p-2 relative after:content-[''] after:absolute after:right-0 after:top-[5%] after:h-[90%] after:bg-gray-200 after:opacity-70">
        <Box className="flex flex-col h-full">
          <Box className="flex justify-between items-center flex-none">
            <FeatureExplanation featureInfo={featureInfo}/>
          </Box>
          <Box className="p-2 mt-3 flex-1 h-[250px] gap-2">
            <Box className="h-full w-full flex items-center justify-center">
              <LogisticsChart featureInfo={featureInfo}/>
            </Box>
          </Box>
        </Box>
      </Box>

      <Box className="w-[60%] h-full flex gap-6 overflow-hidden ">
        <Box className="flex-1 min-h-0 h-full w-full">
          <GridMatrixPlot onPointsSelected={handlePointsSelected}/>
        </Box>

        <Box className="flex-none w-[250px] h-full flex flex-col p-2">
          <TokenHistogram selectedPoints={selectedPoints}/>
        </Box>
      </Box>
    </Paper>
  );
}
